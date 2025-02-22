import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader

from a_dataset import PointCloudDataset, pad_collate_fn, custom_collate_fn


###############################################################
# (1) 정석적 Diffusion (DDPM 스타일) + Multi-step Reverse
#     포인트 클라우드를 다루는 예시 코드
#
# - 노이즈(\epsilon) 예측 (MSE) 방식
# - Multi-step reverse (sample_loop)로 노이즈 제거
# - Chamfer Distance / mask 사용 X
# - "vertices" 불필요, "point_cloud"만 사용
#
# 이 코드는 기존의 "DiffusionUNet + Chamfer" 코드를 전면 교체합니다.
###############################################################


###############################################################
# Diffusion 클래스: 베타 스케줄, add_noise, sample_loop
###############################################################
class Diffusion:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device="cpu"):
        """정석적인 Diffusion(DDPM) 식을 간단히 구현"""
        self.num_timesteps = num_timesteps
        self.device = device

        # Beta 스케줄 (선형)
        beta = torch.linspace(beta_start, beta_end, num_timesteps)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        self.register_buffers(beta, alpha, alpha_bar)

    def register_buffers(self, beta, alpha, alpha_bar):
        # PyTorch에서 학습되지 않는 텐서를 등록하기 위해
        self.beta = beta.to(self.device)
        self.alpha = alpha.to(self.device)
        self.alpha_bar = alpha_bar.to(self.device)

    def add_noise(self, x0, t):
        """
        x0: (B,N,3) 원본 포인트 클라우드
        t:  (B,)    int형 텐서
        return: (x_t, noise)
        """
        x0 = x0.to(self.device)
        t = t.to(self.device)
        batch_size = x0.shape[0]

        noise = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bar[t].view(batch_size, 1, 1)
        x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
        return x_t, noise

    @torch.no_grad()
    def sample_loop(self, model, batch_size=1, num_points=1024):
        """Reverse process: x_T ~ N(0,I)에서 시작하여 x_0로 수렴"""
        # 완전 노이즈 상태 x_T
        x_t = torch.randn(batch_size, num_points, 3, device=self.device)
        for i in reversed(range(self.num_timesteps)):
            t = torch.tensor([i]*batch_size, dtype=torch.long, device=self.device)

            # 모델로 노이즈 예측 (B,N,3)
            epsilon_theta = model(x_t, t)

            alpha_t = self.alpha[t].view(batch_size, 1, 1)
            alpha_bar_t = self.alpha_bar[t].view(batch_size, 1, 1)
            if i > 0:
                beta_t = self.beta[t].view(batch_size, 1, 1)
                # DDPM 식 (단순화)
                mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - (1 - alpha_t)/torch.sqrt(1 - alpha_bar_t)*epsilon_theta)
                sigma_t = torch.sqrt(beta_t)
                z = torch.randn_like(x_t)
                x_t = mean + sigma_t * z
            else:
                # t=0이면 최종 결과
                x_t = (1.0 / torch.sqrt(alpha_t)) * (x_t - (1 - alpha_t)/torch.sqrt(1 - alpha_bar_t)*epsilon_theta)
        return x_t


###############################################################
# 모델: PointNet 스타일 (간단 버전)
# x_t, t -> 예측 노이즈 (B,N,3)
###############################################################

class SharedMLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        # x: (B, in_channels, N)
        return self.conv(x)

class DiffusionUNet(nn.Module):
    """
    여기선 이름은 UNet이지만 실제론 간단한 PointNet 구조를 예시로 함.
    """
    def __init__(self, hidden_dim=128):
        super().__init__()
        # encoder: point feature
        self.mlp1 = SharedMLP(3, 64)
        self.mlp2 = SharedMLP(64, 128)

        # t 임베딩
        self.t_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )

        # bottleneck
        self.fc = nn.Linear(128+128, hidden_dim)  # point feat + t feat

        # decoder
        self.decoder1 = nn.Linear(hidden_dim, hidden_dim)
        self.decoder2 = nn.Linear(hidden_dim, 3)

    def forward(self, x, t):
        """
        x: (B,N,3)  --> noisy pc
        t: (B,)     --> timestep
        return: pred_noise (B,N,3)
        """
        B, N, _ = x.shape

        # pointnet enc
        x_trans = x.permute(0,2,1)  # (B,3,N)
        x_trans = F.relu(self.mlp1(x_trans))  # (B,64,N)
        x_trans = F.relu(self.mlp2(x_trans))  # (B,128,N)
        x_global = torch.max(x_trans, dim=-1)[0]  # (B,128)

        # t embedding
        t = t.float().unsqueeze(-1)  # (B,1)
        t_feat = self.t_embed(t)     # (B,128)

        feat = torch.cat([x_global, t_feat], dim=-1)  # (B,256)
        feat = F.relu(self.fc(feat))                  # (B, hidden_dim)

        d = F.relu(self.decoder1(feat))  # (B, hidden_dim)
        d = self.decoder2(d)            # (B,3)

        # 모든 점에 동일한 noise를 예측 (간단)
        d = d.unsqueeze(1).expand(-1, N, 3)  # (B,N,3)
        return d

###############################################################
# 학습 함수: 노이즈 예측 (MSE) -> Diffusion 정석
###############################################################
def train_diffusion_model(model, diffusion, dataloader, optimizer, device="cpu", num_epochs=10, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_data in dataloader:
            # batch_data = (point_cloud, vertices, mask)...
            # 여기서는 point_cloud만 사용.
            # 만약 a_dataset.py에서 (point_cloud)만 반환하도록 수정하면 더 깔끔.
            # 예: point_cloud = batch_data[0]
            point_cloud = batch_data[0]

            point_cloud = point_cloud.to(device)
            B = point_cloud.shape[0]

            # 랜덤 타임스텝
            t = torch.randint(0, diffusion.num_timesteps, (B,), dtype=torch.long, device=device)
            x_t, noise = diffusion.add_noise(point_cloud, t)

            pred_noise = model(x_t, t)

            # MSE between actual noise and predicted noise
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # checkpoint
        # ckpt_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
        # torch.save(model.state_dict(), ckpt_path)
        # print(f"Checkpoint saved: {ckpt_path}")

###############################################################
# 샘플링 함수
###############################################################
@torch.no_grad()
def sample_diffusion_model(model, diffusion, device="cpu", checkpoint_path=None, batch_size=1, num_points=1024):
    # checkpoint 로드
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    model.to(device)

    # multi-step reverse
    x_0 = diffusion.sample_loop(model, batch_size=batch_size, num_points=num_points)
    return x_0

###############################################################
# 메인 실행 예시
###############################################################
if __name__ == "__main__":
    # 1) Diffusion 객체
    device = "cuda" if torch.cuda.is_available() else "cpu"
    diffusion = Diffusion(num_timesteps=50, device=device)

    # 2) Dataset & DataLoader
    dataset = PointCloudDataset(pointcloud_dir="pointclouds", vertices_dir="vertices", num_points=1024)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)

    # 3) 모델 & 옵티마
    model = DiffusionUNet(hidden_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 4) 학습 (사용자 데이터셋이 준비된 상태 가정)
    train_diffusion_model(model, diffusion, dataloader, optimizer, device=device, num_epochs=100)

    # 5) 샘플링
    # sampled = sample_diffusion_model(model, diffusion, device=device, checkpoint_path="checkpoints/model_epoch_10.pth", batch_size=1, num_points=1024)
    # print("Sampled shape:", sampled.shape)

    print("Diffusion code replaced successfully.")
