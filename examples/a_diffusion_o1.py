import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader

from a_dataset import PointCloudDataset, pad_collate_fn  # vertices/mask는 무시해도 됨


# (기존 custom_collate_fn에서 vertices, mask가 생기는데 여기선 안 써도 됨)

########################################
# 1. Diffusion 클래스 (노이즈 추가, 샘플링)
########################################
class Diffusion:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cuda'):
        self.num_timesteps = num_timesteps
        self.device = device

        beta = torch.linspace(beta_start, beta_end, num_timesteps)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        # GPU로 옮기기
        self.alpha_bar = alpha_bar.to(device)
        self.alpha = alpha.to(device)
        self.beta = beta.to(device)

    def add_noise(self, x0, t):
        """
        x0: (B, N, 3)
        t:  (B,) long
        """
        # 혹시 t가 다른 디바이스라면 통일
        t = t.to(self.alpha_bar.device)
        x0 = x0.to(self.alpha_bar.device)

        batch_size = x0.shape[0]
        noise = torch.randn_like(x0)  # (B, N, 3)

        alpha_bar_t = self.alpha_bar[t].view(batch_size, 1, 1)  # (B,1,1)
        x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
        return x_t, noise

    @torch.no_grad()
    def sample_loop(self, model, batch_size=1, num_points=1024, device='cpu'):
        """
        Reverse process: x_T ~ N(0,I) 에서 시작 -> x_0로 수렴
        여기선 DDPM 공식의 간단 버전만 구현
        """
        x_t = torch.randn(batch_size, num_points, 3, device=device)  # 초기 노이즈
        for i in reversed(range(self.num_timesteps)):
            t = torch.tensor([i] * batch_size, device=device, dtype=torch.long)
            epsilon_theta = model(x_t, t)  # (B, N, 3) 예측 노이즈

            alpha_t = self.alpha[t].to(device).view(batch_size, 1, 1)
            alpha_bar_t = self.alpha_bar[t].to(device).view(batch_size, 1, 1)
            if i > 0:
                beta_t = self.beta[t].to(device).view(batch_size, 1, 1)
                # DDPM 공식 (단순화 버전)
                mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * epsilon_theta)
                sigma_t = torch.sqrt(beta_t)
                z = torch.randn_like(x_t)
                x_t = mean + sigma_t * z
            else:
                # t=0 단계
                x_t = (1.0 / torch.sqrt(alpha_t)) * (x_t - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * epsilon_theta)
        return x_t


########################################
# 2. PointNet 스타일 모델
########################################
class SharedMLP(nn.Module):
    """Conv1d( in_channels -> out_channels, kernel_size=1 )"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        # x: (B, in_channels, N)
        return self.conv(x)


class DiffusionPointNet(nn.Module):
    """
    (x_t, t) -> 예측 노이즈 epsilon_theta
    - x_t: (B,N,3)
    - t:   (B,)  int
    return: (B,N,3)
    """

    def __init__(self, hidden_dim=128):
        super().__init__()

        # encoder: point feature
        self.mlp1 = SharedMLP(3, 64)
        self.mlp2 = SharedMLP(64, 128)

        # Timestep 임베딩
        self.t_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
        )

        # bottleneck
        self.fc = nn.Linear(128 + 128, hidden_dim)  # point feature + t feature

        # decoder
        self.decoder1 = nn.Linear(hidden_dim, hidden_dim)
        self.decoder2 = nn.Linear(hidden_dim, 3)

    def forward(self, x, t):
        B, N, _ = x.shape

        # PointNet 인코딩
        x_trans = x.permute(0, 2, 1)  # (B,3,N)
        x_trans = F.relu(self.mlp1(x_trans))  # (B,64,N)
        x_trans = F.relu(self.mlp2(x_trans))  # (B,128,N)
        x_global = torch.max(x_trans, dim=-1)[0]  # (B,128) global feature

        # Timestep 임베딩
        t = t.float().unsqueeze(-1)  # (B,1)
        t_feat = self.t_embed(t)  # (B,128)

        # Concat
        feat = torch.cat([x_global, t_feat], dim=-1)  # (B,256)
        feat = F.relu(self.fc(feat))  # (B, hidden_dim)

        # Decoder -> 예측 노이즈 (B,N,3)
        d = F.relu(self.decoder1(feat))  # (B, hidden_dim)
        d = self.decoder2(d)  # (B, 3)

        # 모든 점에 동일한 noise vector를 예측 (간단 버전)
        d = d.unsqueeze(1).expand(-1, N, 3)  # (B,N,3)
        return d


########################################
# 3. 학습 함수 (MSE로 노이즈 예측)
########################################
def train_diffusion_model(model, dataloader, diffusion, optimizer, device='cpu', num_epochs=10,
                          checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for point_cloud, _vertices, _mask in dataloader:
            # 여기선 vertices, mask 무시
            point_cloud = point_cloud.to(device)  # (B,N,3)

            # 랜덤 타임스텝
            t = torch.randint(0, diffusion.num_timesteps, (point_cloud.shape[0],), dtype=torch.long, device=device)
            x_t, noise = diffusion.add_noise(point_cloud, t)  # (B,N,3)

            # 노이즈 예측
            pred_noise = model(x_t, t)  # (B,N,3)

            # MSE
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Checkpoint 저장
        ckpt_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")


########################################
# 4. 샘플링 함수 (reverse process)
########################################
@torch.no_grad()
def sample_diffusion_model(model, diffusion, device='cpu', checkpoint_path=None, batch_size=1, num_points=1024):
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    model.to(device)

    # Reverse process
    x_0 = diffusion.sample_loop(model, batch_size=batch_size, num_points=num_points, device=device)
    return x_0  # (B,N,3)


########################################
# 5. 메인 실행
########################################
if __name__ == "__main__":
    # (1) Diffusion 객체 생성
    diffusion = Diffusion(num_timesteps=50)  # 50 스텝만 예시

    # (2) Dataset/Dataloader
    # - vertices, mask 무시
    dataset = PointCloudDataset(
        pointcloud_dir="pointclouds",
        vertices_dir="vertices",  # 있으나 안 씀
        num_points=1024
    )
    # pad_collate_fn을 써도 되지만, vertices는 안 쓸 거라서 문제가 없으면 그냥 써도 됨.
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=pad_collate_fn)

    # (3) 모델/옵티마 준비
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DiffusionPointNet(hidden_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # 학습률 낮춤

    # (4) 학습
    train_diffusion_model(model, dataloader, diffusion, optimizer, device=device, num_epochs=10)

    # (5) 샘플링
    sampled_points = sample_diffusion_model(model, diffusion, device=device,
                                            checkpoint_path="checkpoints/model_epoch_10.pth", batch_size=1,
                                            num_points=1024)
    print("Sampled shape:", sampled_points.shape)
