import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader

from examples.a_dataset import PointCloudDataset, PointCloudOnlyDataset


########################################################
# 1. Diffusion 클래스 (pos+mask 4D, DDPM)
########################################################
class Diffusion:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device="cpu"):
        self.num_timesteps = num_timesteps
        self.device = device

        # Beta schedule
        beta = torch.linspace(beta_start, beta_end, num_timesteps)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        self.register_buffers(beta, alpha, alpha_bar)

    def register_buffers(self, beta, alpha, alpha_bar):
        self.beta = beta.to(self.device)
        self.alpha = alpha.to(self.device)
        self.alpha_bar = alpha_bar.to(self.device)

    def add_noise(self, x0, t):
        """
        x0: (B,N,4)  => (pos: x0[:,:,:3], mask: x0[:,:,3])
        t:  (B,)
        return x_t, noise (둘 다 (B,N,4))
        """
        x0 = x0.to(self.device)
        t = t.to(self.device)
        B = x0.shape[0]

        noise = torch.randn_like(x0)  # (B,N,4)
        alpha_bar_t = self.alpha_bar[t].view(B, 1, 1)
        # sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t)*noise
        # mask 부분도 동일하게 노이즈 추가
        x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
        return x_t, noise

    @torch.no_grad()
    def sample_loop(self, model, batch_size=1, num_points=1024):
        """
        Reverse process:
          x_T ~ Normal(0,I) in R^{(N,4)} => x_0
        """
        x_t = torch.randn(batch_size, num_points, 4, device=self.device)
        for i in reversed(range(self.num_timesteps)):
            t = torch.tensor([i] * batch_size, dtype=torch.long, device=self.device)

            # 예측 노이즈 (B,N,4)
            eps_theta = model(x_t, t)

            alpha_t = self.alpha[t].view(batch_size, 1, 1)
            alpha_bar_t = self.alpha_bar[t].view(batch_size, 1, 1)
            if i > 0:
                beta_t = self.beta[t].view(batch_size, 1, 1)
                mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * eps_theta)
                sigma_t = torch.sqrt(beta_t)
                z = torch.randn_like(x_t)
                x_t = mean + sigma_t * z
            else:
                x_t = (1.0 / torch.sqrt(alpha_t)) * (x_t - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * eps_theta)

        return x_t  # (B,N,4) => pos+mask


########################################################
# 2. PointNet + MaskBranch (Input 4D => Output 4D)
########################################################
class SharedMLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # x: (B, in_channels, N)
        return self.conv(x)


class DiffusionPointNet(nn.Module):
    """
    (x_t, t) => (pred_noise)  shape: (B,N,4)
    - x_t: (B,N,4) (xyz + mask)
    - t:   (B,)
    """

    def __init__(self, hidden_dim=128):
        super().__init__()

        # Encoder
        #   input_dim=4
        self.mlp1 = SharedMLP(4, 64)
        self.mlp2 = SharedMLP(64, 128)

        # Timestep embed
        self.t_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )

        # Bottleneck
        self.fc = nn.Linear(128 + 128, hidden_dim)

        # Decoder
        self.decoder1 = nn.Linear(hidden_dim, hidden_dim)
        self.decoder2 = nn.Linear(hidden_dim, 4)  # predict noise in (pos+mask) => 4D

    def forward(self, x, t):
        """
        x: (B,N,4)
        t: (B,)
        return: eps_pred: (B,N,4)
        """
        B, N, dim = x.shape
        # PointNet Enc
        x_trans = x.permute(0, 2, 1)  # (B,4,N)
        x_trans = F.relu(self.mlp1(x_trans))  # (B,64,N)
        x_trans = F.relu(self.mlp2(x_trans))  # (B,128,N)
        x_global = torch.max(x_trans, dim=-1)[0]  # (B,128)

        # Timestep
        t = t.float().unsqueeze(-1)  # (B,1)
        t_feat = self.t_embed(t)  # (B,128)

        feat = torch.cat([x_global, t_feat], dim=-1)  # (B,256)
        feat = F.relu(self.fc(feat))  # (B, hidden_dim)

        d = F.relu(self.decoder1(feat))  # (B, hidden_dim)
        d = self.decoder2(d)  # (B,4)

        # 모든 점 동일한 noise => expand
        d = d.unsqueeze(1).expand(-1, N, 4)  # (B,N,4)
        return d


########################################################
# 3. 학습 함수 (MSE on pos+mask noise)
########################################################
def train_diffusion_model(model, diffusion, dataloader, optimizer, device="cpu", num_epochs=10, ckpt_dir="checkpoints"):
    os.makedirs(ckpt_dir, exist_ok=True)
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_data in dataloader:
            # batch_data = (pos_mask_4d, ???)
            # ex) pos_mask_4d: (B,N,4)
            #    where last dim is mask=1 or 0
            pos_mask_4d = batch_data[0].to(device)  # (B,N,4)

            B = pos_mask_4d.shape[0]
            t = torch.randint(0, diffusion.num_timesteps, (B,), dtype=torch.long, device=device)

            x_t, noise = diffusion.add_noise(pos_mask_4d, t)  # both (B,N,4)

            eps_pred = model(x_t, t)  # (B,N,4)

            loss = F.mse_loss(eps_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}]  Loss: {avg_loss:.4f}")

        # save ckpt
        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved: {ckpt_path}")


########################################################
# 4. 샘플링 (reverse)
########################################################
@torch.no_grad()
def sample_diffusion_model(model, diffusion, device="cpu", ckpt_path=None, batch_size=1, num_points=1024):
    if ckpt_path is not None:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    model.to(device)

    x_0 = diffusion.sample_loop(model, batch_size=batch_size, num_points=num_points)  # (B,N,4)
    return x_0  # pos+mask


###############################################################
# 메인 실행 예시
###############################################################
if __name__ == "__main__":
    # 1) Diffusion 객체
    device = "cuda" if torch.cuda.is_available() else "cpu"
    diffusion = Diffusion(num_timesteps=50, device=device)

    # 2) Dataset & DataLoader
    dataset = PointCloudOnlyDataset(pointcloud_dir="pointclouds", num_points=1024)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

    # 3) 모델 & 옵티마
    model = DiffusionPointNet(hidden_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 4) 학습 (사용자 데이터셋이 준비된 상태 가정)
    train_diffusion_model(model, diffusion, dataloader, optimizer, device=device, num_epochs=100)

    # 5) 샘플링
    # sampled = sample_diffusion_model(model, diffusion, device=device, checkpoint_path="checkpoints/model_epoch_10.pth", batch_size=1, num_points=1024)
    # print("Sampled shape:", sampled.shape)

    print("Diffusion code replaced successfully.")
