import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader

from a_dataset import PointCloudMaskDataset


###############################################################
# (1) 정석적 Diffusion (DDPM) + Multi-step
#     (B,N,4) => (pos + mask), etc.
###############################################################
class Diffusion:
    def __init__(self, num_timesteps=50, beta_start=0.0001, beta_end=0.02, device="cpu"):
        self.num_timesteps = num_timesteps
        self.device = device

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
        x0: (B,N,4)  ex) (x,y,z,mask)
        t:  (B,)
        return: (x_t, noise) -> both (B,N,4)
        """
        x0 = x0.to(self.device)
        t = t.to(self.device)
        B = x0.shape[0]

        noise = torch.randn_like(x0)  # (B,N,4)
        alpha_bar_t = self.alpha_bar[t].view(B, 1, 1)
        x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
        return x_t, noise

    @torch.no_grad()
    def sample_loop(self, model, batch_size=1, num_points=1024):
        """
        Reverse: x_T ~ Normal(0,I) in R^(N,4) => x_0
        returns: (B,N,4)
        """
        x_t = torch.randn(batch_size, num_points, 4, device=self.device)  # (B,N,4)
        for i in reversed(range(self.num_timesteps)):
            t = torch.tensor([i] * batch_size, dtype=torch.long, device=self.device)

            eps_theta = model(x_t, t)  # (B,N,4)
            alpha_t = self.alpha[t].view(batch_size, 1, 1)
            alpha_bar_t = self.alpha_bar[t].view(batch_size, 1, 1)

            if i > 0:
                beta_t = self.beta[t].view(batch_size, 1, 1)
                mean = (1.0 / torch.sqrt(alpha_t)) * (
                        x_t - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * eps_theta
                )
                sigma_t = torch.sqrt(beta_t)
                z = torch.randn_like(x_t)
                x_t = mean + sigma_t * z
            else:
                x_t = (1.0 / torch.sqrt(alpha_t)) * (
                        x_t - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * eps_theta
                )
        return x_t  # (B,N,4)


###############################################################
# (2) 모델: PointNet 스타일 (in=4, out=4)
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
    (x_t, t) => pred_noise, shape=(B,N,4)
    """

    def __init__(self, hidden_dim=128):
        super().__init__()
        # in_channels=4 => (x,y,z,mask)
        self.mlp1 = SharedMLP(4, 64)
        self.mlp2 = SharedMLP(64, 128)

        self.t_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )

        self.fc = nn.Linear(128 + 128, hidden_dim)

        self.decoder1 = nn.Linear(hidden_dim, hidden_dim)
        self.decoder2 = nn.Linear(hidden_dim, 4)  # output=(B,4)

    def forward(self, x, t):
        """
        x: (B,N,4)
        t: (B,)
        return: (B,N,4)
        """
        B, N, dim_in = x.shape  # dim_in=4
        # (B,4,N)
        x_trans = x.permute(0, 2, 1)
        x_trans = F.relu(self.mlp1(x_trans))  # (B,64,N)
        x_trans = F.relu(self.mlp2(x_trans))  # (B,128,N)
        x_global = torch.max(x_trans, dim=-1)[0]  # (B,128)

        t = t.float().unsqueeze(-1)  # (B,1)
        t_feat = self.t_embed(t)  # (B,128)

        feat = torch.cat([x_global, t_feat], dim=-1)  # (B,256)
        feat = F.relu(self.fc(feat))  # (B, hidden_dim)

        d = F.relu(self.decoder1(feat))  # (B, hidden_dim)
        d = self.decoder2(d)  # (B,4)

        # 모든 점에 동일한 noise => expand
        d = d.unsqueeze(1).expand(-1, N, 4)  # (B,N,4)
        return d


###############################################################
# (3) 학습 함수
###############################################################
def train_diffusion_model(model, diffusion, dataloader, optimizer,
                          device="cpu", num_epochs=10, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for pos_mask_4d in dataloader:
            # pos_mask_4d: (B,N,4)
            pos_mask_4d = pos_mask_4d.to(device)
            B = pos_mask_4d.shape[0]

            # random timestep
            t = torch.randint(0, diffusion.num_timesteps, (B,), dtype=torch.long, device=device)
            x_t, noise = diffusion.add_noise(pos_mask_4d, t)  # (B,N,4)

            pred_noise = model(x_t, t)  # (B,N,4)

            loss = F.mse_loss(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch + 1}/{num_epochs}] Loss: {avg_loss:.4f}")

        ckpt_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")


###############################################################
# (4) 샘플링 함수
###############################################################
@torch.no_grad()
def sample_diffusion_model(model, diffusion, device="cpu",
                           checkpoint_path=None, batch_size=1, num_points=1024):
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    model.to(device)

    x_0 = diffusion.sample_loop(model, batch_size=batch_size, num_points=num_points)
    return x_0  # (B,N,4)


###############################################################
# (5) 메인 실행 예시
###############################################################
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    diffusion = Diffusion(num_timesteps=50, device=device)

    dataset = PointCloudMaskDataset(
        pointcloud_dir="pointclouds",
        num_points=1024,
        default_mask=1.0
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = DiffusionUNet(hidden_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_diffusion_model(model, diffusion, dataloader, optimizer,
                          device=device, num_epochs=10)

    # sample
    # x_0 = sample_diffusion_model(model, diffusion, device=device,
    #     checkpoint_path="checkpoints/model_epoch_10.pth",
    #     batch_size=1, num_points=1024)
    # print("x_0 shape:", x_0.shape)  # (1,1024,4)

    print("Done with (B,N,4) approach!")
