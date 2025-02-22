import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.loss import chamfer_distance

from a_dataloader import create_dataloader


class PointCloudEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=256):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, 1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # BatchNorm 추가
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        # x: (batch_size, num_points, 3)
        x = x.transpose(1, 2)  # (batch_size, 3, num_points)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = torch.max(x, 2)[0]  # (batch_size, hidden_dim)
        return x


class DiffusionModel(nn.Module):
    def __init__(self, num_vertices=5000, hidden_dim=256):
        super().__init__()
        self.num_vertices = num_vertices
        self.encoder = PointCloudEncoder()

        # Diffusion model components
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # self.net = nn.Sequential(
        #     nn.Linear(3 + hidden_dim * 2, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, 3)
        # )
        self.net = nn.Sequential(
            nn.Linear(3 + hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),  # Dropout 추가

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),

            nn.Linear(hidden_dim // 2, 3)
        )

    def forward(self, x, point_cloud, t):
        # x: noisy vertices (batch_size, num_vertices, 3)
        # point_cloud: (batch_size, num_points, 3)
        # t: timesteps (batch_size, 1)

        # Encode point cloud
        point_cloud_features = self.encoder(point_cloud)  # (batch_size, hidden_dim)

        # Time embedding
        t_emb = self.time_embed(t)  # (batch_size, hidden_dim)

        # Combine features
        batch_size = x.shape[0]
        actual_num_vertices = x.shape[1]  # 실제 입력된 정점의 수 사용

        point_cloud_features = point_cloud_features.unsqueeze(1).expand(-1, actual_num_vertices, -1)
        t_emb = t_emb.unsqueeze(1).expand(-1, actual_num_vertices, -1)

        # Predict noise
        x_input = torch.cat([x, point_cloud_features, t_emb], dim=-1)
        noise_pred = self.net(x_input)

        return noise_pred


class VertexExtractionDiffusion:
    def __init__(self, num_vertices=5000, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.num_vertices = num_vertices

        # Define noise schedule
        self.beta = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        self.model = DiffusionModel(num_vertices=num_vertices)

    def add_noise(self, x, t):
        alpha_bar_t = self.alpha_bar[t]
        noise = torch.randn_like(x)
        noisy_x = torch.sqrt(alpha_bar_t).view(-1, 1, 1) * x + \
                  torch.sqrt(1 - alpha_bar_t).view(-1, 1, 1) * noise
        return noisy_x, noise

    def training_step(self, vertices, point_cloud):
        batch_size = vertices.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,))

        # Add noise to vertices
        noisy_vertices, noise = self.add_noise(vertices, t)

        # Predict noise
        noise_pred = self.model(noisy_vertices, point_cloud, t.float().unsqueeze(-1))

        # Calculate noise prediction loss
        noise_loss = F.mse_loss(noise_pred, noise)

        # Calculate Chamfer distance between denoised vertices and point cloud
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1)
        denoised_vertices = (noisy_vertices - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
        chamfer_loss, _ = chamfer_distance(denoised_vertices, vertices)

        normalized_chamfer_loss = chamfer_loss / vertices.shape[1]  # 정점 수로 정규화
        total_loss = noise_loss + 0.1 * normalized_chamfer_loss  # 가중치 상향 조정

        return {
            'total_loss': total_loss,
            'noise_loss': noise_loss,
            'chamfer_loss': chamfer_loss
        }

    @torch.no_grad()
    def sample(self, point_cloud, device='cuda'):
        self.model.eval()
        batch_size = point_cloud.shape[0]

        # Start from random noise
        x = torch.randn(batch_size, self.num_vertices, 3).to(device)

        # Gradually denoise
        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.ones(batch_size, 1).to(device) * t

            # Predict noise
            noise_pred = self.model(x, point_cloud, t_batch)

            # Update x
            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_bar[t]
            beta_t = self.beta[t]

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0

            x = (1 / torch.sqrt(alpha_t)) * (
                    x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred
            ) + torch.sqrt(beta_t) * noise

        return x


# Training loop example
def train_model(model, train_loader, optimizer, num_epochs=100, checkpoint_dir='checkpoints'):
    # 체크포인트 디렉토리 생성
    os.makedirs(checkpoint_dir, exist_ok=True)

    # OneCycleLR 스케줄러 설정
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-9,  # 최대 learning rate 감소
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
        pct_start=0.2,  # warmup 비율 감소
        div_factor=10,  # 초기 lr = max_lr/10
        final_div_factor=1e3,  # 최종 lr = max_lr/1000
        anneal_strategy='cos'
    )

    best_loss = float('inf')

    running_avg_loss = 0.0
    running_avg_chamfer = 0.0

    for epoch in range(num_epochs):
        epoch_losses = []

        for batch_idx, (vertices, point_cloud) in enumerate(train_loader):
            optimizer.zero_grad()
            losses = model.training_step(vertices, point_cloud)

            # Exponential moving average로 loss 변화 추적
            running_avg_loss = 0.95 * running_avg_loss + 0.05 * losses['total_loss'].item()
            running_avg_chamfer = 0.95 * running_avg_chamfer + 0.05 * losses['chamfer_loss'].item()

            if batch_idx % 100 == 0:
                print(f"Running avg loss: {running_avg_loss:.4f}, Running avg chamfer: {running_avg_chamfer:.4f}")

            total_loss = losses['total_loss']
            total_loss.backward()

            # Gradient clipping 추가
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=0.1)

            optimizer.step()
            scheduler.step()

            epoch_losses.append(total_loss.item())

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, "
                      f"Total Loss: {losses['total_loss']:.4f}, "
                      f"Noise Loss: {losses['noise_loss']:.4f}, "
                      f"Chamfer Loss: {losses['chamfer_loss']:.4f}, "
                      f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

        # 에포크 평균 손실 계산
        avg_loss = sum(epoch_losses) / len(epoch_losses)

        # 모델 저장
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
        }

        # 매 에포크마다 체크포인트 저장
        torch.save(checkpoint,
                   os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt'))

        # 최고 성능 모델 저장
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(checkpoint,
                       os.path.join(checkpoint_dir, 'best_model.pt'))

        print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")


def load_checkpoint(model, optimizer,
                    # scheduler,
                    checkpoint_path):
    """저장된 체크포인트를 로드하는 함수"""
    checkpoint = torch.load(checkpoint_path)
    model.model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    print(f"Loaded checkpoint from epoch {epoch} with loss {loss:.4f}")
    return epoch, loss


# Usage example
if __name__ == "__main__":
    # Initialize model
    diffusion = VertexExtractionDiffusion(num_vertices=5000)
    # optimizer = torch.optim.Adam(diffusion.model.parameters(), lr=1e-4)
    optimizer = torch.optim.AdamW(
        diffusion.model.parameters(),
        lr=5e-9,  # 초기 learning rate 감소
        weight_decay=0.1,  # weight decay 증가
        betas=(0.9, 0.999)
    )

    # Assuming you have a DataLoader with (vertices, point_cloud) pairs
    # 데이터 디렉토리 설정
    vertex_dir = "vertices"
    pointcloud_dir = "pointclouds"

    # DataLoader 생성
    train_loader = create_dataloader(
        vertex_dir=vertex_dir,
        pointcloud_dir=pointcloud_dir,
        batch_size=3,
        num_workers=1
    )

    # Train model
    # 학습 시작
    train_model(diffusion, train_loader, optimizer,
                num_epochs=100,
                checkpoint_dir='checkpoints')

    # 저장된 모델 로드
    # epoch, loss = load_checkpoint(diffusion, optimizer,
    #                               'checkpoints/best_model.pt')

    # Sample vertices from point cloud
    # point_cloud = ...
    # vertices = diffusion.sample(point_cloud)
