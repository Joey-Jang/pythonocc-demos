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

        # Improved time embedding with wider network
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2)
        )

        # Improved network architecture with residual connections
        self.net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3 + hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.1)
            ),
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.1)
            ),
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.1)
            )
        ])

        self.final_layer = nn.Linear(hidden_dim, 3)

    def forward(self, x, point_cloud, t):
        # Normalize input point cloud and vertices
        point_cloud = F.normalize(point_cloud, p=2, dim=-1)
        x = F.normalize(x, p=2, dim=-1)

        # Encode point cloud
        point_cloud_features = self.encoder(point_cloud)

        # Time embedding
        t_emb = self.time_embed(t)

        # Combine features
        batch_size = x.shape[0]
        actual_num_vertices = x.shape[1]

        point_cloud_features = point_cloud_features.unsqueeze(1).expand(-1, actual_num_vertices, -1)
        t_emb = t_emb.unsqueeze(1).expand(-1, actual_num_vertices, -1)

        # Input preparation
        h = torch.cat([x, point_cloud_features, t_emb], dim=-1)

        # Residual network with skip connections
        for layer in self.net:
            h_prev = h
            h = layer(h)
            if h.shape == h_prev.shape:  # Add skip connection if shapes match
                h = h + h_prev

        # Final prediction
        noise_pred = self.final_layer(h)
        return noise_pred


class VertexExtractionDiffusion:
    def __init__(self, num_vertices=5000, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.num_vertices = num_vertices

        # Improved noise schedule
        self.beta = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        # Add sqrt of alpha and beta terms for efficiency
        self.sqrt_alpha = torch.sqrt(self.alpha)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)

        self.model = DiffusionModel(num_vertices=num_vertices)

    def add_noise(self, x, t):
        """
        Add noise to the input vertices based on the diffusion timestep.

        Args:
            x (torch.Tensor): Input vertices of shape (batch_size, num_vertices, 3)
            t (torch.Tensor): Timesteps of shape (batch_size,)

        Returns:
            tuple: (noisy_vertices, noise) tensors
        """
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1)
        noise = torch.randn_like(x)

        # Calculate noisy vertices using the pre-computed sqrt terms
        noisy_x = torch.sqrt(alpha_bar_t) * x + \
                  torch.sqrt(1 - alpha_bar_t) * noise

        return noisy_x, noise

    def training_step(self, vertices, point_cloud):
        batch_size = vertices.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,))

        # Normalize vertices
        vertices = F.normalize(vertices, p=2, dim=-1)

        # Add noise to vertices
        noisy_vertices, noise = self.add_noise(vertices, t)

        # Predict noise
        noise_pred = self.model(noisy_vertices, point_cloud, t.float().unsqueeze(-1))

        # Weighted MSE loss for noise prediction
        noise_weight = (1 - self.alpha_bar[t]).view(-1, 1, 1)
        noise_loss = F.mse_loss(noise_pred * noise_weight, noise * noise_weight)

        # Improved Chamfer distance calculation
        with torch.no_grad():
            alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1)
            denoised_vertices = (noisy_vertices - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
            chamfer_loss, _ = chamfer_distance(denoised_vertices, vertices)

        # Dynamic loss weighting
        normalized_chamfer_loss = chamfer_loss / vertices.shape[1]
        chamfer_weight = min(0.1, 1.0 / (1.0 + normalized_chamfer_loss.item()))

        total_loss = noise_loss + chamfer_weight * normalized_chamfer_loss

        return {
            'total_loss': total_loss,
            'noise_loss': noise_loss,
            'chamfer_loss': chamfer_loss,
            'chamfer_weight': chamfer_weight
        }


def train_model(model, train_loader, optimizer, num_epochs=100, checkpoint_dir='checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Improved learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-4,  # Lower maximum learning rate
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
        pct_start=0.1,  # Shorter warmup
        div_factor=25,  # Smaller initial learning rate
        final_div_factor=1e4,  # Lower final learning rate
        anneal_strategy='cos'
    )

    # Initialize exponential moving averages
    ema_loss = None
    ema_chamfer = None
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.model.train()
        epoch_losses = []

        for batch_idx, (vertices, point_cloud) in enumerate(train_loader):
            optimizer.zero_grad()

            # Gradient accumulation for larger effective batch size
            num_accumulation_steps = 4
            losses = model.training_step(vertices, point_cloud)
            losses['total_loss'] = losses['total_loss'] / num_accumulation_steps
            losses['total_loss'].backward()

            if (batch_idx + 1) % num_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

            # Update moving averages
            if ema_loss is None:
                ema_loss = losses['total_loss'].item()
                ema_chamfer = losses['chamfer_loss'].item()
            else:
                ema_loss = 0.95 * ema_loss + 0.05 * losses['total_loss'].item()
                ema_chamfer = 0.95 * ema_chamfer + 0.05 * losses['chamfer_loss'].item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, "
                      f"EMA Loss: {ema_loss:.4f}, "
                      f"EMA Chamfer: {ema_chamfer:.4f}, "
                      f"Chamfer Weight: {losses['chamfer_weight']:.4f}, "
                      f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

            epoch_losses.append(losses['total_loss'].item() * num_accumulation_steps)

        # Save checkpoints
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
        }

        torch.save(checkpoint, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt'))

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_model.pt'))

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
        lr=5e-5,  # 초기 learning rate 감소
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
