import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader

from a_dataset import PointCloudDataset, custom_collate_fn, pad_collate_fn
from a_diffusion import Diffusion


class DiffusionUNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128):
        super(DiffusionUNet, self).__init__()

        self.encoder1 = nn.Linear(input_dim + 1, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # 🔹 BatchNorm 추가
        self.encoder2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)  # 🔹 BatchNorm 추가

        self.bottleneck = nn.Linear(hidden_dim, hidden_dim)
        self.bn_bottleneck = nn.BatchNorm1d(hidden_dim)  # 🔹 BatchNorm 추가

        self.decoder1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)  # 🔹 BatchNorm 추가
        self.decoder2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, t):
        """ Forward pass with timestep embedding """
        t_emb = t.view(-1, 1, 1).expand(-1, x.shape[1], 1)
        x = torch.cat((x, t_emb), dim=-1)

        x = x.view(-1, x.shape[-1])  # Flatten batch & points

        x = F.relu(self.bn1(self.encoder1(x)))  # 🔹 BatchNorm 적용
        x = F.relu(self.bn2(self.encoder2(x)))

        x = F.relu(self.bn_bottleneck(self.bottleneck(x)))

        x = F.relu(self.bn3(self.decoder1(x)))
        x = self.decoder2(x)

        x = x.view(-1, 1024, 3)  # Reshape back
        return x


# Define loss function
def chamfer_distance(x, y, mask):
    """ Compute Chamfer Distance between two point sets """
    x_expand = x.unsqueeze(1)  # (B, 1, N, 3)
    y_expand = y.unsqueeze(2)  # (B, N, 1, 3)

    dist = torch.norm(x_expand - y_expand, dim=-1)  # Compute pairwise distance
    min_dist_x = torch.min(dist, dim=2)[0]
    min_dist_y = torch.min(dist, dim=1)[0]

    print(mask.shape, min_dist_x.shape, min_dist_y.shape)

    mask = mask.unsqueeze(-1) if mask.dim() == 2 else mask  # Ensure correct dimension
    mask = mask[:, :min_dist_y.shape[1]].expand_as(min_dist_y)  # Expand mask to match min_dist_y shape
    min_dist_y = min_dist_y * mask  # Apply mask

    return min_dist_x.mean() + (min_dist_y.sum() / mask.sum())


# Define training loop
def train_diffusion_model(model, dataloader, optimizer, num_epochs=10, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)  # Create checkpoint directory if not exist
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for point_cloud, vertices, mask in dataloader:
            t = torch.randint(0, 1000, (point_cloud.shape[0],), dtype=torch.long)  # Random timestep
            point_cloud_noised, _ = diffusion.add_noise(point_cloud, t)  # Apply noise

            # 🔹 vertices를 텐서로 변환
            vertices = torch.stack(vertices) if isinstance(vertices, list) else vertices
            vertices = vertices.to(point_cloud.device)  # GPU/CPU 맞추기

            optimizer.zero_grad()
            predicted_vertices = model(point_cloud_noised, t)
            loss = chamfer_distance(predicted_vertices, vertices, mask)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 🔹 Gradient Clipping 추가
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}")

        # Save checkpoint
        # checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pth")
        # torch.save(model.state_dict(), checkpoint_path)
        # print(f"Checkpoint saved: {checkpoint_path}")


# Define model evaluation
def evaluate_diffusion_model(model, dataloader, checkpoint_path):
    """ Evaluate model by loading checkpoint and predicting vertices """
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    with torch.no_grad():
        for point_cloud, vertices, mask in dataloader:
            t = torch.randint(0, 1000, (point_cloud.shape[0],), dtype=torch.long)  # Random timestep
            point_cloud_noised, _ = diffusion.add_noise(point_cloud, t)  # Apply noise
            predicted_vertices = model(point_cloud_noised, t)

            print(f"Predicted Vertices Shape: {predicted_vertices.shape}")  # Debugging output
            return predicted_vertices  # Return first batch


if __name__ == "__main__":
    diffusion = Diffusion()

    # Define dataset and dataloader
    dataset = PointCloudDataset(pointcloud_dir="pointclouds", vertices_dir="vertices")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=pad_collate_fn)

    # Initialize model and optimizer
    model = DiffusionUNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    # Train the model
    train_diffusion_model(model, dataloader, optimizer, num_epochs=100)

    # Evaluate the model using the last checkpoint
    # checkpoint_path = "checkpoints/model_epoch_10.pth"
    # evaluate_diffusion_model(model, dataloader, checkpoint_path)
