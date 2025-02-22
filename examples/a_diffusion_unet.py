import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader

from a_dataset import PointCloudDataset, custom_collate_fn
from a_diffusion import Diffusion


class DiffusionUNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128):
        super(DiffusionUNet, self).__init__()

        self.encoder1 = nn.Linear(input_dim, hidden_dim)
        self.encoder2 = nn.Linear(hidden_dim, hidden_dim)

        self.bottleneck = nn.Linear(hidden_dim, hidden_dim)

        self.decoder1 = nn.Linear(hidden_dim, hidden_dim)
        self.decoder2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, t):
        """ Forward pass with timestep embedding """
        t_emb = t.unsqueeze(-1).expand(-1, x.shape[1], 1)  # Expand timestep dimension
        x = torch.cat((x, t_emb), dim=-1)  # Concatenate timestep to input

        x = F.relu(self.encoder1(x))
        x = F.relu(self.encoder2(x))

        x = F.relu(self.bottleneck(x))

        x = F.relu(self.decoder1(x))
        x = self.decoder2(x)  # No activation, since we predict continuous values

        return x


# Define loss function
def chamfer_distance(x, y):
    """ Compute Chamfer Distance between two point sets """
    x_expand = x.unsqueeze(1)  # (B, 1, N, 3)
    y_expand = y.unsqueeze(2)  # (B, N, 1, 3)

    dist = torch.norm(x_expand - y_expand, dim=-1)  # Compute pairwise distance
    min_dist_x = torch.min(dist, dim=2)[0]
    min_dist_y = torch.min(dist, dim=1)[0]

    return min_dist_x.mean() + min_dist_y.mean()


# Define training loop
def train_diffusion_model(model, dataloader, optimizer, num_epochs=10, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)  # Create checkpoint directory if not exist
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for point_cloud, vertices in dataloader:
            t = torch.randint(0, 1000, (point_cloud.shape[0],), dtype=torch.long)  # Random timestep
            point_cloud_noised, _ = diffusion.add_noise(point_cloud, t)  # Apply noise

            optimizer.zero_grad()
            predicted_vertices = model(point_cloud_noised, t)
            loss = chamfer_distance(predicted_vertices, vertices)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")


# Define model evaluation
def evaluate_diffusion_model(model, dataloader, checkpoint_path):
    """ Evaluate model by loading checkpoint and predicting vertices """
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    with torch.no_grad():
        for point_cloud, vertices in dataloader:
            t = torch.randint(0, 1000, (point_cloud.shape[0],), dtype=torch.long)  # Random timestep
            point_cloud_noised, _ = diffusion.add_noise(point_cloud, t)  # Apply noise
            predicted_vertices = model(point_cloud_noised, t)

            print(f"Predicted Vertices Shape: {predicted_vertices.shape}")  # Debugging output
            return predicted_vertices  # Return first batch


if __name__ == "__main__":
    diffusion = Diffusion()

    # Define dataset and dataloader
    dataset = PointCloudDataset(pointcloud_dir="pointclouds", vertices_dir="vertices")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)

    # Initialize model and optimizer
    model = DiffusionUNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_diffusion_model(model, dataloader, optimizer, num_epochs=10)

    # Evaluate the model using the last checkpoint
    # checkpoint_path = "checkpoints/model_epoch_10.pth"
    # evaluate_diffusion_model(model, dataloader, checkpoint_path)
