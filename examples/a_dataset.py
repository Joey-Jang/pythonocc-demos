import torch
import numpy as np
import os


class PointCloudDataset(torch.utils.data.Dataset):
    """
    포인트 클라우드와 정점 데이터셋을 PyTorch Dataset 형식으로 변환
    """

    def __init__(self, pointcloud_dir, vertices_dir, num_points=1024):
        self.pointcloud_dir = pointcloud_dir
        self.vertices_dir = vertices_dir
        self.num_points = num_points
        self.file_names = sorted(f for f in os.listdir(pointcloud_dir) if f.endswith('.npy'))

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        pointcloud_path = os.path.join(self.pointcloud_dir, file_name)
        vertices_path = os.path.join(self.vertices_dir, file_name)

        # 포인트 클라우드 로드
        point_cloud = np.load(pointcloud_path)  # (N, 3)
        vertices = np.load(vertices_path)  # (M, 3)

        # 포인트 클라우드 샘플링 (num_points 크기로 맞추기)
        if len(point_cloud) > self.num_points:
            indices = np.random.choice(len(point_cloud), self.num_points, replace=False)
            point_cloud = point_cloud[indices]

        # 텐서 변환
        point_cloud = torch.tensor(point_cloud, dtype=torch.float32)
        vertices = torch.tensor(vertices, dtype=torch.float32)

        return point_cloud, vertices


def custom_collate_fn(batch):
    """
    변동 크기 데이터를 처리하기 위한 Collate Function
    """
    point_clouds = torch.stack([item[0] for item in batch])  # (B, 1024, 3)
    vertices = [item[1] for item in batch]  # 리스트 형태로 유지 (길이가 다를 수 있음)

    return point_clouds, vertices


# 데이터셋 경로 설정
pointcloud_dir = "pointclouds"
vertices_dir = "vertices"

# 데이터셋 로드
dataset = PointCloudDataset(pointcloud_dir, vertices_dir)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)

# 데이터 확인
for point_cloud, vertices in dataloader:
    print(f"Point Cloud Shape: {point_cloud.shape}")  # (Batch, 1024, 3)
    print(f"Vertices Batch Size: {len(vertices)}")  # 리스트 길이 확인
    print(f"First Sample Vertices Shape: {vertices[0].shape}")  # 각 샘플의 정점 개수 확인
    break
