import torch
import numpy as np
import os

import torch
import numpy as np
import os


def farthest_point_sampling(points, num_samples):
    """
    Farthest Point Sampling (FPS)을 이용해 포인트 클라우드에서 num_samples 개의 포인트 선택.
    """
    N, _ = points.shape
    sampled_indices = np.zeros(num_samples, dtype=np.int64)
    distances = np.ones(N) * np.inf  # 모든 점까지의 거리 초기화
    farthest = np.random.randint(0, N)  # 랜덤한 점 하나 선택

    for i in range(num_samples):
        sampled_indices[i] = farthest
        centroid = points[farthest, :]
        dist = np.linalg.norm(points - centroid, axis=1)
        distances = np.minimum(distances, dist)  # 가장 가까운 거리 업데이트
        farthest = np.argmax(distances)  # 가장 먼 점 선택

    return points[sampled_indices]


class PointCloudDataset(torch.utils.data.Dataset):
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

        # FPS로 1024개 샘플링
        if len(point_cloud) > self.num_points:
            point_cloud = farthest_point_sampling(point_cloud, self.num_points)

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


def pad_collate_fn(batch):
    """
    Custom collate function to pad variable-length vertices tensors.
    """
    point_clouds, vertices_list = zip(*batch)  # Unpack batch

    # Convert point clouds to tensor (same size by default)
    point_clouds = torch.stack(point_clouds)  # (Batch, 1024, 3)

    # Find max number of vertices in the batch
    max_vertices = max(v.shape[0] for v in vertices_list)

    # Pad vertices to the max size in the batch
    padded_vertices = []
    mask = []
    for v in vertices_list:
        pad_size = max_vertices - v.shape[0]
        padded = torch.cat([v, torch.zeros((pad_size, 3))], dim=0)  # Pad with zeros
        padded_vertices.append(padded)

        mask.append(torch.cat([torch.ones(v.shape[0]), torch.zeros(pad_size)]))  # Mask for valid points

    # Convert lists to tensors
    padded_vertices = torch.stack(padded_vertices)  # (Batch, max_vertices, 3)
    mask = torch.stack(mask)  # (Batch, max_vertices)

    return point_clouds, padded_vertices, mask


if __name__ == '__main__':
    # 데이터셋 경로 설정
    pointcloud_dir = "pointclouds"
    vertices_dir = "vertices"

    # 데이터셋 로드
    dataset = PointCloudDataset(pointcloud_dir, vertices_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=pad_collate_fn)

    # 데이터 확인
    for point_cloud, vertices, mask in dataloader:
        print(f"Point Cloud Shape: {point_cloud.shape}")  # (Batch, 1024, 3)
        print(f"Vertices Batch Size: {len(vertices)}")  # 리스트 길이 확인
        print(f"First Sample Vertices Shape: {vertices[0].shape}")  # 각 샘플의 정점 개수 확인
        break
