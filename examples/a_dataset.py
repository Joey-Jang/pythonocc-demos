import torch
import numpy as np
import os


class PointCloudDataset(torch.utils.data.Dataset):
    """
    포인트 클라우드와 정점 데이터셋을 PyTorch Dataset 형식으로 변환
    """

    def __init__(self, pointcloud_dir, vertices_dir, num_points=1024):
        """
        Args:
            pointcloud_dir (str): 포인트 클라우드 NPY 파일이 저장된 디렉토리
            vertices_dir (str): 정점 데이터 NPY 파일이 저장된 디렉토리
            num_points (int): 샘플링할 포인트 개수 (1024 기본값)
        """
        self.pointcloud_dir = pointcloud_dir
        self.vertices_dir = vertices_dir
        self.num_points = num_points

        # 파일 리스트 생성 (파일명 기준으로 정렬)
        self.file_names = sorted(f for f in os.listdir(pointcloud_dir) if f.endswith('.npy'))

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        """
        데이터 로드 및 전처리
        Returns:
            point_cloud (torch.Tensor): (num_points, 3) 포인트 클라우드 데이터
            vertices (torch.Tensor): (num_vertices, 3) 정점 데이터
        """
        file_name = self.file_names[idx]
        pointcloud_path = os.path.join(self.pointcloud_dir, file_name)
        vertices_path = os.path.join(self.vertices_dir, file_name)

        # 포인트 클라우드 로드
        point_cloud = np.load(pointcloud_path)  # (N, 3)
        vertices = np.load(vertices_path)  # (M, 3)

        # 포인트 클라우드 샘플링 (num_points 크기로 균등 샘플링)
        if len(point_cloud) > self.num_points:
            indices = np.random.choice(len(point_cloud), self.num_points, replace=False)
            point_cloud = point_cloud[indices]

        # 텐서 변환
        point_cloud = torch.tensor(point_cloud, dtype=torch.float32)
        vertices = torch.tensor(vertices, dtype=torch.float32)

        return point_cloud, vertices


if __name__ == '__main__':
    # 데이터셋 경로 설정
    pointcloud_dir = "pointclouds"
    vertices_dir = "vertices"

    # 데이터셋 로드
    dataset = PointCloudDataset(pointcloud_dir, vertices_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    # 데이터 확인
    for point_cloud, vertices in dataloader:
        print(f"Point Cloud Shape: {point_cloud.shape}")  # (Batch, 1024, 3)
        print(f"Vertices Shape: {vertices.shape}")  # (Batch, M, 3)
        break
