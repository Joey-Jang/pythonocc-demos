import os

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from sklearn.decomposition import PCA
import struct


def compute_covariance_eigenvalues(points, center):
    """로컬 포인트들의 공분산 행렬 고유값 계산"""
    points_centered = points - center
    covariance = np.cov(points_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # 스케일 값을 더 작게 조정
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    # 스케일 감소 팩터 적용
    scale_factor = 0.05  # 이 값을 조절하여 전체적인 스케일 조정 가능
    eigenvalues *= scale_factor

    return eigenvalues, eigenvectors


def mesh_to_gaussian_splats(input_mesh_path, output_ply_path, num_points=10000):
    # 1. 메쉬 로드
    mesh = o3d.io.read_triangle_mesh(input_mesh_path)
    mesh.compute_vertex_normals()

    # 2. 포인트 클라우드 샘플링
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    # 3. 각 점에 대한 가우시안 특성 계산
    splats = []
    kdtree = KDTree(points)
    radius = np.mean([(points.max(axis=0) - points.min(axis=0))]) / 50.0  # 적응형 반경

    for i in range(len(points)):
        position = points[i]
        normal = normals[i]

        # 주변 점들 찾기
        neighbors_idx = kdtree.query_ball_point(position, r=radius)
        if len(neighbors_idx) < 4:  # 최소 이웃 수 확인
            continue

        local_points = points[neighbors_idx]

        # 공분산 분석
        eigenvalues, eigenvectors = compute_covariance_eigenvalues(local_points, position)

        # 스케일 계산 (표준편차)
        scales = np.sqrt(eigenvalues)

        # 회전 행렬 계산
        rotation = eigenvectors.T  # 주의: 전치 필요

        # 법선 방향이 일관되게 하기 위한 조정
        if np.dot(normal, eigenvectors[:, 2]) < 0:
            eigenvectors[:, 2] *= -1

        splat = {
            'position': position,
            'normal': normal,
            'scale': scales,
            'rotation': rotation,
            'sh': np.array([1.0, 1.0, 1.0]),  # RGB 색상
            'opacity': 1.0
        }
        splats.append(splat)

    write_splats_to_ply(splats, output_ply_path)
    return len(splats)


def write_splats_to_ply(splats, output_path):
    with open(output_path, 'wb') as f:
        # PLY 헤더
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {len(splats)}\n".encode())
        f.write(b"property float x\n")
        f.write(b"property float y\n")
        f.write(b"property float z\n")
        f.write(b"property float nx\n")
        f.write(b"property float ny\n")
        f.write(b"property float nz\n")
        f.write(b"property float f_dc_0\n")
        f.write(b"property float f_dc_1\n")
        f.write(b"property float f_dc_2\n")
        f.write(b"property float opacity\n")
        f.write(b"property float scale_0\n")
        f.write(b"property float scale_1\n")
        f.write(b"property float scale_2\n")
        f.write(b"property float rot_0\n")
        f.write(b"property float rot_1\n")
        f.write(b"property float rot_2\n")
        f.write(b"property float rot_3\n")
        f.write(b"property float rot_4\n")
        f.write(b"property float rot_5\n")
        f.write(b"property float rot_6\n")
        f.write(b"property float rot_7\n")
        f.write(b"property float rot_8\n")
        f.write(b"end_header\n")

        # 데이터 쓰기
        for splat in splats:
            # 위치 (x, y, z)
            f.write(struct.pack('<fff', *splat['position']))

            # 법선 (nx, ny, nz)
            f.write(struct.pack('<fff', *splat['normal']))

            # 색상 계수 (f_dc_0, f_dc_1, f_dc_2)
            f.write(struct.pack('<fff', *splat['sh']))

            # 불투명도
            f.write(struct.pack('<f', splat['opacity']))

            # 스케일 (scale_0, scale_1, scale_2)
            f.write(struct.pack('<fff', *splat['scale']))

            # 회전 행렬 (3x3 행렬을 flatten)
            rot_flat = splat['rotation'].flatten()
            f.write(struct.pack('<fffffffff', *rot_flat))


# 사용 예시
input_path = os.path.join("..", "assets", "models", "processed", "Unnamed-SC10 Assem v1.ply")
output_path = os.path.join("..", "assets", "models", "processed", "Unnamed-SC10 Assem v1 - GSPLAT.ply")
mesh_to_gaussian_splats(input_path, output_path, 233440)