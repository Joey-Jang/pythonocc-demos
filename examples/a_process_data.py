import numpy as np
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_VERTEX
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.BRep import BRep_Tool
import open3d as o3d
import os

from OCC.Core.TopoDS import topods_Vertex


def fps_sampling(points, n_samples):
    """
    Farthest Point Sampling (FPS) 방식으로 포인트 샘플링

    Args:
        points (np.ndarray): 샘플링할 포인트 배열 (N x 3)
        n_samples (int): 샘플링 후 원하는 포인트 개수

    Returns:
        np.ndarray: 샘플링된 포인트 배열 (n_samples x 3)
    """
    n_points = len(points)
    if n_points <= n_samples:
        return points

    # 첫 번째 점은 무작위로 선택
    sampled_indices = [np.random.randint(n_points)]
    distances = np.full(n_points, np.inf)

    # FPS 알고리즘
    for _ in range(1, n_samples):
        last_sampled = points[sampled_indices[-1]]
        dist = np.linalg.norm(points - last_sampled, axis=1)
        distances = np.minimum(distances, dist)
        sampled_indices.append(np.argmax(distances))

    return points[sampled_indices]


def step_to_vertices_npy(step_file_path, output_path, max_vertices=5000):
    """
    STEP 파일에서 vertices 정보를 추출하여 NPY 파일로 저장
    FPS 방식으로 샘플링 수행

    Args:
        step_file_path (str): 입력 STEP 파일 경로
        output_path (str): 출력 NPY 파일 경로
        max_vertices (int): 최대 vertex 개수
    """
    # STEP 파일 읽기
    step_reader = STEPControl_Reader()
    step_reader.ReadFile(step_file_path)
    step_reader.TransferRoot()
    shape = step_reader.OneShape()

    # vertices 추출
    vertices = []
    explorer = TopExp_Explorer(shape, TopAbs_VERTEX)

    while explorer.More():
        vertex = explorer.Current()
        point = BRep_Tool.Pnt(topods_Vertex(vertex))
        vertices.append([point.X(), point.Y(), point.Z()])
        explorer.Next()

    # numpy 배열로 변환
    vertices = np.array(vertices, dtype=np.float32)

    # FPS 샘플링 수행
    original_count = len(vertices)
    vertices = fps_sampling(vertices, max_vertices)

    # 저장
    np.save(output_path, vertices)

    print(f"Processed vertices: {original_count} -> {len(vertices)}")
    print(f"Saved to {output_path}")
    return vertices


def ply_to_pointcloud_npy(ply_file_path, output_path, max_points=20000):
    """
    PLY 파일에서 포인트 클라우드 정보를 추출하여 NPY 파일로 저장
    FPS 방식으로 샘플링 수행

    Args:
        ply_file_path (str): 입력 PLY 파일 경로
        output_path (str): 출력 NPY 파일 경로
        max_points (int): 최대 포인트 개수
    """
    # PLY 파일 읽기
    pcd = o3d.io.read_point_cloud(ply_file_path)

    # 포인트 클라우드 좌표 추출
    points = np.asarray(pcd.points, dtype=np.float32)

    # FPS 샘플링 수행
    original_count = len(points)
    points = fps_sampling(points, max_points)

    # NPY 파일로 저장
    np.save(output_path, points)

    print(f"Processed points: {original_count} -> {len(points)}")
    print(f"Saved to {output_path}")
    return points


def process_directory(step_dir, ply_dir, output_dir, max_vertices=5000, max_points=5000):
    """
    디렉토리 내의 모든 STEP과 PLY 파일을 처리

    Args:
        step_dir (str): STEP 파일이 있는 디렉토리
        ply_dir (str): PLY 파일이 있는 디렉토리
        output_dir (str): NPY 파일을 저장할 디렉토리
        max_vertices (int): STEP 파일당 최대 vertex 개수
        max_points (int): PLY 파일당 최대 포인트 개수
    """
    # 출력 디렉토리 생성
    vertex_dir = os.path.join(output_dir, 'vertices')
    pointcloud_dir = os.path.join(output_dir, 'pointclouds')
    os.makedirs(vertex_dir, exist_ok=True)
    os.makedirs(pointcloud_dir, exist_ok=True)

    # STEP 파일 처리
    for file_name in os.listdir(step_dir):
        if file_name.endswith('.step') or file_name.endswith('.stp'):
            base_name = os.path.splitext(file_name)[0]
            step_path = os.path.join(step_dir, file_name)
            vertex_path = os.path.join(vertex_dir, f"{base_name}.npy")
            step_to_vertices_npy(step_path, vertex_path, max_vertices)

    # PLY 파일 처리
    for file_name in os.listdir(ply_dir):
        if file_name.endswith('.ply'):
            base_name = os.path.splitext(file_name)[0]
            ply_path = os.path.join(ply_dir, file_name)
            pointcloud_path = os.path.join(pointcloud_dir, f"{base_name}.npy")
            ply_to_pointcloud_npy(ply_path, pointcloud_path, max_points)


# 사용 예시
if __name__ == "__main__":
    # 단일 파일 처리
    # step_to_vertices_npy("path/to/model.step", "output/vertices.npy")
    # ply_to_pointcloud_npy("path/to/model.ply", "output/pointclouds.npy")

    # 디렉토리 일괄 처리
    process_directory(
        step_dir="step/normalized",
        ply_dir="ply",
        output_dir=".",
        max_vertices=5000,
        max_points=200000
    )
