import numpy as np
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_VERTEX
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.BRep import BRep_Tool
import open3d as o3d
import os

from OCC.Core.TopoDS import topods_Vertex


def step_to_vertices_npy(step_file_path, output_path):
    """
    STEP 파일에서 vertices 정보를 추출하여 NPY 파일로 저장

    Args:
        step_file_path (str): 입력 STEP 파일 경로
        output_path (str): 출력 NPY 파일 경로
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

    # numpy 배열로 변환 및 저장
    vertices = np.array(vertices, dtype=np.float32)
    np.save(output_path, vertices)

    print(f"Extracted {len(vertices)} vertices and saved to {output_path}")
    return vertices


def ply_to_pointcloud_npy(ply_file_path, output_path):
    """
    PLY 파일에서 포인트 클라우드 정보를 추출하여 NPY 파일로 저장

    Args:
        ply_file_path (str): 입력 PLY 파일 경로
        output_path (str): 출력 NPY 파일 경로
    """
    # PLY 파일 읽기
    pcd = o3d.io.read_point_cloud(ply_file_path)

    # 포인트 클라우드 좌표 추출
    points = np.asarray(pcd.points, dtype=np.float32)

    # NPY 파일로 저장
    np.save(output_path, points)

    print(f"Extracted {len(points)} points and saved to {output_path}")
    return points


def process_directory(step_dir, ply_dir, output_dir):
    """
    디렉토리 내의 모든 STEP과 PLY 파일을 처리

    Args:
        step_dir (str): STEP 파일이 있는 디렉토리
        ply_dir (str): PLY 파일이 있는 디렉토리
        output_dir (str): NPY 파일을 저장할 디렉토리
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
            step_to_vertices_npy(step_path, vertex_path)

    # PLY 파일 처리
    for file_name in os.listdir(ply_dir):
        if file_name.endswith('.ply'):
            base_name = os.path.splitext(file_name)[0]
            ply_path = os.path.join(ply_dir, file_name)
            pointcloud_path = os.path.join(pointcloud_dir, f"{base_name}.npy")
            ply_to_pointcloud_npy(ply_path, pointcloud_path)


# 사용 예시
if __name__ == "__main__":
    # 단일 파일 처리
    # step_to_vertices_npy("path/to/model.step", "output/vertices.npy")
    # ply_to_pointcloud_npy("path/to/model.ply", "output/pointclouds.npy")

    # 디렉토리 일괄 처리
    process_directory(
        step_dir="step/normalized",
        ply_dir="ply",
        output_dir="."
    )
