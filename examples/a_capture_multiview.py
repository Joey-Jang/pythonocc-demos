import trimesh
import numpy as np
import os
import math
from PIL import Image
import pyrender


def generate_multiview_images(step_mesh_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # trimesh로 메시 로드
    mesh_trimesh = trimesh.load(step_mesh_path)

    # trimesh -> pyrender 메시로 변환
    mesh = pyrender.Mesh.from_trimesh(mesh_trimesh)

    # 장면 설정
    scene = pyrender.Scene()
    mesh_node = scene.add(mesh)

    # 카메라 설정
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)

    # 조명 설정
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)

    # 렌더러 설정
    r = pyrender.OffscreenRenderer(800, 600)

    # 카메라 위치 계산
    camera_positions = []
    radius = 2.0  # 카메라와 물체 사이의 거리

    # 고도각 -30도, 30도에서 각각 8장 (45도 간격)
    for elevation in [-30, 30]:
        for azimuth in range(0, 360, 45):
            elevation_rad = math.radians(elevation)
            azimuth_rad = math.radians(azimuth)

            x = radius * math.cos(elevation_rad) * math.cos(azimuth_rad)
            y = radius * math.cos(elevation_rad) * math.sin(azimuth_rad)
            z = radius * math.sin(elevation_rad)

            camera_positions.append((x, y, z))

    # 고도각 -60도, 60도에서 각각 4장 (90도 간격)
    for elevation in [-60, 60]:
        for azimuth in range(0, 360, 90):
            elevation_rad = math.radians(elevation)
            azimuth_rad = math.radians(azimuth)

            x = radius * math.cos(elevation_rad) * math.cos(azimuth_rad)
            y = radius * math.cos(elevation_rad) * math.sin(azimuth_rad)
            z = radius * math.sin(elevation_rad)

            camera_positions.append((x, y, z))

    images = []
    camera_node = None
    light_node = None

    for i, pos in enumerate(camera_positions):
        # 이전 카메라와 조명 노드가 있다면 제거
        if camera_node is not None:
            scene.remove_node(camera_node)
        if light_node is not None:
            scene.remove_node(light_node)

        # 카메라 노드 생성
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = pos

        # 카메라가 원점을 바라보도록 방향 설정
        camera_pos = np.array(pos)
        target = np.array([0, 0, 0])
        up = np.array([0, 0, 1])

        z_axis = target - camera_pos
        z_axis = z_axis / np.linalg.norm(z_axis)

        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)

        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        camera_pose[:3, 0] = x_axis
        camera_pose[:3, 1] = y_axis
        camera_pose[:3, 2] = z_axis

        # 장면에 카메라와 조명 추가
        camera_node = scene.add(camera, pose=camera_pose)
        light_pose = np.copy(camera_pose)
        light_node = scene.add(light, pose=light_pose)

        # 렌더링
        color, depth = r.render(scene)

        # 이미지 저장
        image = Image.fromarray(color)
        save_path = os.path.join(output_folder, f"view_{i:03d}.png")
        image.save(save_path)
        images.append(save_path)

    r.delete()
    return images


if __name__ == "__main__":
    # 실행 예시
    mesh_path = os.path.join("..", "assets", "models", "bunny.obj")
    output_folder = os.path.join("..", "assets", "multiview")
    images = generate_multiview_images(mesh_path, output_folder)