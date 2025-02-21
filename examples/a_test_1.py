import math
import os

import numpy as np
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepGProp import brepgprop_VolumeProperties
from OCC.Core.GProp import GProp_GProps
from OCC.Core.Graphic3d import Graphic3d_Camera
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_VERTEX, TopAbs_EDGE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods_Face, topods
from OCC.Core.V3d import V3d_AmbientLight, V3d_DirectionalLight, V3d_SpotLight
from OCC.Core.gp import gp_Pnt, gp_Dir
from OCC.Display.SimpleGui import init_display
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB


def calculate_camera_direction(azimuth, elevation):
    """
    방위각(azimuth)과 고도각(elevation)으로부터 카메라 방향 계산
    azimuth: y축 기준 회전각 (도)
    elevation: xz 평면 기준 고도각 (도)
    """
    # 라디안으로 변환
    azimuth_rad = math.radians(azimuth)
    elevation_rad = math.radians(elevation)

    # 구면 좌표계에서 방향 벡터 계산
    x = math.cos(elevation_rad) * math.sin(azimuth_rad)
    y = math.sin(elevation_rad)
    z = math.cos(elevation_rad) * math.cos(azimuth_rad)

    return gp_Dir(-x, -y, -z)


def calculate_camera_position(azimuth, elevation, distance, center_point):
    """
    방위각(azimuth), 고도각(elevation), 원하는 거리(distance)와 중심점으로부터
    카메라 위치 계산
    """
    # 라디안으로 변환
    azimuth_rad = math.radians(azimuth)
    elevation_rad = math.radians(elevation)

    # 구면 좌표계에서 카메라 위치 계산
    x = distance * math.cos(elevation_rad) * math.sin(azimuth_rad)
    y = distance * math.sin(elevation_rad)
    z = distance * math.cos(elevation_rad) * math.cos(azimuth_rad)

    # 중심점으로부터의 상대 위치
    camera_x = center_point.X() + x
    camera_y = center_point.Y() + y
    camera_z = center_point.Z() + z

    return gp_Pnt(camera_x, camera_y, camera_z)


def capture_views(display, shp):  # 원하는 거리 파라미터 추가
    # 출력 디렉토리 설정
    output_dir = "captured_views"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 물체의 중심점 계산
    props = GProp_GProps()
    brepgprop_VolumeProperties(shp, props)
    center_point = props.CentreOfMass()

    # 뷰 설정
    elevation_angles = {
        30: 8,
        60: 4,
    }

    for elevation, num_shots in elevation_angles.items():
        azimuth_step = 360 / num_shots

        for i in range(num_shots):
            azimuth = i * azimuth_step

            camera = display.View.Camera()
            direction = calculate_camera_direction(azimuth, elevation)

            up = gp_Dir(0, 1, 0)
            camera.SetUp(up)

            # 방향 설정
            camera.SetDirection(direction)

            camera.SetScale(initial_scale * 1.25)

            # 뷰 업데이트
            display.View.Update()

            # 이미지 저장
            filename = os.path.join(output_dir, f"view_elev{elevation}_azim{int(azimuth)}.png")
            display.ExportToImage(filename)
            # display.View.Dump(filename)


def analyze_surface(face):
    # Edge 탐색
    edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
    edges = []

    while edge_explorer.More():
        edge = topods.Edge(edge_explorer.Current())
        edge_data = {"edge": edge, "vertices": []}

        # 각 Edge의 Vertex 탐색
        vertex_explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
        while vertex_explorer.More():
            vertex = topods.Vertex(vertex_explorer.Current())
            pnt = BRep_Tool.Pnt(vertex)
            edge_data["vertices"].append(pnt)
            vertex_explorer.Next()

        edges.append(edge_data)

        edge_explorer.Next()

    return edges


if __name__ == "__main__":
    # Display 초기화
    display, start_display, add_menu, add_function_to_menu = init_display()

    # STEP 파일 읽기
    step_reader = STEPControl_Reader()
    step_reader.ReadFile(os.path.join("..", "assets", "models", "processed", "model_003.step"))
    step_reader.TransferRoots()
    shp = step_reader.OneShape()

    explorer = TopExp_Explorer(shp, TopAbs_FACE)

    surface_color = Quantity_Color(0.25, 0.25, 0.25, Quantity_TOC_RGB)
    edge_color = Quantity_Color(0, 0, 0, Quantity_TOC_RGB)

    while explorer.More():
        face = topods_Face(explorer.Current())
        display.DisplayShape(face, color=surface_color, update=False)

        edges = analyze_surface(face)
        for edge in edges:
            display.DisplayShape(edge["edge"], color=edge_color, update=False)

        explorer.Next()

    camera = display.View.Camera()

    # 디스플레이 업데이트
    display.FitAll()
    initial_scale = camera.Scale()

    display.hide_triedron()
    display.View.Update()

    # 멀티뷰 캡쳐
    # capture_views(display, shp)

    # 디스플레이 시작
    start_display()
