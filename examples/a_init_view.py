import os
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods_Face
from OCC.Display.SimpleGui import init_display
from OCC.Core.gp import gp_Dir, gp_Pnt

if __name__ == "__main__":
    display, start_display, add_menu, add_function_to_menu = init_display()

    # STEP 파일 읽기
    step_reader = STEPControl_Reader()
    step_reader.ReadFile(os.path.join("..", "assets", "models", "processed", "model_002.step"))
    step_reader.TransferRoots()
    shp = step_reader.OneShape()

    explorer = TopExp_Explorer(shp, TopAbs_FACE)

    surface_color = Quantity_Color(0.2, 0.2, 0.2, Quantity_TOC_RGB)
    edge_color = Quantity_Color(0, 0, 0, Quantity_TOC_RGB)

    while explorer.More():
        face = topods_Face(explorer.Current())
        display.DisplayShape(face, color=surface_color, update=False)
        explorer.Next()

    # 카메라 위치 설정
    view = display.View
    # view.SetProj(0, 1, 0)  # y축 방향에서 바라보기
    view.SetProj(0, 1, 0)
    view.SetUp(0, 1, 0)  # z축을 위로 설정

    # 디스플레이 업데이트
    display.FitAll()

    # 디스플레이 시작
    start_display()
