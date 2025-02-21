import os
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.GeomAbs import (GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
                              GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BezierSurface,
                              GeomAbs_BSplineSurface, GeomAbs_SurfaceOfRevolution,
                              GeomAbs_SurfaceOfExtrusion, GeomAbs_OffsetSurface,
                              GeomAbs_OtherSurface, GeomAbs_C0, GeomAbs_C1, GeomAbs_C2, GeomAbs_C3)
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods_Face
from OCC.Display.SimpleGui import init_display
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.Geom import (Geom_Plane, Geom_CylindricalSurface, Geom_ConicalSurface,
                           Geom_SphericalSurface, Geom_ToroidalSurface, Geom_BezierSurface,
                           Geom_BSplineSurface, Geom_SurfaceOfRevolution,
                           Geom_SurfaceOfLinearExtrusion, Geom_OffsetSurface,
                           Geom_Surface)
from OCC.Core.GeomConvert import geomconvert_SurfaceToBSplineSurface, GeomConvert_ApproxSurface
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.Core.GeomAdaptor import GeomAdaptor_Surface
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_NurbsConvert
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.Core.gp import gp_Pln
from OCC.Core.GeomConvert import GeomConvert_ApproxSurface
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.ShapeFix import ShapeFix_Shape

if __name__ == "__main__":
    # Display 초기화
    display, start_display, add_menu, add_function_to_menu = init_display()

    # STEP 파일 읽기
    step_reader = STEPControl_Reader()
    step_reader.ReadFile(os.path.join("..", "assets", "models", "processed", "model_002.step"))
    step_reader.TransferRoots()
    shp = step_reader.OneShape()

    fixer = ShapeFix_Shape(shp)
    fixer.Perform()
    shp = fixer.Shape()

    # nurbs_converter = BRepBuilderAPI_NurbsConvert(shp)
    # nurbs_shape = nurbs_converter.Shape()
    # if nurbs_shape is not None:
    #     shp = nurbs_shape

    # 면 타입별 색상 정의 (R, G, B) - 0~1 사이의 값
    surface_colors = {
        GeomAbs_Plane: (1.0, 0.2, 0.0),  # 주황
        GeomAbs_Cylinder: (0.0, 1.0, 0.0),  # 초록
        GeomAbs_Cone: (0.0, 0.0, 1.0),  # 파랑
        GeomAbs_Sphere: (1.0, 1.0, 0.0),  # 노랑
        GeomAbs_Torus: (1.0, 0.0, 1.0),  # 마젠타
        GeomAbs_BezierSurface: (0.0, 1.0, 1.0),  # 시안
        GeomAbs_BSplineSurface: (0.2, 0.2, 0.0),  # 올리브
        GeomAbs_SurfaceOfRevolution: (1.0, 0.5, 0.0),  # 주황
        GeomAbs_SurfaceOfExtrusion: (0.5, 0.0, 0.5),  # 보라
        GeomAbs_OffsetSurface: (0.0, 0.5, 0.5),  # 청록
        GeomAbs_OtherSurface: (1.0, 0.0, 0.0)  # 빨강
    }

    # 면 타입별 카운터 초기화
    surface_types = {
        GeomAbs_Plane: "Plane",
        GeomAbs_Cylinder: "Cylinder",
        GeomAbs_Cone: "Cone",
        GeomAbs_Sphere: "Sphere",
        GeomAbs_Torus: "Torus",
        GeomAbs_BezierSurface: "Bezier",
        GeomAbs_BSplineSurface: "BSpline",
        GeomAbs_SurfaceOfRevolution: "Revolution",
        GeomAbs_SurfaceOfExtrusion: "Extrusion",
        GeomAbs_OffsetSurface: "Offset",
        GeomAbs_OtherSurface: "Other"
    }

    surface_counts = {name: 0 for name in surface_types.values()}

    # 모든 면 탐색 및 색상 지정
    explorer = TopExp_Explorer(shp, TopAbs_FACE)
    total_faces = 0

    while explorer.More():
        total_faces += 1
        face = topods_Face(explorer.Current())

        # 면의 타입 확인
        brep_surface = BRepAdaptor_Surface(face)
        surface_type = brep_surface.GetType()

        # 해당하는 타입의 카운터 증가
        if surface_type in surface_types:
            surface_counts[surface_types[surface_type]] += 1
        else:
            surface_counts["Other"] += 1

            # 색상 지정
        if surface_type in surface_colors:
            r, g, b = surface_colors[surface_type]
            color = Quantity_Color(r, g, b, Quantity_TOC_RGB)
            display.DisplayShape(face, color=color, update=False)
        else:
            # 알 수 없는 타입의 경우 회색으로 표시
            r, g, b = surface_colors[GeomAbs_OtherSurface]
            color = Quantity_Color(r, g, b, Quantity_TOC_RGB)
            display.DisplayShape(face, color=color, update=False)

        explorer.Next()

    # 디스플레이 업데이트
    display.FitAll()

    # 결과 출력
    print(f"\nTotal number of faces: {total_faces}")
    print("\nSurface types distribution:")
    for surface_type, count in surface_counts.items():
        if count > 0:  # 개수가 0인 타입은 출력하지 않음
            percentage = (count / total_faces) * 100
            print(f"{surface_type}: {count} ({percentage:.1f}%)")

    # 디스플레이 시작
    start_display()
