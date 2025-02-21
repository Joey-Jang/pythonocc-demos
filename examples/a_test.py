import os

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.GeomAbs import GeomAbs_BSplineSurface
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods_Face
from OCC.Core.Geom import Geom_BSplineSurface

from OCC.Extend.DataExchange import read_step_file
from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.BRep import BRep_Tool

if __name__ == "__main__":
    step_reader = STEPControl_Reader()
    step_reader.ReadFile(os.path.join("..", "assets", "models", "model_001.step"))
    step_reader.TransferRoots()
    shp = step_reader.OneShape()

    # 모든 face 탐색
    explorer = TopExp_Explorer(shp, TopAbs_FACE)

    count = 0
    while explorer.More():
        face = topods_Face(explorer.Current())
        surface = BRep_Tool.Surface(face)

        # B-spline surface로 변환 시도
        try:
            bspline = Geom_BSplineSurface.DownCast(surface)
            if bspline:
                count += 1
                # print("\nFound B-Spline Surface:")
                # B-spline surface 데이터 출력
                # print("Degrees:", bspline.UDegree(), bspline.VDegree())
                # print("Number of control points:", bspline.NbUPoles(), bspline.NbVPoles())
                # print("Number of knots:", bspline.NbUKnots(), bspline.NbVKnots())

                # Control points 조회
                for i in range(1, bspline.NbUPoles() + 1):
                    for j in range(1, bspline.NbVPoles() + 1):
                        point = bspline.Pole(i, j)
                        # print(f"Control point ({i},{j}):", point.X(), point.Y(), point.Z())

                # Knot vectors 조회
                # print("\nU knots:")
                # for i in range(1, bspline.NbUKnots() + 1):
                #     print(bspline.UKnot(i))
                #
                # print("\nV knots:")
                # for i in range(1, bspline.NbVKnots() + 1):
                #     print(bspline.VKnot(i))

                # Rational인 경우 weight 값도 조회
                # if bspline.IsRational():
                #     print("\nWeights:")
                #     for i in range(1, bspline.NbUPoles() + 1):
                #         for j in range(1, bspline.NbVPoles() + 1):
                #             print(f"Weight ({i},{j}):", bspline.Weight(i, j))

        except Exception as e:
            print(f"Not a B-spline surface: {e}")

        explorer.Next()
    print("\nNumber of bspline:", count)

    # 점(Vertex) 조회
    vertex_explorer = TopExp_Explorer(shp, TopAbs_VERTEX)
    while vertex_explorer.More():
        vertex = vertex_explorer.Current()
        # 점의 좌표 얻기
        pnt = BRep_Tool.Pnt(vertex)
        # print(f"Vertex coordinates: ({pnt.X()}, {pnt.Y()}, {pnt.Z()})")
        vertex_explorer.Next()

    # 선(Edge) 조회
    edge_explorer = TopExp_Explorer(shp, TopAbs_EDGE)
    while edge_explorer.More():
        edge = edge_explorer.Current()
        # 여기서 edge의 속성을 조회할 수 있습니다
        # print("Found an edge")
        edge_explorer.Next()

    # 면(Face) 조회
    face_explorer = TopExp_Explorer(shp, TopAbs_FACE)
    while face_explorer.More():
        face = face_explorer.Current()
        # 여기서 face의 속성을 조회할 수 있습니다
        # print("Found a face")
        face_explorer.Next()

    # 각 요소의 개수 세기
    vertex_count = 0
    edge_count = 0
    face_count = 0

    vertex_explorer = TopExp_Explorer(shp, TopAbs_VERTEX)
    edge_explorer = TopExp_Explorer(shp, TopAbs_EDGE)
    face_explorer = TopExp_Explorer(shp, TopAbs_FACE)

    while vertex_explorer.More():
        vertex_count += 1
        vertex_explorer.Next()

    while edge_explorer.More():
        edge_count += 1
        edge_explorer.Next()

    while face_explorer.More():
        face_count += 1
        face_explorer.Next()

    print(f"Total counts - Vertices: {vertex_count}, Edges: {edge_count}, Faces: {face_count}")
