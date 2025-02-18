import os.path

from OCC.Extend.DataExchange import read_step_file
from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.BRep import BRep_Tool

if __name__ == "__main__":
    shp = read_step_file(os.path.join("..", "assets", "models", "SC10 Assem v1.step"))

    # 점(Vertex) 조회
    vertex_explorer = TopExp_Explorer(shp, TopAbs_VERTEX)
    while vertex_explorer.More():
        vertex = vertex_explorer.Current()
        # 점의 좌표 얻기
        pnt = BRep_Tool.Pnt(vertex)
        print(f"Vertex coordinates: ({pnt.X()}, {pnt.Y()}, {pnt.Z()})")
        vertex_explorer.Next()

    # 선(Edge) 조회
    edge_explorer = TopExp_Explorer(shp, TopAbs_EDGE)
    while edge_explorer.More():
        edge = edge_explorer.Current()
        # 여기서 edge의 속성을 조회할 수 있습니다
        print("Found an edge")
        edge_explorer.Next()

    # 면(Face) 조회
    face_explorer = TopExp_Explorer(shp, TopAbs_FACE)
    while face_explorer.More():
        face = face_explorer.Current()
        # 여기서 face의 속성을 조회할 수 있습니다
        print("Found a face")
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
