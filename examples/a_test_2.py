import os

from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_Reader


def check_shape_quality(shape: TopoDS_Shape) -> bool:
    """
    형상의 품질을 검사하는 함수

    Args:
        shape: 검사할 TopoDS_Shape 객체

    Returns:
        bool: 형상이 유효하면 True, 문제가 있으면 False
    """
    analyzer = BRepCheck_Analyzer(shape)
    return analyzer.IsValid()


# STEP 파일 로드
step_reader = STEPControl_Reader()
status = step_reader.ReadFile(os.path.join("..", "assets", "models", "processed", "SRB.step"))

if status == IFSelect_RetDone:
    step_reader.TransferRoots()
    shape = step_reader.Shape()

    # 전체 형상 검사
    is_valid = check_shape_quality(shape)
    print(f"Shape is valid: {is_valid}")