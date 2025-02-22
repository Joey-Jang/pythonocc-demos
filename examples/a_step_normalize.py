import os

from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.gp import gp_Trsf, gp_Vec, gp_Pnt
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs, STEPControl_Reader
from OCC.Core.Interface import Interface_Static_SetCVal
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib_Add, brepbndlib_AddOptimal
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform


def save_shape(shape, filename, format='STEP'):
    """
    shape를 지정된 형식으로 저장합니다.

    Parameters:
    shape: TopoDS_Shape - 저장할 형상
    filename: str - 저장할 파일 경로
    format: str - 'STEP' 또는 'STL' (기본값: 'STEP')
    """
    if format.upper() == 'STEP':
        # STEP 파일로 저장
        step_writer = STEPControl_Writer()
        Interface_Static_SetCVal("write.step.schema", "AP203")

        status = step_writer.Transfer(shape, STEPControl_AsIs)
        if status == IFSelect_RetDone:
            status = step_writer.Write(filename)
            return status == IFSelect_RetDone
        return False

    elif format.upper() == 'STL':
        # STL 파일로 저장하기 전에 메시 생성
        mesh = BRepMesh_IncrementalMesh(shape, 0.1)
        mesh.Perform()

        # STL 파일로 저장
        stl_writer = StlAPI_Writer()
        stl_writer.Write(shape, filename)
        return True

    else:
        raise ValueError(f"Unsupported format: {format}. Use 'STEP' or 'STL'.")


def get_precise_bounding_box(shape, mesh_precision=0.01):
    """
    메시 생성을 통해 정확한 바운딩 박스를 계산합니다.

    Parameters:
    shape: TopoDS_Shape - 바운딩 박스를 계산할 형상
    mesh_precision: float - 메시 생성 정밀도 (기본값: 0.01)

    Returns:
    tuple - (xmin, ymin, zmin, xmax, ymax, zmax)
    """
    # 메시 생성
    mesh = BRepMesh_IncrementalMesh(shape, mesh_precision)
    mesh.Perform()
    if not mesh.IsDone():
        raise RuntimeError("Mesh generation failed")

    # 바운딩 박스 계산
    bbox = Bnd_Box()
    brepbndlib_AddOptimal(shape, bbox)

    return bbox.Get()


def normalize_shape(shape, mesh_precision=0.01):
    """
    주어진 shape를 정규화하여 중심이 (0,0,0)이 되고 1x1x1 바운딩 박스 안에 들어가도록 조정합니다.

    Parameters:
    shape: TopoDS_Shape - 정규화할 형상

    Returns:
    TopoDS_Shape - 정규화된 형상
    """
    # 정확한 바운딩 박스 계산
    xmin, ymin, zmin, xmax, ymax, zmax = get_precise_bounding_box(shape, mesh_precision)

    # 원본 크기 출력
    print(
        f"Original bounds before normalization: ({xmin:.3f}, {ymin:.3f}, {zmin:.3f}) to ({xmax:.3f}, {ymax:.3f}, {zmax:.3f})")

    # 중심점 계산
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    center_z = (zmin + zmax) / 2

    # 크기 계산
    size_x = xmax - xmin
    size_y = ymax - ymin
    size_z = zmax - zmin

    # 최대 크기 찾기
    max_size = max(size_x, size_y, size_z)

    # 스케일 팩터 계산 (1x1x1 박스에 맞추기 위해)
    scale = 1.0 / max_size if max_size > 0 else 1.0

    # 변환 행렬 생성
    transform = gp_Trsf()

    # 원점으로 이동
    transform.SetTranslation(gp_Vec(-center_x, -center_y, -center_z))

    # 스케일 적용
    scale_transform = gp_Trsf()
    scale_transform.SetScale(gp_Pnt(0, 0, 0), scale)

    # 변환 적용
    shape_centered = BRepBuilderAPI_Transform(shape, transform).Shape()
    shape_normalized = BRepBuilderAPI_Transform(shape_centered, scale_transform).Shape()

    return shape_normalized


# 사용 예시
def normalize_and_save(input_shape, output_filename, format='STEP', mesh_precision=0.01):
    """
    shape를 정규화하고 저장합니다.

    Parameters:
    input_shape: TopoDS_Shape - 입력 형상
    output_filename: str - 저장할 파일 경로
    format: str - 'STEP' 또는 'STL' (기본값: 'STEP')

    Returns:
    bool - 저장 성공 여부
    """
    normalized = normalize_shape(input_shape)
    return save_shape(normalized, output_filename, format)


# 사용 예시
if __name__ == "__main__":
    step_reader = STEPControl_Reader()
    step_reader.ReadFile(os.path.join(".", "step", "model_001.step"))
    step_reader.TransferRoots()
    shp = step_reader.OneShape()

    # 정규화 적용
    normalized_shp = normalize_shape(shp)

    # 결과 확인을 위한 새로운 바운딩 박스 계산
    result_bbox = Bnd_Box()
    brepbndlib_Add(normalized_shp, result_bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = result_bbox.Get()
    print(f"Normalized box bounds: ({xmin:.3f}, {ymin:.3f}, {zmin:.3f}) to ({xmax:.3f}, {ymax:.3f}, {zmax:.3f})")

    # save_shape(normalized_shp, os.path.join(".", "step", "model_001_normalized.step"))
    save_shape(normalized_shp, os.path.join(".", "step", "model_001_normalized.stl"), "STL")
    print("Normalized shape has been saved in both STEP and STL formats.")
