import os

import numpy as np
import open3d as o3d

# 두 PLY 파일 로드
original = o3d.io.read_triangle_mesh(os.path.join(".", "step", "model_001_normalized.stl"))
converted = o3d.io.read_point_cloud(os.path.join(".", "ply", "model_001.ply"))

# Y좌표의 분포 비교
original_y = np.asarray(original.points)[:, 1]
converted_y = np.asarray(converted.points)[:, 1]

# 평균, 중앙값 등 통계값 비교
print("Original Y mean:", np.mean(original_y))
print("Converted Y mean:", np.mean(converted_y))
