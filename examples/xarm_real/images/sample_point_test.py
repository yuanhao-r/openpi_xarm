import cv2
import numpy as np
import random
import json
import os
from pathlib import Path
from scipy.spatial import ConvexHull

# -----------------------------------------------------------------------------
# 配置
# -----------------------------------------------------------------------------
SAVE_DIR = "/home/openpi/examples/xarm_real/images"
JSON_FILENAME = "test_points.json"
IMG_FILENAME = "test_points_map.png"

# 边界定义
BOUNDARY_POINTS_2D = np.array([
   [505.982422, -150.631149],
   [712.302856, -66.848724],
   [697.232117, 163.981003],
   [466.805481, 144.618057],
])

# 固定高度和姿态 (与推理代码一致)
HOME_POS = [539.120605, 17.047951] 
FIXED_Z = -69.568863 # POS_A[2]
FIXED_ROLL = 3.12897
FIXED_PITCH = 0.012689
BASE_YAW = -1.01436
YAW_RANDOM_RANGE = (-np.pi/6, np.pi/6)

# -----------------------------------------------------------------------------
# 采样逻辑
# -----------------------------------------------------------------------------
class PointGenerator:
    def __init__(self):
        self.hull = ConvexHull(BOUNDARY_POINTS_2D)
        self.min_x = np.min(BOUNDARY_POINTS_2D[:, 0])
        self.max_x = np.max(BOUNDARY_POINTS_2D[:, 0])
        self.min_y = np.min(BOUNDARY_POINTS_2D[:, 1])
        self.max_y = np.max(BOUNDARY_POINTS_2D[:, 1])
        self.grid_rows, self.grid_cols = 4, 4
        
        ccw_indices = self.hull.vertices
        self.ccw_vertices = BOUNDARY_POINTS_2D[ccw_indices]
        self.path_points_2d = self._generate_boundary_path(self.ccw_vertices, 20.0)

    def _generate_boundary_path(self, vertices, step_size):
        path = []
        num_v = len(vertices)
        for i in range(num_v):
            p_curr = vertices[i]
            p_next = vertices[(i + 1) % num_v]
            vec = p_next - p_curr
            dist = np.linalg.norm(vec)
            steps = int(max(1, dist / step_size))
            unit_vec = vec / dist
            for s in range(steps):
                path.append(p_curr + unit_vec * (s * step_size))
        return np.array(path)

    def is_inside(self, x, y):
        return all(np.dot(eq, [x, y, 1]) <= 1e-6 for eq in self.hull.equations)

# -----------------------------------------------------------------------------
# 可视化与保存逻辑
# -----------------------------------------------------------------------------
def main():
    gen = PointGenerator()
    
    # 准备画布
    scale = 1.5
    pad = 100
    offset_x = -gen.min_x * scale + 50
    offset_y = -gen.min_y * scale + 50
    
    def to_pixel(x, y):
        return int(x * scale + offset_x), int(y * scale + offset_y)

    w = int((gen.max_x - gen.min_x) * scale + pad)
    h = int((gen.max_y - gen.min_y) * scale + pad)
    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    
    # 画边界和 Home
    pts = np.array([to_pixel(p[0], p[1]) for p in BOUNDARY_POINTS_2D], np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], True, (0, 0, 0), 2)
    hx, hy = to_pixel(HOME_POS[0], HOME_POS[1])
    cv2.circle(img, (hx, hy), 8, (255, 0, 0), -1)

    # 数据列表
    saved_points = {
        "grid": [],
        "boundary": []
    }

    # 1. 生成网格点 (16个区域 * 2个点)
    step_x = (gen.max_x - gen.min_x) / gen.grid_cols
    step_y = (gen.max_y - gen.min_y) / gen.grid_rows

    print("Generating Grid Points...")
    for r in range(gen.grid_rows):
        for c in range(gen.grid_cols):
            # 定义格子范围
            c_min_x = gen.min_x + c * step_x
            c_max_x = gen.min_x + (c + 1) * step_x
            c_min_y = gen.min_y + r * step_y
            c_max_y = gen.min_y + (r + 1) * step_y
            
            # 画网格线 (灰色)
            u1, v1 = to_pixel(c_min_x, c_min_y)
            u2, v2 = to_pixel(c_max_x, c_max_y)
            cv2.rectangle(img, (u1, v1), (u2, v2), (240, 240, 240), 1)

            # 每个格子取 2 个点
            count = 0
            for _ in range(50): # 尝试多次
                if count >= 2: break
                tx = random.uniform(c_min_x, c_max_x)
                ty = random.uniform(c_min_y, c_max_y)
                
                if gen.is_inside(tx, ty):
                    yaw = BASE_YAW + random.uniform(*YAW_RANDOM_RANGE)
                    
                    # 保存数据
                    pose = [tx, ty, FIXED_Z, FIXED_ROLL, FIXED_PITCH, yaw]
                    saved_points["grid"].append(pose)
                    
                    # 画图 (紫色箭头)
                    px, py = to_pixel(tx, ty)
                    end_x = int(px + 20 * np.cos(yaw))
                    end_y = int(py + 20 * np.sin(yaw))
                    cv2.circle(img, (px, py), 4, (255, 0, 255), -1)
                    cv2.arrowedLine(img, (px, py), (end_x, end_y), (255, 0, 255), 2, tipLength=0.3)
                    
                    count += 1
    
    # 2. 生成边界点 (均匀 10 个)
    print("Generating Boundary Points...")
    indices = np.linspace(0, len(gen.path_points_2d) - 1, 10, dtype=int)
    for idx in indices:
        pt = gen.path_points_2d[idx]
        yaw = BASE_YAW + random.uniform(*YAW_RANDOM_RANGE)
        
        # 保存数据
        pose = [pt[0], pt[1], FIXED_Z, FIXED_ROLL, FIXED_PITCH, yaw]
        saved_points["boundary"].append(pose)
        
        # 画图
        px, py = to_pixel(pt[0], pt[1])
        end_x = int(px + 20 * np.cos(yaw))
        end_y = int(py + 20 * np.sin(yaw))
        cv2.circle(img, (px, py), 4, (255, 0, 255), -1)
        cv2.arrowedLine(img, (px, py), (end_x, end_y), (255, 0, 255), 2, tipLength=0.3)

    # 保存文件
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    json_path = os.path.join(SAVE_DIR, JSON_FILENAME)
    img_path = os.path.join(SAVE_DIR, IMG_FILENAME)
    
    with open(json_path, 'w') as f:
        json.dump(saved_points, f, indent=4)
        
    cv2.imwrite(img_path, img)
    
    print(f"Done!")
    print(f"Points saved to: {json_path}")
    print(f"Map image saved to: {img_path}")
    print(f"Total Grid Points: {len(saved_points['grid'])}")
    print(f"Total Boundary Points: {len(saved_points['boundary'])}")

if __name__ == "__main__":
    main()