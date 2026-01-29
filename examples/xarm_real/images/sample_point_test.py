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

# 边界定义 (四个角点)
BOUNDARY_POINTS_2D = np.array([
    [528.6, 126.5],
    [745.0, 250.2],
    [501.9, 539.4],
    [338.1, 425.0],
])

# 固定高度和姿态
HOME_POS = [486.626923, 297.343277] 
FIXED_Z = -69.568863 
FIXED_ROLL = 3.12897
FIXED_PITCH = 0.012689
BASE_YAW = -1.01436
YAW_RANDOM_RANGE = (-np.pi/6, np.pi/6)

# -----------------------------------------------------------------------------
# 采样逻辑
# -----------------------------------------------------------------------------
class PointGenerator:
    def __init__(self):
        self.original_v = BOUNDARY_POINTS_2D 
        center = np.mean(self.original_v, axis=0)
        # 定义缩放因子 (2/3 ≈ 0.666)
        scale_factor = 2.0 / 3.0
        # 计算收缩后的顶点 V_shrunk
        # 公式：P_new = Center + (P_old - Center) * scale
        self.v = center + (self.original_v - center) * scale_factor
        self.hull = ConvexHull(self.original_v)
        
        # 重新定义可视化需要的边界属性
        # 可视化边界属性 (用原始边界画框，保证图的大小不变)
        self.min_x = np.min(self.original_v[:, 0])
        self.max_x = np.max(self.original_v[:, 0])
        self.min_y = np.min(self.original_v[:, 1])
        self.max_y = np.max(self.original_v[:, 1])
        
        self.grid_rows, self.grid_cols = 4, 4
        
        # 生成边界路径点
        # 如果你希望边界测试点也收缩，就用 self.v
        # 如果你希望边界测试点保持在最边缘，就用 self.original_v
        ccw_indices = self.hull.vertices
        self.ccw_vertices = self.v[ccw_indices]
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
    
    def get_bilinear_point(self, u, v):
        """双线性插值：将单位正方形映射到四个角点定义的四边形"""
        p0, p1, p2, p3 = self.v[0], self.v[1], self.v[2], self.v[3]
        res = (1 - u) * (1 - v) * p0 + \
              u * (1 - v) * p1 + \
              u * v * p2 + \
              (1 - u) * v * p3
        return res

    def is_inside(self, x, y):
        return all(np.dot(eq, [x, y, 1]) <= 1e-6 for eq in self.hull.equations)
    
# -----------------------------------------------------------------------------
# 可视化与保存逻辑
# -----------------------------------------------------------------------------
def main():
    gen = PointGenerator()
    
    # 准备画布参数
    scale = 1.5
    pad = 100
    offset_x = -gen.min_x * scale + 50
    offset_y = -gen.min_y * scale + 50
    
    def to_pixel(x, y):
        return int(x * scale + offset_x), int(y * scale + offset_y)

    w = int((gen.max_x - gen.min_x) * scale + pad)
    h = int((gen.max_y - gen.min_y) * scale + pad)
    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    
    # 画边界和 Home 点
    pts = np.array([to_pixel(p[0], p[1]) for p in BOUNDARY_POINTS_2D], np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], True, (0, 0, 0), 2)
    hx, hy = to_pixel(HOME_POS[0], HOME_POS[1])
    cv2.circle(img, (hx, hy), 8, (255, 0, 0), -1)

    saved_points = {"grid": [], "boundary": []}

    # 1. 执行双线性网格采样 (严格按照四角点划分 4x4)
    print("Generating Quadrilateral Grid Points...")
    for r in range(gen.grid_rows):
        for c in range(gen.grid_cols):
            u_min, u_max = c / gen.grid_cols, (c + 1) / gen.grid_cols
            v_min, v_max = r / gen.grid_rows, (r + 1) / gen.grid_rows
            
            # 绘制网格线
            grid_corners = [
                gen.get_bilinear_point(u_min, v_min),
                gen.get_bilinear_point(u_max, v_min),
                gen.get_bilinear_point(u_max, v_max),
                gen.get_bilinear_point(u_min, v_max)
            ]
            px_corners = [to_pixel(p[0], p[1]) for p in grid_corners]
            cv2.polylines(img, [np.array(px_corners, np.int32)], True, (230, 230, 230), 1)

            # 每个格子采样 2 个点
            count = 0
            for _ in range(100):
                if count >= 1: break
                u_rand = random.uniform(u_min, u_max)
                v_rand = random.uniform(v_min, v_max)
                
                target_pt = gen.get_bilinear_point(u_rand, v_rand)
                tx, ty = target_pt[0], target_pt[1]
                
                if gen.is_inside(tx, ty):
                    yaw = BASE_YAW + random.uniform(*YAW_RANDOM_RANGE)
                    saved_points["grid"].append([tx, ty, FIXED_Z, FIXED_ROLL, FIXED_PITCH, yaw])
                    
                    px, py = to_pixel(tx, ty)
                    end_x = int(px + 20 * np.cos(yaw))
                    end_y = int(py + 20 * np.sin(yaw))
                    cv2.circle(img, (px, py), 4, (255, 0, 255), -1)
                    cv2.arrowedLine(img, (px, py), (end_x, end_y), (255, 0, 255), 1, tipLength=0.3)
                    count += 1
    
    # 2. 生成边界采样点
    print("Generating Boundary Points...")
    indices = np.linspace(0, len(gen.path_points_2d) - 1, 5, dtype=int)
    for idx in indices:
        pt = gen.path_points_2d[idx]
        yaw = BASE_YAW + random.uniform(*YAW_RANDOM_RANGE)
        saved_points["boundary"].append([pt[0], pt[1], FIXED_Z, FIXED_ROLL, FIXED_PITCH, yaw])
        
        px, py = to_pixel(pt[0], pt[1])
        end_x = int(px + 20 * np.cos(yaw))
        end_y = int(py + 20 * np.sin(yaw))
        cv2.circle(img, (px, py), 4, (0, 165, 255), -1) # 橙色标记边界点
        cv2.arrowedLine(img, (px, py), (end_x, end_y), (0, 165, 255), 1, tipLength=0.3)

    # 保存结果
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    with open(os.path.join(SAVE_DIR, JSON_FILENAME), 'w') as f:
        json.dump(saved_points, f, indent=4)
    cv2.imwrite(os.path.join(SAVE_DIR, IMG_FILENAME), img)
    
    print(f"Success! Map saved to {os.path.join(SAVE_DIR, IMG_FILENAME)}")

if __name__ == "__main__":
    main()