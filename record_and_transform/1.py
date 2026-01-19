import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import math

class MockSampler:
    def __init__(self):
        # ================= 配置参数 (完全复制你的代码) =================
        self.pos_home = [486.626923, 158.343277, 30.431152, 3.12897, 0.012689, -1.01436]
        self.pos_A = [486.626923, 158.343277, -65.431152, 3.12897, 0.012689, -1.01436]
        self.fixed_z = self.pos_A[2]
        
        # 2D凸包区域定义 (原始点)
        self.boundary_points_2d = np.array([
            [519.6, -62.5],
            [779.8, 15.8],
            [668.9, 370.4],
            [406.0, 306.3],
        ])
        
        # 凸包缩放逻辑
        scale_factor = 0.8 
        center_point = np.mean(self.boundary_points_2d, axis=0)
        self.boundary_points_2d = center_point + (self.boundary_points_2d - center_point) * scale_factor
        
        # 构建凸包
        self.hull_2d = ConvexHull(self.boundary_points_2d)
        self.hull_points_2d = self.boundary_points_2d[self.hull_2d.vertices]
        
        # 包围盒
        self.min_x = np.min(self.boundary_points_2d[:, 0])
        self.max_x = np.max(self.boundary_points_2d[:, 0])
        self.min_y = np.min(self.boundary_points_2d[:, 1])
        self.max_y = np.max(self.boundary_points_2d[:, 1])
        
        # 网格参数
        self.grid_rows = 4
        self.grid_cols = 4
        self.total_grids = self.grid_rows * self.grid_cols
        
        # Yaw 参数
        self.base_yaw = self.pos_A[5]
        self.yaw_random_range = (-np.pi/2, np.pi/6)
        self.fixed_roll = self.pos_A[3]
        self.fixed_pitch = self.pos_A[4]
        
        # 初始化网格
        self.grid_indices = []
        self._refill_grid_indices()

    def _refill_grid_indices(self):
        self.grid_indices = []
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                self.grid_indices.append((r, c))
        random.shuffle(self.grid_indices)
        # print(f"[Sim] Grid refilled. Size: {len(self.grid_indices)}")

    def is_point_inside_hull_2d(self, point_2d, hull_2d):
        hull_eq = hull_2d.equations
        point_homo = np.hstack([point_2d, 1])
        for eq in hull_eq:
            if np.dot(eq, point_homo) > 1e-6:
                return False
        return True

    def sample_random_in_grid(self, row, col):
        step_x = (self.max_x - self.min_x) / self.grid_cols
        step_y = (self.max_y - self.min_y) / self.grid_rows
        
        cell_min_x = self.min_x + col * step_x
        cell_max_x = cell_min_x + step_x
        cell_min_y = self.min_y + row * step_y
        cell_max_y = cell_min_y + step_y

        for _ in range(30):
            x = random.uniform(cell_min_x, cell_max_x)
            y = random.uniform(cell_min_y, cell_max_y)
            sample_point_2d = np.array([x, y])
            
            if self.is_point_inside_hull_2d(sample_point_2d, self.hull_2d):
                return np.array([x, y, self.fixed_z])
        return None

    # 模拟机械臂的可达性检查 (假设凸包内的点都可达)
    def check_pose_reachable(self, pose):
        return True

    def sample_random_target(self):
        max_checks = self.total_grids * 2
        checks_count = 0
        
        while checks_count < max_checks:
            if not self.grid_indices:
                self._refill_grid_indices()
            
            # 使用原来的 pop 逻辑
            r, c = self.grid_indices.pop()
            
            sample_xyz = self.sample_random_in_grid(r, c)
            
            if sample_xyz is not None:
                target_x, target_y, target_z = sample_xyz
                
                # 模拟 Yaw 采样
                for _ in range(10): 
                    yaw_noise = random.uniform(*self.yaw_random_range)
                    candidate_yaw = self.base_yaw + yaw_noise
                    candidate_pose = [target_x, target_y, target_z, 0, 0, candidate_yaw]
                    
                    if self.check_pose_reachable(candidate_pose):
                        return candidate_pose # 成功返回

            # 如果失败，这里直接进入下一次循环，r,c 已经被丢弃了
            checks_count += 1
        
        print("Failed to sample")
        return None

# ================= 可视化脚本 =================
def visualize_sampling(num_samples=100):
    sampler = MockSampler()
    
    generated_points = []
    generated_yaws = []
    
    print(f"Generating {num_samples} samples using your logic...")
    for i in range(num_samples):
        res = sampler.sample_random_target()
        if res:
            generated_points.append(res[:3])
            generated_yaws.append(res[5])
            
    generated_points = np.array(generated_points)
    
    # --- 开始绘图 ---
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    
    # 1. 绘制凸包边界 (红色)
    hull = sampler.hull_2d
    # 闭合多边形
    hull_pts = sampler.boundary_points_2d[hull.vertices]
    hull_pts = np.vstack([hull_pts, hull_pts[0]]) 
    plt.plot(hull_pts[:, 0], hull_pts[:, 1], 'r-', linewidth=2, label='Convex Hull (Scaled)')
    
    # 2. 绘制网格线 (灰色虚线) - 用于检查是否有些格子是空的
    step_x = (sampler.max_x - sampler.min_x) / sampler.grid_cols
    step_y = (sampler.max_y - sampler.min_y) / sampler.grid_rows
    
    for i in range(sampler.grid_cols + 1):
        x = sampler.min_x + i * step_x
        plt.axvline(x, color='gray', linestyle='--', alpha=0.3)
    for i in range(sampler.grid_rows + 1):
        y = sampler.min_y + i * step_y
        plt.axhline(y, color='gray', linestyle='--', alpha=0.3)

    # 3. 绘制生成的点 (蓝色散点)
    if len(generated_points) > 0:
        plt.scatter(generated_points[:, 0], generated_points[:, 1], c='blue', alpha=0.6, s=30, label='Sampled Points')
        
        # 4. 绘制 Yaw 方向 (绿色箭头)
        # 箭头长度
        arrow_len = 20 
        for i in range(len(generated_points)):
            x, y = generated_points[i, 0], generated_points[i, 1]
            yaw = generated_yaws[i]
            dx = arrow_len * math.cos(yaw)
            dy = arrow_len * math.sin(yaw)
            plt.arrow(x, y, dx, dy, head_width=5, head_length=5, fc='green', ec='green', alpha=0.5)
            
    # 设置图形属性
    plt.axis('equal') # 保证 XY 比例一致，这很重要！
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.title(f'Simulation of Your Sampler ({num_samples} points)')
    plt.legend()
    plt.grid(False) # 关闭默认网格，看我们自己画的网格
    
    # 保存或显示
    save_path = "sampler_test_result.png"
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")
    # plt.show() # 如果在有图形界面的环境运行，可以取消注释

if __name__ == "__main__":
    visualize_sampling(100)