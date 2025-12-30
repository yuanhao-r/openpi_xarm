import numpy as np
import matplotlib
# matplotlib.use('TkAgg') # 或者 'Qt5Agg'
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import random
import time

class SamplerVisualizer:
    def __init__(self):
        # === 1. 定义与你代码一致的区域 ===
        self.boundary_points_2d = np.array([
            [505.982422, -150.631149],  # 左下
            [712.302856, -66.848724],   # 右下
            [697.232117, 163.981003],   # 右上
            [466.805481, 144.618057],   # 左上
        ])
        
        self.hull_2d = ConvexHull(self.boundary_points_2d)
        self.hull_points_2d = self.boundary_points_2d[self.hull_2d.vertices]
        
        self.min_x = np.min(self.boundary_points_2d[:, 0])
        self.max_x = np.max(self.boundary_points_2d[:, 0])
        self.min_y = np.min(self.boundary_points_2d[:, 1])
        self.max_y = np.max(self.boundary_points_2d[:, 1])

        # === 网格采样参数 ===
        self.grid_rows = 5
        self.grid_cols = 5
        self.grid_indices = []
        self._refill_grid_indices()

    def _refill_grid_indices(self):
        self.grid_indices = []
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                self.grid_indices.append((r, c))
        random.shuffle(self.grid_indices)

    def is_point_inside_hull_2d(self, point_2d):
        hull_eq = self.hull_2d.equations
        point_homo = np.hstack([point_2d, 1])
        for eq in hull_eq:
            if np.dot(eq, point_homo) > 1e-6:
                return False
        return True

    # === 模拟机械臂的可达性检查 ===
    def mock_check_reachable(self, pos):
        """
        模拟真实情况：
        假设边缘区域或随机某些点不可达。
        为了演示效果，我们随机让 20% 的点变成“不可达”。
        """
        # 简单模拟：如果随机数小于0.2，认为该点奇异或不可达
        # 在真实环境中，这通常发生在凸包的尖端或边缘
        if random.random() < 0.2:
            return False
        return True

    # ==========================================
    # 算法 1: 你原始的逻辑 (随机 + 回缩)
    # ==========================================
    def sample_original_logic(self):
        # 1. 随机采样
        max_attempts = 100
        sample_point = None
        
        # 尝试找到凸包内的点
        for _ in range(max_attempts):
            x = random.uniform(self.min_x, self.max_x)
            y = random.uniform(self.min_y, self.max_y)
            pt = np.array([x, y])
            if self.is_point_inside_hull_2d(pt):
                sample_point = pt
                break
        
        if sample_point is None: return None

        # 2. 可达性检查 + 回缩机制
        candidate = sample_point.copy()
        region_center = np.mean(self.hull_points_2d, axis=0)
        
        for i in range(20):
            # 模拟检查
            if self.mock_check_reachable(candidate):
                return candidate
            
            # !!! 这里的回缩是你原本代码的问题所在 !!!
            vec = candidate - region_center
            candidate = region_center + vec * 0.9 
            
        return region_center # 最终失败返回中心

    # ==========================================
    # 算法 2: 改进后的逻辑 (网格 + 丢弃)
    # ==========================================
    def sample_new_grid_logic(self):
        max_total_attempts = 200
        for _ in range(max_total_attempts):
            if not self.grid_indices:
                self._refill_grid_indices()
            
            r, c = self.grid_indices.pop()
            
            # 计算网格边界
            step_x = (self.max_x - self.min_x) / self.grid_cols
            step_y = (self.max_y - self.min_y) / self.grid_rows
            
            cell_min_x = self.min_x + c * step_x
            cell_max_x = cell_min_x + step_x
            cell_min_y = self.min_y + r * step_y
            cell_max_y = cell_min_y + step_y
            
            # 在网格内尝试采样
            valid_pt = None
            for _ in range(20):
                x = random.uniform(cell_min_x, cell_max_x)
                y = random.uniform(cell_min_y, cell_max_y)
                pt = np.array([x, y])
                if self.is_point_inside_hull_2d(pt):
                    valid_pt = pt
                    break
            
            if valid_pt is None: continue # 该格子不在凸包内，跳过

            # 模拟可达性检查 (如果不通过，直接丢弃，不回缩！)
            if self.mock_check_reachable(valid_pt):
                return valid_pt
            
            # 如果不可达，这里直接 continue，程序会去取下一个格子
            # 从而保证了分布不会被人为挤压
            
        return np.mean(self.hull_points_2d, axis=0)

# ==========================================
# 主绘图逻辑
# ==========================================
def run_visualization():
    sim = SamplerVisualizer()
    num_samples = 300 # 模拟采集 300 个数据点
    
    print(f"开始模拟生成 {num_samples} 个点...")
    
    # 1. 生成原版数据
    points_old = []
    for _ in range(num_samples):
        pt = sim.sample_original_logic()
        if pt is not None: points_old.append(pt)
    points_old = np.array(points_old)

    # 2. 生成新版数据
    points_new = []
    # 重置网格
    sim._refill_grid_indices() 
    for _ in range(num_samples):
        pt = sim.sample_new_grid_logic()
        if pt is not None: points_new.append(pt)
    points_new = np.array(points_new)

    # 3. 绘图对比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 画凸包轮廓的辅助函数
    def plot_hull(ax, title, points):
        # 画凸包边界
        hull_pts = np.vstack([sim.hull_points_2d, sim.hull_points_2d[0]])
        ax.plot(hull_pts[:,0], hull_pts[:,1], 'k--', lw=2, label='Boundary')
        
        # 画采样点
        ax.scatter(points[:,0], points[:,1], c='blue', alpha=0.6, s=15, label='Sampled Points')
        
        # 画中心点
        center = np.mean(sim.hull_points_2d, axis=0)
        ax.scatter(center[0], center[1], c='red', marker='x', s=100, label='Center')
        
        ax.set_title(title)
        ax.axis('equal')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)

    # --- 左图：原版 ---
    plot_hull(ax1, f"Original: Random + Shrink\n(Simulated 20% Unreachable)", points_old)
    
    # --- 右图：新版 ---
    plot_hull(ax2, f"Improved: Grid Stratified\n(Drop if Unreachable)", points_new)
    
    # 在右图画出网格线方便理解
    step_x = (sim.max_x - sim.min_x) / sim.grid_cols
    step_y = (sim.max_y - sim.min_y) / sim.grid_rows
    for i in range(sim.grid_cols + 1):
        x = sim.min_x + i * step_x
        ax2.axvline(x, color='gray', alpha=0.3)
    for i in range(sim.grid_rows + 1):
        y = sim.min_y + i * step_y
        ax2.axhline(y, color='gray', alpha=0.3)

    plt.tight_layout()
    # plt.show()
    plt.savefig('/home/openpi/record_and_transform/view_result.png')
    print("图片已保存至 view_result.png")

if __name__ == "__main__":
    run_visualization()