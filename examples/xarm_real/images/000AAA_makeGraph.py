# import matplotlib.pyplot as plt
# from PIL import Image
# import matplotlib.gridspec as gridspec
# import numpy as np
# import os

# # 1. 设置九个不同的路径
# image_paths = [
#     "/home/openpi/examples/xarm_real/images/performance_map_exp9_94000_0105rainingDay_test2.png",
#     "/home/openpi/examples/xarm_real/images/exp10_58000_test2.png", 
#     "/home/openpi/examples/xarm_real/images/exp11_30000_test1.png",
#     "/home/openpi/examples/xarm_real/images/exp11_30000_test2.png",
#     "/home/openpi/examples/xarm_real/images/exp11_30000_test3.png", 
#     "/home/openpi/examples/xarm_real/images/0106morning_overcastDay_exp11_48000_test1.png",
#     "/home/openpi/examples/xarm_real/images/0106morning_overcastDay_exp11_48000_test2(components B).png",
#     "/home/openpi/examples/xarm_real/images/0106morning_overcastDay_exp11_48000_test3(components C).png", 
#     "/home/openpi/examples/xarm_real/images/0107morning_overcastDay_exp11_56000_test1(components A).png"
# ]

# # 2. 图片下方较长的备注（支持用 \n 换行）
# long_notes = [
#     "exp9-96000模型，阴天", 
#     "exp10-58000模型，阴天", 
#     "exp11-30000模型，工件A",
#     "exp11-30000模型，工件B",
#     "exp11-30000模型，工件C",
#     "exp11-48000模型，工件A",
#     "exp11-48000模型，工件B",
#     "exp11-48000模型，工件C",
#     "exp11-56000模型，工件A"
# ]

# # 3. 柱状图下方的简短标签（保持整齐）
# short_labels = ["exp9-96000", "exp10-58000", "exp11-30000-A", "exp11-30000-B", "exp11-30000-C", "exp11-48000-A", "exp11-48000-B", "exp11-48000-C", "exp11-56000-A"]

# # 4. 成功率数值
# success_rates = [96.4, 93.8, 89.6, 100.0, 72.9, 95.8, 100.0, 72.9, 100.0]

# # --- 绘图配置 ---
# plt.rcParams['font.sans-serif'] = ['SimHei'] 
# plt.rcParams['axes.unicode_minus'] = False

# # 创建更大比例的画布以容纳长文本
# fig = plt.figure(figsize=(16, 24))
# # height_ratios: 给前三行图片各分配1.0权重，给底部柱状图分配1.4权重（因为有坐标轴）
# gs = gridspec.GridSpec(4, 3, figure=fig, height_ratios=[1, 1, 1, 1.4], hspace=0.4)

# # 5. 循环绘制 3x3 图片阵列
# for i in range(9):
#     row = i // 3
#     col = i % 3
#     ax = fig.add_subplot(gs[row, col])
    
#     path = image_paths[i]
#     if os.path.exists(path):
#         try:
#             img = Image.open(path).convert('RGB')
#             img = img.resize((800, 600)) 
#             ax.imshow(img)
#             # 在图片正下方添加长备注
#             # y=-0.15 表示将文字放在图片下方一定的距离
#             ax.set_title(f"{long_notes[i]}", fontsize=11, y=-0.25, va='top', color='#333333')
#         except:
#             ax.text(0.5, 0.5, "读取失败", ha='center')
#     else:
#         ax.text(0.5, 0.5, f"找不到:\n{os.path.basename(path)}", ha='center', color='red')
    
#     ax.axis('off')

# # 6. 底部绘制柱状图
# ax_bar = fig.add_subplot(gs[3, :])
# colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, 9))
# bars = ax_bar.bar(short_labels, success_rates, color=colors, edgecolor='black', width=0.5)

# # 在柱状图上方标数值
# for bar in bars:
#     h = bar.get_height()
#     ax_bar.text(bar.get_x() + bar.get_width()/2, h + 1, f'{h}%', 
#                 ha='center', va='bottom', fontsize=12, fontweight='bold')

# # 坐标轴与标题
# ax_bar.set_ylabel('成功率 (%)', fontsize=14, fontweight='bold')
# ax_bar.set_xlabel('方案简称', fontsize=14, labelpad=10)
# ax_bar.set_title('各实验方案成功率对比汇总', fontsize=20, pad=30)
# ax_bar.set_ylim(0, 115)
# ax_bar.tick_params(axis='x', labelsize=12) # 设置简短标签的字体大小
# ax_bar.grid(axis='y', linestyle='--', alpha=0.3)

# # 7. 导出
# # bbox_inches='tight' 确保长备注文字不会被裁剪掉
# plt.savefig('experiment_detailed_report.png', dpi=300, bbox_inches='tight')
# plt.show()

import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.gridspec as gridspec
import numpy as np
import os

# 1. 强制非交互后端
plt.switch_backend('Agg') 
# 使用系统自带字体，避免中文乱码
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial'] 
plt.rcParams['axes.unicode_minus'] = False

# 2. 修复路径列表（添加了缺失的逗号）
image_paths = [
    "/home/openpi/examples/xarm_real/images/performance_map_exp9_94000_0105rainingDay_test2.png",
    "/home/openpi/examples/xarm_real/images/exp10_58000_test2.png", 
    "/home/openpi/examples/xarm_real/images/exp11_30000_test1.png",
    "/home/openpi/examples/xarm_real/images/exp11_30000_test2.png",
    "/home/openpi/examples/xarm_real/images/exp11_30000_test3.png", 
    "/home/openpi/examples/xarm_real/images/0106morning_overcastDay_exp11_48000_test1.png",
    "/home/openpi/examples/xarm_real/images/0106morning_overcastDay_exp11_48000_test2(components B).png",
    "/home/openpi/examples/xarm_real/images/0106morning_overcastDay_exp11_48000_test3(components C).png", 
    "/home/openpi/examples/xarm_real/images/0107morning_overcastDay_exp11_56000_test1(components A).png", # 已补逗号
    
    "/home/openpi/examples/xarm_real/images/0107morning_overcastDay_exp11_56000_test3(components B).png",
    "/home/openpi/examples/xarm_real/images/0107morning_overcastDay_exp11_56000_test2(components C).png"
]

# 3. 详细备注（改为英文防止乱码）
long_notes = [
    "exp9-94k\nRainy Day", "exp10-58k\nOvercast", "exp11-30k\nComp A",
    "exp11-30k\nComp B", "exp11-30k\nComp C", "exp11-48k\nComp A",
    "exp11-48k\nComp B", "exp11-48k\nComp C", "exp11-56k\nComp A", 
    "exp11-56k\nComp B", "exp11-56k\nComp C",
]

short_labels = ["exp9", "exp10", "11-30k-A", "11-30k-B", "11-30k-C", "11-48k-A", "11-48k-B", "11-48k-C", "11-56k-A", "11-56k-B", "11-56k-C"]
success_rates = [96.4, 93.8, 89.6, 100.0, 72.9, 95.8, 100.0, 72.9, 100.0, 100.0, 70.8]

# 4. 创建画布
fig = plt.figure(figsize=(16, 30))
# 5行3列布局
gs = gridspec.GridSpec(5, 3, figure=fig, height_ratios=[1, 1, 1, 1, 1.4], hspace=0.6)

# 设置大标题（英文）
fig.suptitle('Pi0.5 Experiments Performance on Different Components', fontsize=26, fontweight='bold', y=0.96)

# 5. 遍历绘制
for i in range(len(image_paths)):
    row = i // 3
    col = i % 3
    ax = fig.add_subplot(gs[row, col])
    
    path = image_paths[i]
    if os.path.exists(path):
        try:
            img = Image.open(path).convert('RGB')
            ax.imshow(img)
            # 在图下方标注（y值需配合hspace调整）
            ax.set_title(long_notes[i], fontsize=12, y=-0.25, va='top', fontweight='bold')
        except:
            ax.text(0.5, 0.5, "Read Error", ha='center')
    else:
        ax.text(0.5, 0.5, "Image Not Found", ha='center', color='red')
    ax.axis('off')

# 6. 绘制柱状图
ax_bar = fig.add_subplot(gs[4, :])
colors = plt.colormaps['tab20'](np.linspace(0, 1, len(success_rates)))
bars = ax_bar.bar(short_labels, success_rates, color=colors, edgecolor='black', width=0.6)

for bar in bars:
    h = bar.get_height()
    ax_bar.text(bar.get_x() + bar.get_width()/2, h + 1, f'{h}%', ha='center', va='bottom', fontweight='bold')

ax_bar.set_ylabel('Success Rate (%)', fontsize=14)
ax_bar.set_title('Comparison of Experimental Success Rates', fontsize=20, pad=20)
ax_bar.set_ylim(0, 120)
ax_bar.grid(axis='y', linestyle='--', alpha=0.3)

# 7. 保存
plt.savefig('final_report_v3.png', dpi=300, bbox_inches='tight')
print("Report generated: final_report_v3.png")