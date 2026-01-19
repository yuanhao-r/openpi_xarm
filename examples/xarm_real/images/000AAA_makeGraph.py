# import matplotlib.pyplot as plt
# from PIL import Image
# import matplotlib.gridspec as gridspec
# import numpy as np
# import os

# # 1. 强制非交互后端
# plt.switch_backend('Agg') 
# # 使用系统自带字体，避免中文乱码
# plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial'] 
# plt.rcParams['axes.unicode_minus'] = False

# # 2. 修复路径列表（添加了缺失的逗号）
# image_paths = [
#     "/home/openpi/examples/xarm_real/images/performance_map_exp9_94000_0105rainingDay_test2.png",
#     "/home/openpi/examples/xarm_real/images/exp10_58000_test2.png", 
#     "/home/openpi/examples/xarm_real/images/exp11_30000_test1.png",
#     "/home/openpi/examples/xarm_real/images/exp11_30000_test2.png",
#     "/home/openpi/examples/xarm_real/images/exp11_30000_test3.png", 
#     "/home/openpi/examples/xarm_real/images/0106morning_overcastDay_exp11_48000_test1.png",
#     "/home/openpi/examples/xarm_real/images/0106morning_overcastDay_exp11_48000_test2(components B).png",
#     "/home/openpi/examples/xarm_real/images/0106morning_overcastDay_exp11_48000_test3(components C).png", 
#     "/home/openpi/examples/xarm_real/images/0107morning_overcastDay_exp11_56000_test1(components A).png", # 已补逗号
    
#     "/home/openpi/examples/xarm_real/images/0107morning_overcastDay_exp11_56000_test3(components B).png",
#     "/home/openpi/examples/xarm_real/images/0107morning_overcastDay_exp11_56000_test2(components C).png"
# ]

# # 3. 详细备注（改为英文防止乱码）
# long_notes = [
#     "exp9-94k\nRainy Day", "exp10-58k\nOvercast", "exp11-30k\nComp A",
#     "exp11-30k\nComp B", "exp11-30k\nComp C", "exp11-48k\nComp A",
#     "exp11-48k\nComp B", "exp11-48k\nComp C", "exp11-56k\nComp A", 
#     "exp11-56k\nComp B", "exp11-56k\nComp C",
# ]

# short_labels = ["exp9", "exp10", "11-30k-A", "11-30k-B", "11-30k-C", "11-48k-A", "11-48k-B", "11-48k-C", "11-56k-A", "11-56k-B", "11-56k-C"]
# success_rates = [96.4, 93.8, 89.6, 100.0, 72.9, 95.8, 100.0, 72.9, 100.0, 100.0, 70.8]

# # 4. 创建画布
# fig = plt.figure(figsize=(16, 30))
# # 5行3列布局
# gs = gridspec.GridSpec(5, 3, figure=fig, height_ratios=[1, 1, 1, 1, 1.4], hspace=0.6)

# # 设置大标题（英文）
# fig.suptitle('Pi0.5 Experiments Performance on Different Components', fontsize=26, fontweight='bold', y=0.96)

# # 5. 遍历绘制
# for i in range(len(image_paths)):
#     row = i // 3
#     col = i % 3
#     ax = fig.add_subplot(gs[row, col])
    
#     path = image_paths[i]
#     if os.path.exists(path):
#         try:
#             img = Image.open(path).convert('RGB')
#             ax.imshow(img)
#             # 在图下方标注（y值需配合hspace调整）
#             ax.set_title(long_notes[i], fontsize=12, y=-0.25, va='top', fontweight='bold')
#         except:
#             ax.text(0.5, 0.5, "Read Error", ha='center')
#     else:
#         ax.text(0.5, 0.5, "Image Not Found", ha='center', color='red')
#     ax.axis('off')

# # 6. 绘制柱状图
# ax_bar = fig.add_subplot(gs[4, :])
# colors = plt.colormaps['tab20'](np.linspace(0, 1, len(success_rates)))
# bars = ax_bar.bar(short_labels, success_rates, color=colors, edgecolor='black', width=0.6)

# for bar in bars:
#     h = bar.get_height()
#     ax_bar.text(bar.get_x() + bar.get_width()/2, h + 1, f'{h}%', ha='center', va='bottom', fontweight='bold')

# ax_bar.set_ylabel('Success Rate (%)', fontsize=14)
# ax_bar.set_title('Comparison of Experimental Success Rates', fontsize=20, pad=20)
# ax_bar.set_ylim(0, 120)
# ax_bar.grid(axis='y', linestyle='--', alpha=0.3)

# # 7. 保存
# plt.savefig('final_report_v3.png', dpi=300, bbox_inches='tight')
# print("Report generated: final_report_v3.png")


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
    "/home/openpi/examples/xarm_real/images/0119morning_rainDay_exp22_16000_test2(components A).png",
    "/home/openpi/examples/xarm_real/images/0119morning_rainDay_exp22_16000_test3(components B).png", 
    "/home/openpi/examples/xarm_real/images/0119morning_rainDay_exp22_16000_test4(components C).png",
    "/home/openpi/examples/xarm_real/images/0119morning_rainDay_exp22_16000_test1(components D).png"
]

# 3. 详细备注（改为英文防止乱码）
long_notes = [
    "exp22-16k\nRainy Day\nA", "exp22-16k\nRainy Day\nB", "exp22-16k\nRainy Day\nC",
    "exp22-16k\nRainy Day\nD", 
]

short_labels = ["exp22-16k-A", "exexp22-16k-B", "exp22-16k-C", "exp22-16k-D"]
success_rates = [88.9, 83.3, 100.0, 97.2]

# 4. 创建画布
fig = plt.figure(figsize=(16, 30))
# 5行3列布局
# gs = gridspec.GridSpec(5, 3, figure=fig, height_ratios=[1, 1, 1, 1, 1.4], hspace=0.6)
gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1.5], hspace=0.6)

# 设置大标题（英文）
# fig.suptitle('Pi0.5 Experiments Performance on Different Components', fontsize=26, fontweight='bold', y=0.96)
fig.suptitle('exp22 16000model 0119rainDay_test', fontsize=26, fontweight='bold', y=0.96)

# 5. 遍历绘制
for i in range(len(image_paths)):
    # row = i // 3
    # col = i % 3
    row = i // 2
    col = i % 2
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
# ax_bar = fig.add_subplot(gs[4, :])
ax_bar = fig.add_subplot(gs[2, :])

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