import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd
from matplotlib import rcParams

# --------------------------
# 1. 核心修复：字体配置 (Font Configuration)
# --------------------------
# 这一步是关键：设置字体优先级。
# 1. 尝试找 "Times New Roman" (Windows/Mac 常用)
# 2. 尝试找 "Times" (Linux/Mac 常用)
# 3. 都没有，就用 "STIXGeneral" (Matplotlib 内置的 Times 克隆版，绝对安全)
config = {
    "font.family": 'serif',
    "font.serif": ['Times New Roman', 'Times', 'STIXGeneral'],
    "mathtext.fontset": 'stix',  # 数学公式使用 STIX，这本身就是 Times 风格
    "axes.unicode_minus": False, # 解决负号显示问题
    "font.size": 12,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
}
rcParams.update(config)

# --------------------------
# 2. 数据准备
# --------------------------
data_layers = {
    'Layer': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'THU':   [0.8561, 0.8753, 0.8776, 0.8738, 0.8688, 0.8640, 0.8605, 0.8516, 0.8500],
    'CAS':   [0.9181, 0.9239, 0.9246, 0.9232, 0.9192, 0.9187, 0.9154, 0.9139, 0.9124],
    'GIST':  [0.8875, 0.8892, 0.8929, 0.8919, 0.8922, 0.8870, 0.8904, 0.8832, 0.8828]
}

data_heads = {
    'Head': [1, 2, 4, 8],
    'THU':  [0.8790, 0.8792, 0.8793, 0.8790],
    'CAS':  [0.9242, 0.9240, 0.9244, 0.9248],
    'GIST': [0.8937, 0.8961, 0.8959, 0.8920]
}

df_layers = pd.DataFrame(data_layers).melt('Layer', var_name='Dataset', value_name='BA')
df_heads = pd.DataFrame(data_heads).melt('Head', var_name='Dataset', value_name='BA')

# --------------------------
# 3. 绘图 (Plotting)
# --------------------------
# 使用 Lancet 经典配色 (JNE 推荐高对比度)
sci_colors = {'THU': '#00468B', 'CAS': '#ED0000', 'GIST': '#42B540'}
markers = {'THU': 'o', 'CAS': 's', 'GIST': '^'}

# 增大 figsize 的高度，防止内容挤压
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# === 图 A: Layer 的影响 ===
ax1 = axes[0]
sns.lineplot(data=df_layers, x='Layer', y='BA', hue='Dataset', style='Dataset',
             markers=markers, dashes=False, palette=sci_colors,
             markersize=8, linewidth=2, ax=ax1)

ax1.axvline(x=2, color='#555555', linestyle=':', alpha=0.8, linewidth=1.5)
# 注意：这里的 text 字体也会自动跟随 font.serif
ax1.text(2.1, 0.845, 'Optimal $L=2$', color='#333333', fontsize=12, style='italic')

ax1.set_xlabel(r'Transformer Depth ($L$)', fontweight='normal')
ax1.set_ylabel('BA', fontweight='normal')
ax1.set_xticks(range(0, 9))
ax1.set_ylim(0.84, 0.94)

# 将标题位置进一步下移 (y=-0.22)
ax1.set_title('(a) Effect of Transformer Depth', y=-0.22, fontsize=14, color='black')
ax1.legend(loc='upper right', frameon=True, edgecolor='black', fancybox=False, framealpha=1)

# === 图 B: Head 的影响 ===
ax2 = axes[1]
sns.lineplot(data=df_heads, x='Head', y='BA', hue='Dataset', style='Dataset',
             markers=markers, dashes=False, palette=sci_colors,
             markersize=8, linewidth=2, ax=ax2)

ax2.set_xlabel(r'Attention Head Count ($H$)', fontweight='normal')
ax2.set_ylabel('BA', fontweight='normal')
ax2.set_xticks([1, 2, 4, 8])

# 格式化Y轴
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
y_min, y_max = df_heads['BA'].min(), df_heads['BA'].max()
padding = (y_max - y_min) * 0.3
ax2.set_ylim(y_min - padding, y_max + padding)

# 标题位置同步下移
ax2.set_title('(b) Effect of Attention Heads', y=-0.22, fontsize=14, color='black')
ax2.legend(loc='center right', frameon=True, edgecolor='black', fancybox=False, framealpha=1)

# 网格样式
for ax in axes:
    ax.grid(True, linestyle='--', alpha=0.5)

# --------------------------
# 4. 最终布局调整 (Layout)
# --------------------------
# 增加 bottom 边距，确保标题不切底，不和坐标轴打架
plt.subplots_adjust(bottom=0.2, top=0.92, wspace=0.25)

output_path = 'ablation_study_results.pdf'
plt.savefig(output_path, dpi=600, bbox_inches='tight')
plt.show()

# 验证当前使用的字体（调试用）
from matplotlib.font_manager import findfont, FontProperties
print(f"当前正文使用的字体文件路径: {findfont(FontProperties(family=['serif']))}")
print(f"生成的图片已保存为: {output_path}")