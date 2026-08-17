#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图7 三种纠偏机制效果对比（合并版 V4 · 带外框）
面板(a) 纠偏后订货量分布（箱线图）
面板(b) 哑铃图：各机制偏差改善幅度（基线→纠偏后）
数据：04_Data/raw 原始记录 + results/all_experiments_summary.csv
"""
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / '04_Data' / 'raw'
FIG_DIR = BASE_DIR / '06_Writing' / 'figures'
FONT_SIMHEI = Path('C:/Windows/Fonts/simhei.ttf')
FONT_SIMSUN = Path('C:/Windows/Fonts/simsun.ttc')
FONT_TIMES = Path('C:/Windows/Fonts/times.ttf')
for f in (FONT_SIMHEI, FONT_SIMSUN, FONT_TIMES):
    fontManager.addfont(str(f))

font_hei = FontProperties(fname=str(FONT_SIMHEI), size=10.5)
font_hei_9 = FontProperties(fname=str(FONT_SIMHEI), size=9)
font_song = FontProperties(fname=str(FONT_SIMSUN), size=7.5)
font_times = FontProperties(fname=str(FONT_TIMES), size=7.5)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei', 'SimSun', 'Times New Roman', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42

COLORS = {'blue': '#457B9D', 'red': '#E63946', 'green': '#2A9D8F',
          'orange': '#E76F51', 'gray': '#6D6875', 'dark_blue': '#1D3557'}
MM_TO_INCH = 1 / 25.4
FULL_WIDTH = 170 * MM_TO_INCH
Q_STAR = 122

def load(fn):
    return [json.loads(l) for l in open(RAW_DIR / fn, encoding='utf-8') if l.strip()]

# 数据
e2 = load('exp2_anchoring.jsonl')
e6 = load('exp6_cot.jsonl')
e7 = load('exp7_debate.jsonl')
e8 = load('exp8_human_calibration.jsonl')
high_anchor = [d['order_quantity'] for d in e2 if d['condition'] == 'E2_anchor_high']
cot_q = [d['order_quantity'] for d in e6]
debate_q = [d['order_quantity'] for d in e7]
light_q = [d['order_quantity'] for d in e8 if d['condition'] == 'E8_light']
strong_q = [d['order_quantity'] for d in e8 if d['condition'] == 'E8_strong']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, FULL_WIDTH * 0.5))

# ---- 面板(a) 纠偏后订货量分布 ----
groups = [high_anchor, cot_q, debate_q, light_q, strong_q]
labels = ['E2高锚定\n(基线)', 'E6 CoT', 'E7 辩论', 'E8 轻反馈', 'E8 强反馈']
positions = [1, 2, 3, 4, 5]
colors_a = [COLORS['red'], COLORS['green'], COLORS['green'], COLORS['blue'], COLORS['green']]
bp = ax1.boxplot(groups, positions=positions, widths=0.5, patch_artist=True,
                 showfliers=True, showmeans=True,
                 meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='black', markersize=5),
                 flierprops=dict(marker='o', markerfacecolor='gray', markersize=3, alpha=0.5),
                 medianprops=dict(color='black', lw=1.5))
for patch, color in zip(bp['boxes'], colors_a):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
for i, data in enumerate(groups):
    jitter = np.random.normal(0, 0.04, len(data))
    ax1.scatter(np.ones(len(data)) * positions[i] + jitter, data, color=colors_a[i],
                alpha=0.45, s=14, edgecolors='white', linewidth=0.3, zorder=5)
ax1.axhline(y=Q_STAR, color=COLORS['gray'], linestyle='--', lw=1.0, alpha=0.7)
ax1.text(5.25, Q_STAR + 1, 'Q*=122', fontproperties=font_song, fontsize=7,
         color=COLORS['gray'], ha='right')
ax1.set_xticks(positions)
ax1.set_xticklabels(labels, fontproperties=font_song, fontsize=7)
ax1.set_ylabel('订货量 (件)', fontproperties=font_song, fontsize=7.5)
ax1.set_ylim(100, 128)
ax1.tick_params(axis='both', labelsize=7.5)
ax1.set_title('(a) 纠偏后订货量分布', fontproperties=font_hei_9, fontsize=9, pad=6)

# ---- 面板(b) 哑铃图：基线→纠偏后偏差 ----
mechanisms = ['CoT推理', '多智能体辩论', '人类反馈(强)', '人类反馈(轻)']
base = [-6.7, -7.5, -7.5, -7.5]
post = [+0.1, -1.6, -1.5, -5.9]
ypos = np.arange(len(mechanisms))[::-1]
for i, y in enumerate(ypos):
    ax2.plot([base[i], post[i]], [y, y], color=COLORS['gray'], lw=2.0, alpha=0.9, zorder=1)
    ax2.scatter(base[i], y, color=COLORS['red'], s=55, zorder=3, edgecolors='white', linewidth=0.6)
    ax2.scatter(post[i], y, color=COLORS['green'], s=55, zorder=3, edgecolors='white', linewidth=0.6)
    ax2.text(base[i] - 0.3, y, f'{base[i]:+.1f}%', ha='right', va='center',
             fontproperties=font_times, fontsize=6.8, color=COLORS['red'])
    ax2.text(post[i] + 0.3, y, f'{post[i]:+.1f}%', ha='left', va='center',
             fontproperties=font_times, fontsize=6.8, color=COLORS['green'])
ax2.axvline(x=0, color=COLORS['gray'], linestyle=':', lw=0.8, alpha=0.5)
ax2.set_yticks(ypos)
ax2.set_yticklabels(mechanisms, fontproperties=font_song, fontsize=7)
ax2.set_xlabel('订货量偏差 (%)', fontproperties=font_song, fontsize=7.5)
ax2.set_xlim(-9, 2.5)
ax2.tick_params(axis='both', labelsize=7.5)
ax2.set_title('(b) 偏差改善幅度（基线→纠偏后）', fontproperties=font_hei_9, fontsize=9, pad=6)
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='none', markerfacecolor=COLORS['red'], markersize=7, label='基线'),
    Line2D([0], [0], marker='o', color='none', markerfacecolor=COLORS['green'], markersize=7, label='纠偏后'),
]
ax2.legend(handles=legend_elements, prop=font_song, fontsize=6.5, loc='lower right',
           framealpha=0.9, edgecolor='#cccccc')

plt.tight_layout(pad=2)
fig.text(0.5, 0.01, '图7 三种纠偏机制效果对比', ha='center', va='bottom',
         fontproperties=font_hei, fontsize=10.5, fontweight='bold')
for ext in ('svg', 'pdf', 'png'):
    fig.savefig(str(FIG_DIR / ext / f'fig7_debiasing_merged_v4.{ext}'),
                format=ext, dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.close(fig)
print('已生成 fig7_debiasing_merged_v4 (svg/pdf/png)')
