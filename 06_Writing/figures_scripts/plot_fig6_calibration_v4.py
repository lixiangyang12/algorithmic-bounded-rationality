#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图6 难度-校准反转（修正版 V4 · 带外框）
面板顺序与修订后正文一致：(a) 订货量偏差 (b) 置信区间覆盖率 (c) 区间宽度与中心偏移
以各难度理性最优为参照（σ=10→107, σ=30→122, σ=60→144）
数据：04_Data/raw/exp5_overconfidence.jsonl 原始记录重算
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
RAW = BASE_DIR / '04_Data' / 'raw' / 'exp5_overconfidence.jsonl'
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

recs = [json.loads(l) for l in open(RAW, encoding='utf-8') if l.strip()]
groups = defaultdict(list)
for d in recs:
    groups[d['condition']].append(d)

QSTAR = {'E5_easy': 107, 'E5_medium': 122, 'E5_hard': 144}
LABELS = ['简单\n(σ=10)', '中等\n(σ=30)', '困难\n(σ=60)']
conds = ['E5_easy', 'E5_medium', 'E5_hard']

bias, coverage, ce, widths, offsets, cov_k = [], [], [], [], [], []
for c in conds:
    g = groups[c]
    q = np.array([d['order_quantity'] for d in g])
    bias.append((q.mean() - QSTAR[c]) / QSTAR[c] * 100)
    cov = sum(1 for d in g if d['Q_low'] <= QSTAR[c] <= d['Q_high'])
    cov_k.append((cov, len(g)))
    coverage.append(cov / len(g) * 100)
    ce.append(abs(90 - cov / len(g) * 100))
    w = np.array([d['Q_high'] - d['Q_low'] for d in g])
    widths.append(w.mean())
    offsets.append(np.mean([abs((d['Q_high'] + d['Q_low']) / 2 - QSTAR[c]) for d in g]))

fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, FULL_WIDTH * 0.5))
x = np.arange(3)
colors_b = [COLORS['green'], COLORS['blue'], COLORS['red']]

# ---- 面板(a) 订货量偏差（对真最优）----
ax = axes[0]
bars = ax.bar(x, bias, width=0.5, color=colors_b, alpha=0.85, edgecolor='white', linewidth=0.5)
for i, b in enumerate(bars):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() - 1.2,
            f'{bias[i]:.1f}%', ha='center', va='top', fontproperties=font_times,
            fontsize=7, color='white', fontweight='bold')
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6,
            f'Q*={QSTAR[conds[i]]}', ha='center', va='bottom', fontproperties=font_times,
            fontsize=6.3, color='#333333')
ax.axhline(y=0, color=COLORS['gray'], linestyle=':', lw=0.8, alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(LABELS, fontproperties=font_song, fontsize=7)
ax.set_ylabel('订货量偏差（对真最优Q*）(%)', fontproperties=font_song, fontsize=7.5)
ax.set_ylim(-22, 2)
ax.tick_params(axis='both', labelsize=7.5)
ax.set_title('(a) 订货量偏差（随难度放大）', fontproperties=font_hei_9, fontsize=9, pad=6)

# ---- 面板(b) 覆盖率 ----
ax = axes[1]
bars = ax.bar(x, coverage, width=0.5, color=colors_b, alpha=0.85, edgecolor='white', linewidth=0.5)
ax.axhline(y=90, color=COLORS['gray'], linestyle='--', lw=1, alpha=0.7)
ax.text(2.45, 91.5, '名义90%', fontproperties=font_song, fontsize=7, color=COLORS['gray'], ha='right')
for i, (b, (k, n)) in enumerate(zip(bars, cov_k)):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2,
            f'{coverage[i]:.1f}%\n({k}/{n})', ha='center', va='bottom',
            fontproperties=font_times, fontsize=6.8)
ax.set_xticks(x)
ax.set_xticklabels(LABELS, fontproperties=font_song, fontsize=7)
ax.set_ylabel('90%置信区间覆盖率 (%)', fontproperties=font_song, fontsize=7.5)
ax.set_ylim(0, 108)
ax.tick_params(axis='both', labelsize=7.5)
ax.set_title('(b) 区间覆盖率（对真最优Q*）', fontproperties=font_hei_9, fontsize=9, pad=6)
ax.text(0.5, 0.9, '二项检验均 p<0.001', transform=ax.transAxes, fontproperties=font_song,
        fontsize=6.5, ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#cccccc'))

# ---- 面板(c) 区间宽度与中心偏移 ----
ax = axes[2]
bars = ax.bar(x - 0.19, widths, width=0.38, color=COLORS['blue'], alpha=0.85,
              edgecolor='white', linewidth=0.5, label='平均区间宽度')
for i, b in enumerate(bars):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, f'{widths[i]:.1f}',
            ha='center', fontproperties=font_times, fontsize=6.8)
ax2 = ax.twinx()
ax2.plot(x, offsets, 'o-', color=COLORS['red'], lw=1.5, markersize=5,
         markerfacecolor='white', markeredgecolor=COLORS['red'], label='中心与真最优距离')
for i, v in enumerate(offsets):
    ax2.text(x[i] + 0.1, v + 0.8, f'{v:.1f}', color=COLORS['red'],
             fontproperties=font_times, fontsize=6.8)
ax.set_xticks(x)
ax.set_xticklabels(LABELS, fontproperties=font_song, fontsize=7)
ax.set_ylabel('平均区间宽度 (件)', fontproperties=font_song, fontsize=7.5)
ax2.set_ylabel('区间中心与真最优距离 (件)', fontproperties=font_song, fontsize=7.5)
ax.set_ylim(0, 55)
ax2.set_ylim(0, 30)
ax.tick_params(axis='both', labelsize=7.5)
ax2.tick_params(axis='y', labelsize=7.5)
ax.set_title('(c) 区间宽度与中心偏移', fontproperties=font_hei_9, fontsize=9, pad=6)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, prop=font_song, fontsize=6.5,
          loc='upper left', framealpha=0.9, edgecolor='#cccccc')

plt.tight_layout(pad=2)
fig.text(0.5, 0.01, '图6 难度-校准反转：任务难度对决策偏差与置信度校准的影响（按各难度理性最优参照）',
         ha='center', va='bottom', fontproperties=font_hei, fontsize=10.5, fontweight='bold')
for ext in ('svg', 'pdf', 'png'):
    fig.savefig(str(FIG_DIR / ext / f'fig6_calibration_reversal_v4.{ext}'),
                format=ext, dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.close(fig)
print('已生成 fig6_calibration_reversal_v4（面板顺序: 偏差/覆盖率/宽度）')
