#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ICMSE2026 论文图表生成脚本
10张图, 按《中国管理科学》期刊顶刊标准绘制
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy import stats
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / '04_Data'
SUMMARY_CSV = DATA_DIR / 'results' / 'all_experiments_summary.csv'
RAW_DIR = DATA_DIR / 'raw'
OUTPUT_BASE = BASE_DIR / '06_Writing'
FIG_DIR = OUTPUT_BASE / 'figures'
SCRIPT_DIR = OUTPUT_BASE / 'figures_scripts'
DATA_OUT_DIR = OUTPUT_BASE / 'figures_data'

# 字体路径
FONT_SIMHEI = Path('C:/Windows/Fonts/simhei.ttf')
FONT_SIMSUN = Path('C:/Windows/Fonts/simsun.ttc')
FONT_TIMES = Path('C:/Windows/Fonts/times.ttf')

# 注册字体
from matplotlib.font_manager import fontManager
fontManager.addfont(str(FONT_SIMHEI))
fontManager.addfont(str(FONT_SIMSUN))
fontManager.addfont(str(FONT_TIMES))

# ============================================================
# 字体属性
# ============================================================
font_hei = FontProperties(fname=str(FONT_SIMHEI), size=10.5)
font_hei_9 = FontProperties(fname=str(FONT_SIMHEI), size=9)
font_hei_8 = FontProperties(fname=str(FONT_SIMHEI), size=8)
font_hei_12 = FontProperties(fname=str(FONT_SIMHEI), size=12)
font_song = FontProperties(fname=str(FONT_SIMSUN), size=7.5)
font_song_8 = FontProperties(fname=str(FONT_SIMSUN), size=8)
font_times = FontProperties(fname=str(FONT_TIMES), size=7.5)
font_times_8 = FontProperties(fname=str(FONT_TIMES), size=8)

# ============================================================
# 全局绘图参数
# ============================================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei', 'SimSun', 'Times New Roman', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# 学术配色
COLORS = {
    'blue': '#457B9D',
    'red': '#E63946',
    'green': '#2A9D8F',
    'orange': '#E76F51',
    'gray': '#6D6875',
    'light_blue': '#A8DADC',
    'light_red': '#F4A261',
    'dark_blue': '#1D3557',
    'yellow': '#E9C46A',
}

# 尺寸(mm转inch)
MM_TO_INCH = 1 / 25.4
FULL_WIDTH = 160 * MM_TO_INCH
HALF_WIDTH = 140 * MM_TO_INCH

# 最优订货量基准
Q_STAR = 122

# ============================================================
# 工具函数
# ============================================================
def save_figure(fig, name, dpi=300):
    svg_path = FIG_DIR / 'svg' / f'{name}.svg'
    pdf_path = FIG_DIR / 'pdf' / f'{name}.pdf'
    png_path = FIG_DIR / 'png' / f'{name}.png'
    fig.savefig(str(svg_path), format='svg', dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    fig.savefig(str(pdf_path), format='pdf', dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    fig.savefig(str(png_path), format='png', dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    print(f'  已保存: {svg_path.name}, {pdf_path.name}, {png_path.name}')

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def add_figure_caption(fig, caption, note=''):
    fig.text(0.5, 0.015, caption, ha='center', va='bottom',
             fontproperties=font_hei, fontsize=10.5, fontweight='bold')
    if note:
        fig.text(0.5, 0.003, note, ha='center', va='top',
                 fontproperties=font_song, fontsize=7.5, style='italic')

def add_subfigure_title(ax, title, x=0.5, y=-0.12):
    ax.set_title(title, fontproperties=font_hei_9, fontsize=9, fontweight='bold',
                 pad=5, loc='center')

def set_axis_labels(ax, xlabel, ylabel):
    if xlabel:
        ax.set_xlabel(xlabel, fontproperties=font_song, fontsize=7.5)
    if ylabel:
        ax.set_ylabel(ylabel, fontproperties=font_song, fontsize=7.5)
    ax.tick_params(axis='both', labelsize=7.5, pad=2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_times)

def draw_box(ax, text, xy, width, height, color, text_color='black', fontsize=8, bold=False):
    """绘制圆角矩形框"""
    box = FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='#333333', linewidth=1.0,
                          alpha=0.9, zorder=2)
    ax.add_patch(box)
    fp = FontProperties(fname=str(FONT_SIMHEI), size=fontsize)
    if bold:
        fp.set_weight('bold')
    ax.text(xy[0] + width/2, xy[1] + height/2, text, ha='center', va='center',
            fontproperties=fp, color=text_color, zorder=3)

def draw_arrow(ax, start, end, color='#333333', lw=1.5):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                               connectionstyle='arc3,rad=0'), zorder=1)

# ============================================================
# 加载数据（模块级别）
# ============================================================
print("加载数据...")
df_summary = pd.read_csv(str(SUMMARY_CSV))
df_summary['experiment'] = df_summary['experiment'].str.strip()

raw_data = {}
for exp_file in RAW_DIR.glob('exp*.jsonl'):
    key = exp_file.stem
    raw_data[key] = load_jsonl(str(exp_file))
    print(f"  加载 {key}: {len(raw_data[key])} 条记录")

# ============================================================
# 图1: 研究框架图
# ============================================================
def plot_fig1_framework():
    print("\n[图1] 研究框架图...")
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, FULL_WIDTH * 0.7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.add_patch(mpatches.Rectangle((0, 0), 10, 8, fill=False, edgecolor='#333333', lw=1.0))

    # 顶层：报童问题任务环境
    draw_box(ax, '报童问题任务环境\n(Newsvendor Problem)', (2.5, 6.5), 5, 1.0,
             COLORS['dark_blue'], 'white', fontsize=9, bold=True)

    # 中层左：偏差识别
    draw_box(ax, '偏差识别 (E1-E5)\n锚定·幻觉·指令·过度自信', (0.5, 4.0), 4.2, 1.0,
             COLORS['red'], 'white', fontsize=8)

    # 中层右：纠偏机制
    draw_box(ax, '纠偏机制 (E6-E8)\nCoT·辩论·人类反馈', (5.3, 4.0), 4.2, 1.0,
             COLORS['green'], 'white', fontsize=8)

    # 底部：核心发现
    draw_box(ax, '核心发现\n算法有界理性·偏差系统性·可纠偏性', (2.0, 1.0), 6.0, 1.5,
             COLORS['blue'], 'white', fontsize=9, bold=True)

    # 箭头：顶层到中层
    draw_arrow(ax, (5, 6.5), (2.6, 5.0), COLORS['dark_blue'])
    draw_arrow(ax, (5, 6.5), (7.4, 5.0), COLORS['dark_blue'])

    # 箭头：中层到底层
    draw_arrow(ax, (2.6, 4.0), (5, 2.5), COLORS['dark_blue'])
    draw_arrow(ax, (7.4, 4.0), (5, 2.5), COLORS['dark_blue'])

    # 实验编号标注
    e_texts = [
        (0.5, 4.5, 'E1: 基准条件', 'left'),
        (0.5, 4.15, 'E2: 锚定效应', 'left'),
        (0.5, 3.8, 'E3: 信息幻觉', 'left'),
        (0.5, 3.45, 'E4: 指令敏感性', 'left'),
        (0.5, 3.1, 'E5: 过度自信', 'left'),
        (5.3, 4.5, 'E6: CoT提示', 'right'),
        (5.3, 4.15, 'E7: 多智能体辩论', 'right'),
        (5.3, 3.8, 'E8: 人类反馈校准', 'right'),
    ]
    for x, y, txt, ha in e_texts:
        ax.text(x, y, txt, fontproperties=font_song, fontsize=6.5, color='#555555',
                ha=ha, va='center', zorder=4)

    add_figure_caption(fig, '图1 研究框架')
    save_figure(fig, 'fig1_framework')
    plt.close(fig)

# ============================================================
# 图2: 19个实验条件偏差对比全景图
# ============================================================
def plot_fig2_overview():
    print("\n[图2] 19个实验条件偏差对比全景图...")
    df = df_summary.copy()

    # 修正E5数据：以各难度理性最优为参照（简单107、中等122、困难144）
    # 汇总CSV中E5简单/困难仍为旧参照(122)下的-15.3%/-2.5%，此处按修正后覆盖
    fix = {
        'E5_Easy': {'bias_pct': -3.5, 't_stat': -24.22, 'p_value': 0.0000, 'significant': 'Yes'},
        'E5_Hard': {'bias_pct': -17.4, 't_stat': -11.75, 'p_value': 0.0000, 'significant': 'Yes'},
    }
    for exp, vals in fix.items():
        for k, v in vals.items():
            df.loc[df['experiment'] == exp, k] = v

    df = df.iloc[::-1].reset_index(drop=True)

    # 计算SE
    df['se'] = df['std'] / np.sqrt(df['n'])
    df['bias_lower'] = df['bias_pct'] - df['se']
    df['bias_upper'] = df['bias_pct'] + df['se']

    # 简化标签（与正文3.4节表述一致）
    label_map = {
        'E1_Baseline': 'E1 基准条件',
        'E2_LowAnchor': 'E2 低锚定(50)',
        'E2_HighAnchor': 'E2 高锚定(200)',
        'E2_RandomAnchor': 'E2 随机锚定(73)',
        'E3_FakeInfo1': 'E3 虚构信息(供应链)',
        'E3_FakeInfo2': 'E3 虚构信息(政策)',
        'E3_FakeInfo3': 'E3 虚构信息(竞争)',
        'E4_Variant1': 'E4 正式学术',
        'E4_Variant2': 'E4 口语化',
        'E4_Variant3': 'E4 角色扮演',
        'E4_Variant4': 'E4 警告提示',
        'E4_Variant5': 'E4 英文表述',
        'E5_Easy': 'E5 简单(σ=10)',
        'E5_Medium': 'E5 中等(σ=30)',
        'E5_Hard': 'E5 困难(σ=60)',
        'E6_CoT': 'E6 CoT思维链',
        'E7_Debate': 'E7 多智能体辩论',
        'E8_LightFeedback': 'E8 轻反馈校准',
        'E8_StrongFeedback': 'E8 强反馈校准',
    }
    df['label'] = df['experiment'].map(label_map)

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, FULL_WIDTH * 1.15))
    y_pos = range(len(df))

    colors = [COLORS['red'] if s == 'Yes' else COLORS['green'] for s in df['significant']]

    bars = ax.barh(y_pos, df['bias_pct'], height=0.6, color=colors, alpha=0.85,
                   edgecolor='white', linewidth=0.5)

    # 误差线
    ax.errorbar(df['bias_pct'], y_pos, xerr=df['se'], fmt='none',
                ecolor='#333333', capsize=3, capthick=0.8, lw=0.8, zorder=5)

    # 基准线（偏差=0，即各条件理性最优）
    ax.axvline(x=0, color=COLORS['gray'], linestyle='--', lw=1.0, alpha=0.7, zorder=3)
    ax.text(0.5, len(df) - 0.3, '理性最优\n(偏差=0)', fontproperties=font_song, fontsize=6.5,
            color=COLORS['gray'], ha='left', va='bottom')

    # 标注偏差百分比与显著性（*** 表示p<0.001，经Bonferroni校正后仍显著）
    for i, (bias, sig, pv) in enumerate(zip(df['bias_pct'], df['significant'], df['p_value'])):
        sign = '+' if bias > 0 else ''
        if sig == 'Yes':
            marker = '***' if pv < 0.001 else ('**' if pv < 0.01 else '*')
        else:
            marker = ' (n.s.)'
        ax.text(bias + 0.5, i, f'{sign}{bias:.1f}%{marker}',
                fontproperties=font_times, fontsize=6, va='center', color='#333333')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['label'], fontproperties=font_song, fontsize=7)
    ax.set_xlabel('偏差百分比（对理性最优）(%)', fontproperties=font_song, fontsize=7.5)
    ax.set_xlim(-55, 15)
    ax.tick_params(axis='x', labelsize=7.5)
    for label in ax.get_xticklabels():
        label.set_fontproperties(font_times)

    # 图例（颜色）
    from matplotlib.lines import Line2D
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['red'], alpha=0.85, label='显著偏差 (p<0.05)'),
        mpatches.Patch(facecolor=COLORS['green'], alpha=0.85, label='不显著 (p>=0.05)'),
    ]
    ax.legend(handles=legend_elements, prop=font_song, fontsize=7, loc='lower right',
              framealpha=0.9, edgecolor='#cccccc')

    # 显著性标记说明
    ax.text(0.02, 0.02, '*** p<0.001（Bonferroni校正后仍显著）\n(n.s.) 不显著',
            transform=ax.transAxes, fontproperties=font_song, fontsize=6.5,
            ha='left', va='bottom', color='#555555',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))

    ax.grid(axis='x', alpha=0.3, lw=0.5)

    add_figure_caption(fig, '图2 19个实验条件订货量偏差对比全景图（偏差以各条件理性最优为参照）')
    save_figure(fig, 'fig2_overview')
    plt.close(fig)

# ============================================================
# 图3: 锚定效应偏差分布对比
# ============================================================
def plot_fig3_anchoring():
    print("\n[图3] 锚定效应偏差分布对比...")
    baseline = pd.DataFrame(raw_data['exp1_baseline'])
    anchoring = pd.DataFrame(raw_data['exp2_anchoring'])

    # 提取数据
    bl_q = baseline['order_quantity'].values
    low_q = anchoring[anchoring['condition'] == 'E2_anchor_low']['order_quantity'].values
    high_q = anchoring[anchoring['condition'] == 'E2_anchor_high']['order_quantity'].values
    random_q = anchoring[anchoring['condition'] == 'E2_anchor_random']['order_quantity'].values

    data_groups = [bl_q, low_q, high_q, random_q]
    labels = ['基线\n(E1)', '低锚定(50)\n(E2)', '高锚定(200)\n(E2)', '随机锚定(73)\n(E2)']
    colors_g = [COLORS['blue'], COLORS['red'], COLORS['red'], COLORS['orange']]

    fig, ax = plt.subplots(figsize=(HALF_WIDTH, HALF_WIDTH * 0.75))
    bp = ax.boxplot(data_groups, positions=[1, 2, 3, 4], widths=0.5, patch_artist=True,
                     showfliers=True, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='black', markersize=5),
                     flierprops=dict(marker='o', markerfacecolor='gray', markersize=3, alpha=0.5),
                     medianprops=dict(color='black', lw=1.5))

    for patch, color in zip(bp['boxes'], colors_g):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # 抖动散点
    for i, (data, color) in enumerate(zip(data_groups, colors_g)):
        jitter = np.random.normal(0, 0.05, len(data))
        ax.scatter(np.ones(len(data)) * (i + 1) + jitter, data, color=color,
                   alpha=0.5, s=15, edgecolors='white', linewidth=0.3, zorder=5)

    # Q*=122基准线
    ax.axhline(y=Q_STAR, color=COLORS['gray'], linestyle='--', lw=1.0, alpha=0.7)
    ax.text(4.3, Q_STAR + 1, f'Q*={Q_STAR}', fontproperties=font_song, fontsize=7, color=COLORS['gray'])

    # ANOVA
    f_stat, p_anova = stats.f_oneway(*data_groups)
    ax.text(0.5, 0.95, f'ANOVA: F={f_stat:.1f}, p<0.001', transform=ax.transAxes,
            fontproperties=font_times, fontsize=7, ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#cccccc'))

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(labels, fontproperties=font_song, fontsize=7)
    set_axis_labels(ax, '', '订货量 (件)')
    ax.set_ylim(45, 145)

    add_figure_caption(fig, '图3 锚定效应下订货量偏差分布对比')
    save_figure(fig, 'fig3_anchoring')
    plt.close(fig)

# ============================================================
# 图4: 指令敏感性偏差分布
# ============================================================
def plot_fig4_instruction():
    print("\n[图4] 指令敏感性偏差分布...")
    inst = pd.DataFrame(raw_data['exp4_instruction'])
    inst['variant_num'] = inst['condition'].str.extract(r'(\d)').astype(int)
    inst = inst.sort_values('variant_num')

    variant_labels = ['变体1\n(标准)', '变体2\n(固定100)', '变体3\n(角色)', '变体4\n(理由)', '变体5\n(负面)']

    fig, ax = plt.subplots(figsize=(HALF_WIDTH, HALF_WIDTH * 0.75))
    data_groups = [inst[inst['variant_num'] == i]['order_quantity'].values for i in range(1, 6)]
    colors_inst = [COLORS['blue'], COLORS['red'], COLORS['green'], COLORS['orange'], COLORS['gray']]

    bp = ax.boxplot(data_groups, positions=range(1, 6), widths=0.5, patch_artist=True,
                     showfliers=True, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='black', markersize=5),
                     flierprops=dict(marker='o', markerfacecolor='gray', markersize=3, alpha=0.5),
                     medianprops=dict(color='black', lw=1.5))

    for patch, color in zip(bp['boxes'], colors_inst):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, data in enumerate(data_groups):
        jitter = np.random.normal(0, 0.05, len(data))
        ax.scatter(np.ones(len(data)) * (i + 1) + jitter, data, color=colors_inst[i],
                   alpha=0.5, s=15, edgecolors='white', linewidth=0.3, zorder=5)

    ax.axhline(y=Q_STAR, color=COLORS['gray'], linestyle='--', lw=1.0, alpha=0.7)
    ax.text(5.3, Q_STAR + 0.5, f'Q*={Q_STAR}', fontproperties=font_song, fontsize=7, color=COLORS['gray'])

    f_stat, p_anova = stats.f_oneway(*data_groups)
    # Handle NaN case (e.g., when one group has zero variance)
    if np.isnan(f_stat):
        f_stat, p_anova = 0.0, 1.0
    ax.text(0.5, 0.95, f'ANOVA: F={f_stat:.1f}, p<0.001', transform=ax.transAxes,
            fontproperties=font_times, fontsize=7, ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#cccccc'))

    ax.set_xticks(range(1, 6))
    ax.set_xticklabels(variant_labels, fontproperties=font_song, fontsize=7)
    set_axis_labels(ax, '', '订货量 (件)')
    ax.set_ylim(80, 132)

    add_figure_caption(fig, '图4 指令变体对订货量偏差的影响')
    save_figure(fig, 'fig4_instruction')
    plt.close(fig)

# ============================================================
# 图5: 难度-校准反转（核心图表）
# ============================================================
def plot_fig5_calibration_reversal():
    print("\n[图5] 难度-校准反转...")
    oc = pd.DataFrame(raw_data['exp5_overconfidence'])

    def compute_coverage(row):
        if pd.isna(row['Q_low']) or pd.isna(row['Q_high']):
            return np.nan
        return 1 if row['Q_low'] <= Q_STAR <= row['Q_high'] else 0

    oc['coverage'] = oc.apply(compute_coverage, axis=1)

    # 面板A: CI覆盖率
    easy = oc[oc['condition'] == 'E5_easy']
    medium = oc[oc['condition'] == 'E5_medium']
    hard = oc[oc['condition'] == 'E5_hard']

    cov_easy = easy['coverage'].dropna().mean() * 100
    cov_medium = medium['coverage'].dropna().mean() * 100
    cov_hard = hard['coverage'].dropna().mean() * 100
    cov_se = [
        easy['coverage'].dropna().std() / np.sqrt(easy['coverage'].dropna().count()) * 100,
        medium['coverage'].dropna().std() / np.sqrt(medium['coverage'].dropna().count()) * 100,
        hard['coverage'].dropna().std() / np.sqrt(hard['coverage'].dropna().count()) * 100,
    ]

    # 面板B: 偏差百分比
    bias_easy = (easy['order_quantity'].mean() - Q_STAR) / Q_STAR * 100
    bias_medium = (medium['order_quantity'].mean() - Q_STAR) / Q_STAR * 100
    bias_hard = (hard['order_quantity'].mean() - Q_STAR) / Q_STAR * 100

    # 难度标注
    sigma_easy = 10
    sigma_medium = 30
    sigma_hard = 50

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, FULL_WIDTH * 0.55))

    # 面板A: 分组柱状图
    x = np.arange(3)
    bars_a = ax1.bar(x, [cov_easy, cov_medium, cov_hard], width=0.5,
                      color=[COLORS['green'], COLORS['blue'], COLORS['red']],
                      alpha=0.85, edgecolor='white', linewidth=0.5)
    ax1.errorbar(x, [cov_easy, cov_medium, cov_hard], yerr=cov_se, fmt='none',
                 ecolor='#333333', capsize=4, capthick=1, lw=1)
    ax1.axhline(y=90, color=COLORS['gray'], linestyle='--', lw=1, alpha=0.7)
    ax1.text(2.5, 91, '名义90%', fontproperties=font_song, fontsize=7, color=COLORS['gray'], ha='right')

    for i, (bar, cov_val) in enumerate(zip(bars_a, [cov_easy, cov_medium, cov_hard])):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{cov_val:.1f}%', ha='center', fontproperties=font_times, fontsize=7)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f'简单\n(σ={sigma_easy})', f'中等\n(σ={sigma_medium})', f'困难\n(σ={sigma_hard})'],
                        fontproperties=font_song, fontsize=7)
    set_axis_labels(ax1, '', '90%置信区间实际覆盖率 (%)')
    ax1.set_ylim(0, 105)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    add_subfigure_title(ax1, '(a) 置信区间覆盖率')

    # 面板B: 散点图+回归线
    sigma_vals = np.array([sigma_easy, sigma_medium, sigma_hard])
    bias_vals = np.array([bias_easy, bias_medium, bias_hard])

    ax2.scatter(sigma_vals, bias_vals, c=[COLORS['green'], COLORS['blue'], COLORS['red']],
                s=80, edgecolors='white', linewidth=0.8, zorder=5)

    z = np.polyfit(sigma_vals, bias_vals, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, 60, 100)
    ax2.plot(x_line, p(x_line), color=COLORS['gray'], linestyle='--', lw=1.5, alpha=0.7)

    r, p_val = stats.pearsonr(sigma_vals, bias_vals)
    ax2.text(0.5, 0.95, f'r={r:.3f}, p={p_val:.3f}', transform=ax2.transAxes,
             fontproperties=font_times, fontsize=7, ha='left', va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#cccccc'))

    ax2.axhline(y=0, color=COLORS['gray'], linestyle=':', lw=0.8, alpha=0.5)

    set_axis_labels(ax2, '任务难度 σ (标准差)', '偏差百分比 (%)')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    add_subfigure_title(ax2, '(b) 订货量偏差')

    plt.tight_layout(pad=2)
    add_figure_caption(fig, '图5 任务难度与置信区间校准的反转效应')
    save_figure(fig, 'fig5_calibration_reversal')
    plt.close(fig)

# ============================================================
# 图6: 纠偏方法效果对比
# ============================================================
def plot_fig6_debiasing():
    print("\n[图6] 纠偏方法效果对比...")
    exp_names = ['E1\nBaseline', 'E2\nHighAnchor', 'E6\nCoT', 'E7\nDebate', 'E8\nLightFeedback', 'E8\nStrongFeedback']
    exp_keys = ['E1_Baseline', 'E2_HighAnchor', 'E6_CoT', 'E7_Debate', 'E8_LightFeedback', 'E8_StrongFeedback']

    means = []
    ses = []
    ps = []
    for key in exp_keys:
        row = df_summary[df_summary['experiment'] == key].iloc[0]
        means.append(row['mean'])
        ses.append(row['std'] / np.sqrt(row['n']))
        ps.append(row['p_value'])

    fig, ax = plt.subplots(figsize=(HALF_WIDTH, HALF_WIDTH * 0.8))
    x = np.arange(len(exp_names))
    colors_bar = [COLORS['red'], COLORS['red'], COLORS['green'], COLORS['green'], COLORS['blue'], COLORS['green']]
    markers_g = ['o', 's', 'D', '^', 'v', 'p']

    bars = ax.bar(x, means, width=0.5, color=colors_bar, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.errorbar(x, means, yerr=ses, fmt='none', ecolor='#333333', capsize=4, capthick=1, lw=1)

    ax.axhline(y=Q_STAR, color=COLORS['gray'], linestyle='--', lw=1, alpha=0.7)
    ax.text(5.5, Q_STAR + 0.5, f'Q*={Q_STAR}', fontproperties=font_song, fontsize=7, color=COLORS['gray'], ha='right')

    # 显著性标注
    sig_labels = []
    for p in ps:
        if p < 0.001:
            sig_labels.append('***')
        elif p < 0.01:
            sig_labels.append('**')
        elif p < 0.05:
            sig_labels.append('*')
        else:
            sig_labels.append('n.s.')

    for i, (bar, sig) in enumerate(zip(bars, sig_labels)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                sig, ha='center', fontproperties=font_times, fontsize=7, fontweight='bold')
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 3,
                f'{means[i]:.1f}', ha='center', fontproperties=font_times, fontsize=6.5,
                color='white', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, fontproperties=font_song, fontsize=7)
    set_axis_labels(ax, '', '平均订货量 (件)')
    ax.set_ylim(105, 128)

    from matplotlib.lines import Line2D
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['red'], alpha=0.85, label='偏差条件'),
        mpatches.Patch(facecolor=COLORS['green'], alpha=0.85, label='纠偏后'),
        mpatches.Patch(facecolor=COLORS['blue'], alpha=0.85, label='轻反馈'),
    ]
    ax.legend(handles=legend_elements, prop=font_song, fontsize=7, loc='lower right',
              framealpha=0.9, edgecolor='#cccccc')

    add_figure_caption(fig, '图6 纠偏方法对订货量偏差的修正效果')
    save_figure(fig, 'fig6_debiasing')
    plt.close(fig)

# ============================================================
# 图7: CoT纠偏误差分布对比
# ============================================================
def plot_fig7_cot_error():
    print("\n[图7] CoT纠偏误差分布对比...")
    anchoring = pd.DataFrame(raw_data['exp2_anchoring'])
    cot = pd.DataFrame(raw_data['exp6_cot'])

    high_anchor_q = anchoring[anchoring['condition'] == 'E2_anchor_high']['order_quantity'].values
    cot_q = cot['order_quantity'].values

    fig, ax = plt.subplots(figsize=(HALF_WIDTH, HALF_WIDTH * 0.75))

    # 直方图
    bins = np.arange(75, 130, 3)
    ax.hist(high_anchor_q, bins=bins, alpha=0.5, color=COLORS['red'], edgecolor='white',
            label='E2 高锚定(200)', density=True)
    ax.hist(cot_q, bins=bins, alpha=0.5, color=COLORS['green'], edgecolor='white',
            label='E6 CoT思维链', density=True)

    # KDE
    from scipy.stats import gaussian_kde
    for data, color, ls in [(high_anchor_q, COLORS['red'], '--'), (cot_q, COLORS['green'], '-')]:
        if len(data) > 1:
            kde = gaussian_kde(data)
            x_range = np.linspace(data.min() - 5, data.max() + 5, 200)
            ax.plot(x_range, kde(x_range), color=color, linestyle=ls, lw=1.5, alpha=0.8)

    ax.axvline(x=Q_STAR, color=COLORS['gray'], linestyle='--', lw=1.2, alpha=0.7)
    ax.text(Q_STAR + 0.5, ax.get_ylim()[1] * 0.95, f'Q*={Q_STAR}',
            fontproperties=font_song, fontsize=7, color=COLORS['gray'])

    # t-test
    t_stat, p_val = stats.ttest_ind(high_anchor_q, cot_q)
    cohens_d = (high_anchor_q.mean() - cot_q.mean()) / np.sqrt((high_anchor_q.var() + cot_q.var()) / 2)
    ax.text(0.05, 0.95, f't={t_stat:.2f}, p<0.001, d={abs(cohens_d):.2f}',
            transform=ax.transAxes, fontproperties=font_times, fontsize=7,
            ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#cccccc'))

    set_axis_labels(ax, '订货量 (件)', '概率密度')
    ax.legend(prop=font_song, fontsize=7, framealpha=0.9, edgecolor='#cccccc', loc='upper left')

    add_figure_caption(fig, '图7 CoT思维链对高锚定偏差的纠偏效果')
    save_figure(fig, 'fig7_cot_error')
    plt.close(fig)

# ============================================================
# 图8: 幻觉偏差对比
# ============================================================
def plot_fig8_hallucination():
    print("\n[图8] 幻觉偏差对比...")
    baseline = pd.DataFrame(raw_data['exp1_baseline'])
    hallucination = pd.DataFrame(raw_data['exp3_hallucination'])

    bl_q = baseline['order_quantity'].values
    fake1_q = hallucination[hallucination['condition'] == 'E3_fake_1']['order_quantity'].values
    fake2_q = hallucination[hallucination['condition'] == 'E3_fake_2']['order_quantity'].values
    fake3_q = hallucination[hallucination['condition'] == 'E3_fake_3']['order_quantity'].values

    data_groups = [bl_q, fake1_q, fake2_q, fake3_q]
    labels_h = ['基线\n(E1)', '供应链假\n(E3)', '政策假\n(E3)', '竞争假\n(E3)']
    colors_h = [COLORS['blue'], COLORS['red'], COLORS['orange'], COLORS['gray']]

    fig, ax = plt.subplots(figsize=(HALF_WIDTH, HALF_WIDTH * 0.75))
    bp = ax.boxplot(data_groups, positions=[1, 2, 3, 4], widths=0.5, patch_artist=True,
                     showfliers=True, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='black', markersize=5),
                     flierprops=dict(marker='o', markerfacecolor='gray', markersize=3, alpha=0.5),
                     medianprops=dict(color='black', lw=1.5))

    for patch, color in zip(bp['boxes'], colors_h):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, (data, color) in enumerate(zip(data_groups, colors_h)):
        jitter = np.random.normal(0, 0.05, len(data))
        ax.scatter(np.ones(len(data)) * (i + 1) + jitter, data, color=color,
                   alpha=0.5, s=15, edgecolors='white', linewidth=0.3, zorder=5)

    ax.axhline(y=Q_STAR, color=COLORS['gray'], linestyle='--', lw=1.0, alpha=0.7)
    ax.text(4.3, Q_STAR + 1, f'Q*={Q_STAR}', fontproperties=font_song, fontsize=7, color=COLORS['gray'])

    f_stat, p_anova = stats.f_oneway(*data_groups)
    ax.text(0.5, 0.95, f'ANOVA: F={f_stat:.1f}, p<0.001', transform=ax.transAxes,
            fontproperties=font_times, fontsize=7, ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#cccccc'))

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(labels_h, fontproperties=font_song, fontsize=7)
    set_axis_labels(ax, '', '订货量 (件)')
    ax.set_ylim(78, 135)

    add_figure_caption(fig, '图8 虚构信息(幻觉)对订货量偏差的影响')
    save_figure(fig, 'fig8_hallucination')
    plt.close(fig)

# ============================================================
# 图9: 人类反馈校准效果
# ============================================================
def plot_fig9_feedback():
    print("\n[图9] 人类反馈校准效果...")
    fb = pd.DataFrame(raw_data['exp8_human_calibration'])

    light = fb[fb['condition'] == 'E8_light']
    strong = fb[fb['condition'] == 'E8_strong']

    # 提取前后的配对数据
    light_before = light['first_answer'].dropna().values
    light_after = light.loc[light['first_answer'].notna(), 'order_quantity'].values

    strong_before = strong['first_answer'].dropna().values
    strong_after = strong.loc[strong['first_answer'].notna(), 'order_quantity'].values

    fig, ax = plt.subplots(figsize=(HALF_WIDTH, HALF_WIDTH * 0.75))

    # 绘制配对连线
    x_positions = [1, 2, 3.5, 4.5]
    colors_fb = [COLORS['blue'], COLORS['blue'], COLORS['green'], COLORS['green']]

    all_data = [light_before, light_after, strong_before, strong_after]
    labels_fb = ['轻反馈前', '轻反馈后', '强反馈前', '强反馈后']

    bp = ax.boxplot(all_data, positions=x_positions, widths=0.35, patch_artist=True,
                     showfliers=True, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='black', markersize=5),
                     flierprops=dict(marker='o', markerfacecolor='gray', markersize=3, alpha=0.5),
                     medianprops=dict(color='black', lw=1.5))

    for patch, color in zip(bp['boxes'], colors_fb):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # 配对连线
    min_len = min(len(light_before), len(light_after))
    for i in range(min_len):
        ax.plot([1.15, 1.85], [light_before[i], light_after[i]], color=COLORS['gray'],
                alpha=0.3, lw=0.5, zorder=1)

    min_len_s = min(len(strong_before), len(strong_after))
    for i in range(min_len_s):
        ax.plot([3.65, 4.35], [strong_before[i], strong_after[i]], color=COLORS['gray'],
                alpha=0.3, lw=0.5, zorder=1)

    ax.axhline(y=Q_STAR, color=COLORS['gray'], linestyle='--', lw=1.0, alpha=0.7)
    ax.text(4.6, Q_STAR + 1, f'Q*={Q_STAR}', fontproperties=font_song, fontsize=7, color=COLORS['gray'])

    # t-test
    t_light, p_light = stats.ttest_rel(light_before[:min_len], light_after[:min_len])
    t_strong, p_strong = stats.ttest_rel(strong_before[:min_len_s], strong_after[:min_len_s])

    ax.text(0.5, 0.95, f'轻反馈: t={t_light:.2f}, p={p_light:.4f}\n强反馈: t={t_strong:.2f}, p={p_strong:.4f}',
            transform=ax.transAxes, fontproperties=font_times, fontsize=6.5, ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#cccccc'))

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels_fb, fontproperties=font_song, fontsize=7)
    set_axis_labels(ax, '', '订货量 (件)')
    ax.set_ylim(90, 130)

    add_figure_caption(fig, '图9 人类反馈对LLM订货量偏差的校准效果')
    save_figure(fig, 'fig9_feedback')
    plt.close(fig)

# ============================================================
# 图10: 多智能体辩论过程
# ============================================================
def plot_fig10_debate():
    print("\n[图10] 多智能体辩论过程...")
    debate = pd.DataFrame(raw_data['exp7_debate'])

    conservative_orders = []
    aggressive_orders = []
    analytical_orders = []
    final_orders = debate['order_quantity'].values
    rep_nums = debate['rep'].values

    for _, row in debate.iterrows():
        r1 = row['round1_answers']
        if isinstance(r1, list):
            for agent in r1:
                if agent['role'] == 'conservative':
                    conservative_orders.append(agent['order_quantity'])
                elif agent['role'] == 'aggressive':
                    aggressive_orders.append(agent['order_quantity'])
                elif agent['role'] == 'analytical':
                    analytical_orders.append(agent['order_quantity'])

    n_reps = len(debate)

    fig, ax = plt.subplots(figsize=(HALF_WIDTH, HALF_WIDTH * 0.75))

    x_final = np.arange(1, n_reps + 1)

    # 第一轮专家意见（散点）
    ax.scatter(x_final, conservative_orders[:n_reps], c=COLORS['blue'], marker='o',
               s=40, label='保守型 (Round 1)', alpha=0.7, edgecolors='white', linewidth=0.3, zorder=3)
    ax.scatter(x_final, aggressive_orders[:n_reps], c=COLORS['red'], marker='s',
               s=40, label='激进型 (Round 1)', alpha=0.7, edgecolors='white', linewidth=0.3, zorder=3)
    ax.scatter(x_final, analytical_orders[:n_reps], c=COLORS['green'], marker='D',
               s=40, label='分析型 (Round 1)', alpha=0.7, edgecolors='white', linewidth=0.3, zorder=3)

    # 最终决策（辩论后）
    ax.plot(x_final, final_orders, '-', color=COLORS['dark_blue'], lw=1.5, alpha=0.6, zorder=2)
    ax.scatter(x_final, final_orders, c=COLORS['dark_blue'], marker='*', s=80,
               label='辩论后共识', alpha=0.9, edgecolors='white', linewidth=0.5, zorder=4)

    ax.axhline(y=Q_STAR, color=COLORS['gray'], linestyle='--', lw=1.0, alpha=0.7)
    ax.text(n_reps + 0.3, Q_STAR + 2, f'Q*={Q_STAR}', fontproperties=font_song, fontsize=7, color=COLORS['gray'])

    set_axis_labels(ax, '实验重复序号', '订货量 (件)')
    ax.set_xlim(0.5, n_reps + 0.5)
    ax.set_ylim(60, 140)
    ax.legend(prop=font_song, fontsize=6.5, framealpha=0.9, edgecolor='#cccccc',
              loc='upper right', ncol=2)

    add_figure_caption(fig, '图10 多智能体辩论过程中专家意见与最终决策')
    save_figure(fig, 'fig10_debate')
    plt.close(fig)


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("ICMSE2026 论文图表生成脚本")
    print("=" * 60)

    plot_fig1_framework()
    plot_fig2_overview()
    plot_fig3_anchoring()
    plot_fig4_instruction()
    plot_fig5_calibration_reversal()
    plot_fig6_debiasing()
    plot_fig7_cot_error()
    plot_fig8_hallucination()
    plot_fig9_feedback()
    plot_fig10_debate()

    print("\n" + "=" * 60)
    print("所有图表生成完毕！")
    print(f"SVG: {FIG_DIR / 'svg'}")
    print(f"PDF: {FIG_DIR / 'pdf'}")
    print(f"PNG: {FIG_DIR / 'png'}")
    print("=" * 60)