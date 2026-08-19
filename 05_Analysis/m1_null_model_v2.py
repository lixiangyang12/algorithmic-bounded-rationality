#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M1 零模型 V2：用观测点估计分布重采样中心
问题：中等条件 30% 覆盖率是否由点估计方差（部分试验点估计接近最优）驱动？

设计：
  Null1（原M1）：固定0.3σ有偏中心（offset=0.435σ）+ 宽度重采样
  Null2（V2）  ：中心从观测点估计分布重采样 + 宽度独立重采样
                 （保留点估计的均值偏差与方差，切断试验级 点估计-宽度 关联）
  Null3（方差剥离对照）：中心固定为条件均值（均值偏差保留、方差移除）+ 宽度重采样
  判断：若 Null2≈观测 且 Null3≈0 → 中等30%覆盖率由点估计方差驱动
        若 Null2≈观测 且 Null3≈观测 → 由均值偏差/宽度驱动，方差无贡献
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE = Path(r'e:\博士论文 研究计划以及小论文\ICMSE2026_AlgorithmicBoundedRationality')
RAW = BASE / '04_Data' / 'raw' / 'exp5_overconfidence.jsonl'
SIGMA = {'E5_easy': 10, 'E5_medium': 30, 'E5_hard': 60}
QSTAR = {'E5_easy': 107, 'E5_medium': 122, 'E5_hard': 144}
N, REPS, SEED = 30, 20000, 42

recs = [json.loads(l) for l in open(RAW, encoding='utf-8') if l.strip()]
groups = defaultdict(list)
for d in recs:
    groups[d['condition']].append(d)

rng = np.random.default_rng(SEED)


def null_fixed_center(widths, offset):
    out = np.empty(REPS)
    for r in range(REPS):
        ws = rng.choice(widths, size=N, replace=True)
        out[r] = np.mean(ws / 2 >= offset) * 100
    return out


def null_observed_center(qs, widths):
    """中心从观测点估计重采样（保留均值偏差+方差），宽度独立重采样"""
    out = np.empty(REPS)
    for r in range(REPS):
        q = rng.choice(qs, size=N, replace=True)
        ws = rng.choice(widths, size=N, replace=True)
        out[r] = np.mean(np.abs(q - QSTAR[None]) <= ws / 2) * 100  # 占位，实际按条件
    return out


print('=' * 88)
print('M1 零模型 V2：观测点估计分布重采样中心（20000次）')
print('=' * 88)
hdr = '{:<9}{:>8}{:>9}{:>9}{:>9}{:>9}{:>10}{:>9}'.format(
    '条件', '观测', 'Null1固定', 'Null2点估计', 'Null3均值中心', 'p2(零2≥观)', '偏移均值/σ', 'q-SD/σ')
print(hdr)
res = {}
for c in ['E5_easy', 'E5_medium', 'E5_hard']:
    g = groups[c]
    qs = np.array([d['order_quantity'] for d in g], dtype=float)
    low = np.array([d['Q_low'] for d in g], dtype=float)
    high = np.array([d['Q_high'] for d in g], dtype=float)
    widths = high - low
    qs_star = QSTAR[c]
    obs = np.mean((low <= qs_star) & (qs_star <= high)) * 100

    # Null1：固定0.435σ中心
    null1 = null_fixed_center(widths, 0.435 * SIGMA[c])
    # Null2：中心=观测点估计重采样（注意：重采样q后，需按该条件Q*计算覆盖）
    out2 = np.empty(REPS)
    for r in range(REPS):
        q = rng.choice(qs, size=N, replace=True)
        ws = rng.choice(widths, size=N, replace=True)
        out2[r] = np.mean(np.abs(q - qs_star) <= ws / 2) * 100
    # Null3：中心固定为条件均值（方差剥离）
    mean_q = qs.mean()
    out3 = np.empty(REPS)
    for r in range(REPS):
        ws = rng.choice(widths, size=N, replace=True)
        out3[r] = np.mean(np.abs(mean_q - qs_star) <= ws / 2) * 100

    p2 = np.mean(out2 >= obs)
    res[c] = dict(obs=obs, null2=out2, null3=out3,
                  off_mean=np.abs((high + low) / 2 - qs_star).mean(),
                  q_std=qs.std(ddof=1))
    print('{:<9}{:>8.1f}{:>9.1f}{:>9.1f}{:>9.1f}{:>10.3f}{:>10.2f}{:>9.2f}'.format(
        c, obs, null1.mean(), out2.mean(), out3.mean(), p2,
        res[c]['off_mean'] / SIGMA[c], res[c]['q_std'] / SIGMA[c]))

print()
print('=== 各条件 Null2/Null3 分布与观测对比 ===')
for c in ['E5_easy', 'E5_medium', 'E5_hard']:
    r = res[c]
    ci2 = (np.percentile(r['null2'], 2.5), np.percentile(r['null2'], 97.5))
    ci3 = (np.percentile(r['null3'], 2.5), np.percentile(r['null3'], 97.5))
    print('{:<9} 观测={:>5.1f}% | Null2点估计重采样: 均值={:>5.1f}% 95%CI=[{:>5.1f},{:>5.1f}] | Null3方差剥离: 均值={:>5.1f}% 95%CI=[{:>5.1f},{:>5.1f}]'.format(
        c, r['obs'], r['null2'].mean(), ci2[0], ci2[1], r['null3'].mean(), ci3[0], ci3[1]))

print()
print('=== 中等条件点估计分布诊断 ===')
g = groups['E5_medium']
qs = np.array([d['order_quantity'] for d in g], dtype=float)
print('  点估计: mean=%.1f, SD=%.2f, 范围[%d,%d]' % (qs.mean(), qs.std(ddof=1), qs.min(), qs.max()))
print('  距最优122较近(≥115)的试验: %d/30' % (np.sum(qs >= 115)))
print('  点估计≥115占比: %.1f%%' % (np.mean(qs >= 115) * 100))
print()
print('判定：中等 观测30.0% vs Null2(点估计方差) vs Null3(方差剥离)')
