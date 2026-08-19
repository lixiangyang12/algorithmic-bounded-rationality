# -*- coding: utf-8 -*-
"""M3-1: '均值+0.3σ'启发式回归拟合检验"""
import json
import numpy as np
from scipy import stats
from collections import defaultdict

recs = [json.loads(l) for l in open(
    r'e:\博士论文 研究计划以及小论文\ICMSE2026_AlgorithmicBoundedRationality\04_Data\raw\exp5_overconfidence.jsonl',
    encoding='utf-8') if l.strip()]
g = defaultdict(list)
for d in recs:
    g[d['condition']].append(d)
S = {'E5_easy': 10, 'E5_medium': 30, 'E5_hard': 60}

# 逐试验 z 值（z = (q-100)/σ）
zs, qs, sigs = [], [], []
for c, s in S.items():
    for d in g[c]:
        q = d['order_quantity']
        zs.append((q - 100) / s)
        qs.append(q)
        sigs.append(s)
zs = np.array(zs); qs = np.array(qs); sigs = np.array(sigs)

print('=== 逐试验 z=(q-100)/σ (n=90) ===')
ci95 = stats.t.interval(0.95, 89, zs.mean(), zs.std(ddof=1) / np.sqrt(90))
print('z 均值=%.3f, SD=%.3f, 95%%CI=[%.3f, %.3f]' % (zs.mean(), zs.std(ddof=1), ci95[0], ci95[1]))
t_735, p_735 = stats.ttest_1samp(zs, 0.735)
t_03, p_03 = stats.ttest_1samp(zs, 0.3)
t_032, p_032 = stats.ttest_1samp(zs, 0.32)
print('H0: z=0.735（理论最优）: t=%.2f, p=%.2e → %s' % (t_735, p_735, '拒绝' if p_735 < 0.05 else '不拒绝'))
print('H0: z=0.3: t=%.2f, p=%.3f' % (t_03, p_03))
print('H0: z=0.32: t=%.2f, p=%.3f' % (t_032, p_032))

print()
print('=== 回归拟合 Q ~ intercept + slope·σ (n=90) ===')
slope, intercept, r, p_reg, se = stats.linregress(sigs, qs)
ci_s = stats.t.interval(0.95, 88, slope, se)
print('slope=%.3f (SE=%.3f, 95%%CI=[%.3f, %.3f]), intercept=%.2f, R²=%.3f, p=%.2e' % (
    slope, se, ci_s[0], ci_s[1], intercept, r ** 2, p_reg))
t_s, p_s = (slope - 0.735) / se, 2 * (1 - stats.t.cdf(abs((slope - 0.735) / se), 88))
print('H0: slope=0.735: t=%.2f, p=%.2e → 拒绝（斜率显著低于理论最优）' % (t_s, p_s))
t_s03, p_s03 = (slope - 0.3) / se, 2 * (1 - stats.t.cdf(abs((slope - 0.3) / se), 88))
print('H0: slope=0.3: t=%.2f, p=%.3f → %s' % (t_s03, p_s03, '拒绝' if p_s03 < 0.05 else '不拒绝'))
print('R²=%.3f 表示σ解释了订货量变异的%.1f%%' % (r ** 2, r ** 2 * 100))

print()
print('=== 各条件均值z ===')
for c, s in S.items():
    m = np.mean([d['order_quantity'] for d in g[c]])
    print('%s: mean_Q=%.1f, z=(mean-100)/σ=%.3f' % (c, m, (m - 100) / s))
