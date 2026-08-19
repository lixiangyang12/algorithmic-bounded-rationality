#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M1 零模型蒙特卡洛检验：难度-校准反转（DCR）是否可由机械几何关系解释

质疑（M1）：校准误差随难度递减（83.3%→60.0%→56.7%）可能与区间宽度随σ扩张
（w/σ: 0.41→0.47→0.79）机械共线——若区间中心固定在"均值+0.3σ"的有偏点估计上，
覆盖率随难度上升只是 w/2 ≥ offset 的几何必然，而非结构机制。

零模型：固定有偏中心（center = Q* - 0.435σ，对应"均值+0.3σ"启发式，
        Q*位于μ+0.735σ），区间宽度从各条件实测宽度分布中重采样（保留实测宽度缩放），
        蒙特卡洛 20000 次，比较零模型覆盖率/CE与观测值。
对照：另用实测中心偏移（offset/σ=0.36/0.36/0.42）作灵敏度。

输出：几何诊断表 + 零模型分布 vs 观测 + 结论
"""
import json
import numpy as np
from pathlib import Path
from scipy import stats
from collections import defaultdict

BASE = Path(r'e:\博士论文 研究计划以及小论文\ICMSE2026_AlgorithmicBoundedRationality')
RAW = BASE / '04_Data' / 'raw' / 'exp5_overconfidence.jsonl'
SIGMA = {'E5_easy': 10, 'E5_medium': 30, 'E5_hard': 60}
QSTAR = {'E5_easy': 107, 'E5_medium': 122, 'E5_hard': 144}
N_TRIALS = 30
REPS = 20000
SEED = 42

recs = [json.loads(l) for l in open(RAW, encoding='utf-8') if l.strip()]
groups = defaultdict(list)
for d in recs:
    groups[d['condition']].append(d)


def null_coverage_dist(widths, offset, n_trials=N_TRIALS, reps=REPS, seed=SEED):
    """零模型：固定有偏中心（距真最优 offset），宽度从实测分布重采样"""
    rng = np.random.default_rng(seed)
    out = np.empty(reps)
    for r in range(reps):
        ws = rng.choice(widths, size=n_trials, replace=True)
        out[r] = np.mean(ws / 2 >= offset) * 100   # 覆盖 ⇔ w/2 ≥ offset
    return out


print('=' * 78)
print('M1 零模型蒙特卡洛检验：DCR 的机械共线诊断')
print('=' * 78)
print(f"\n{'条件':<10}{'σ':>4}{'Q*':>5}{'w̄':>7}{'w/σ':>7}{'偏移̄':>7}{'偏移/σ':>8}{'R=(w/2)/偏移':>13}")
summary = {}
for c in ['E5_easy', 'E5_medium', 'E5_hard']:
    g = groups[c]
    q = np.array([d['order_quantity'] for d in g])
    low = np.array([d['Q_low'] for d in g], dtype=float)
    high = np.array([d['Q_high'] for d in g], dtype=float)
    widths = high - low
    centers = (high + low) / 2
    qs = QSTAR[c]
    offset = np.abs(centers - qs)
    w_mean = widths.mean()
    off_mean = offset.mean()
    r = (w_mean / 2) / off_mean
    obs_cov = np.mean((low <= qs) & (qs <= high)) * 100
    obs_ce = abs(obs_cov - 90)
    print(f"{c:<10}{SIGMA[c]:>4}{qs:>5}{w_mean:>7.1f}{w_mean/SIGMA[c]:>7.2f}"
          f"{off_mean:>7.1f}{off_mean/SIGMA[c]:>8.2f}{r:>13.2f}")

    # 零模型1：固定 0.3σ 有偏中心（offset=0.435σ）
    off_null = 0.435 * SIGMA[c]
    null1 = null_coverage_dist(widths, off_null)
    # 零模型2：实测中心偏移
    null2 = null_coverage_dist(widths, off_mean)

    obs_cov_n = obs_cov
    p1 = np.mean(null1 >= obs_cov_n)   # P(零模型≥观测)
    p2 = np.mean(null2 >= obs_cov_n)
    summary[c] = dict(widths=widths, obs_cov=obs_cov_n, obs_ce=obs_ce,
                      off_null=off_null, off_meas=off_mean,
                      null1_mean=null1.mean(), null1_ci=(np.percentile(null1, 2.5), np.percentile(null1, 97.5)), p1=p1,
                      null2_mean=null2.mean(), null2_ci=(np.percentile(null2, 2.5), np.percentile(null2, 97.5)), p2=p2)

print('\n' + '-' * 78)
print('零模型结果（固定0.3σ有偏中心，offset=0.435σ；宽度=实测分布重采样，20000次）')
print(f"{'条件':<10}{'观测覆盖':>9}{'零模型均值':>10}{'零模型95%CI':>18}{'P(零≥观)':>10}{'观测CE':>8}{'零模型CE':>9}")
for c in ['E5_easy', 'E5_medium', 'E5_hard']:
    s = summary[c]
    null1_ce = abs(s['null1_mean'] - 90)
    print(f"{c:<10}{s['obs_cov']:>9.1f}{s['null1_mean']:>10.1f}"
          f"[{s['null1_ci'][0]:>5.1f},{s['null1_ci'][1]:>5.1f}]".replace('[', '[').replace(',', ','),
          end='')
    print(f"{s['p1']:>10.3f}{s['obs_ce']:>8.1f}{null1_ce:>9.1f}")

print('\n灵敏度：实测中心偏移（offset/σ=0.36/0.36/0.42）')
for c in ['E5_easy', 'E5_medium', 'E5_hard']:
    s = summary[c]
    print(f"{c:<10} 零模型均值={s['null2_mean']:.1f}%  95%CI=[{s['null2_ci'][0]:.1f},{s['null2_ci'][1]:.1f}]  P(零≥观)={s['p2']:.3f}")

# 零模型 CE 的难度梯度
ces = []
for c in ['E5_easy', 'E5_medium', 'E5_hard']:
    s = summary[c]
    # 重算一次用于CE分布
    null1 = null_coverage_dist(s['widths'], s['off_null'])
    ce_dist = np.abs(null1 - 90)
    ces.append((ce_dist.mean(), np.percentile(ce_dist, 2.5), np.percentile(ce_dist, 97.5)))
print('\n零模型校准误差CE随难度变化：', [f"{c}: {m:.1f} [CI {lo:.1f},{hi:.1f}]" for c, (m, lo, hi) in zip(['easy','medium','hard'], ces)])

# 结论判定
print('\n' + '=' * 78)
print('判定：')
print('1) 若 P(零模型≥观测覆盖) 接近0.5 或 观测落在零模型95%CI内 → 覆盖率的难度梯度可由几何共线解释（M1成立）')
print('2) 若 观测显著高于零模型（P很小）→ 存在超越几何的结构性覆盖改善')
print('=' * 78)
