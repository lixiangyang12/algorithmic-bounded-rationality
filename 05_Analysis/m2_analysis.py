#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M2 前置分析：
 (1) 区间宽度-难度关系推断检验（w/σ 0.41→0.79）
 (2) 三种纠偏机制统一基线（E2高锚定 -6.7%）重算
"""
import json
import numpy as np
from scipy import stats
from collections import defaultdict

BASE = r'e:\博士论文 研究计划以及小论文\ICMSE2026_AlgorithmicBoundedRationality\04_Data\raw'
Q = 122

def load(fn):
    return [json.loads(l) for l in open(BASE + '\\' + fn, encoding='utf-8') if l.strip()]

print('=' * 74)
print('(1) 区间宽度-难度关系推断检验')
print('=' * 74)
recs = [json.loads(l) for l in open(BASE + '\\exp5_overconfidence.jsonl', encoding='utf-8') if l.strip()]
g = defaultdict(list)
for d in recs:
    g[d['condition']].append(d)
S = {'E5_easy': 10, 'E5_medium': 30, 'E5_hard': 60}
ws, ws_s, sigs = [], [], []
for c in ['E5_easy', 'E5_medium', 'E5_hard']:
    w = np.array([d['Q_high'] - d['Q_low'] for d in g[c]])
    ws.append(w)
    ws_s.append(w / S[c])
    sigs.extend([S[c]] * len(w))
w_all = np.concatenate(ws)
w_s_all = np.concatenate(ws_s)
sig_all = np.array(sigs)

print('w̄(±SD): 简单 %.1f±%.1f, 中等 %.1f±%.1f, 困难 %.1f±%.1f'
      % (ws[0].mean(), ws[0].std(ddof=1), ws[1].mean(), ws[1].std(ddof=1),
         ws[2].mean(), ws[2].std(ddof=1)))
print('w/σ(±SD): 简单 %.2f±%.2f, 中等 %.2f±%.2f, 困难 %.2f±%.2f'
      % (ws_s[0].mean(), ws_s[0].std(ddof=1), ws_s[1].mean(), ws_s[1].std(ddof=1),
         ws_s[2].mean(), ws_s[2].std(ddof=1)))
h, p_kw = stats.kruskal(*ws)
print('Kruskal-Wallis(宽度×难度): H(2)=%.2f, p=%.2e' % (h, p_kw))
rho, p_sp = stats.spearmanr(sig_all, w_all)
print('Spearman(σ, 宽度) 逐试验 n=90: ρ=%.3f, p=%.2e' % (rho, p_sp))
slope, intercept, r, p_reg, se = stats.linregress(sig_all, w_all)
print('线性回归 宽度~σ: 斜率=%.3f 件/σ, R²=%.3f, p=%.2e' % (slope, r ** 2, p_reg))
h2, p_kw2 = stats.kruskal(*ws_s)
print('Kruskal-Wallis(w/σ×难度): H(2)=%.2f, p=%.2e' % (h2, p_kw2))
rho2, p_sp2 = stats.spearmanr(sig_all, w_s_all)
print('Spearman(σ, w/σ) 逐试验: ρ=%.3f, p=%.2e' % (rho2, p_sp2))

print()
print('=' * 74)
print('(2) 三种纠偏机制统一基线（E2高锚定 -6.7%）')
print('=' * 74)
e2 = load('exp2_anchoring.jsonl')
e6 = load('exp6_cot.jsonl')
e7 = load('exp7_debate.jsonl')
e8 = load('exp8_human_calibration.jsonl')
base_q = np.array([d['order_quantity'] for d in e2 if d['condition'] == 'E2_anchor_high'])
base_bias = (base_q.mean() - Q) / Q * 100
print('统一基线 E2高锚定: mean=%.1f, 偏差=%.1f%%' % (base_q.mean(), base_bias))

mechs = [
    ('CoT推理(E6)', e6, None, 1),
    ('多智能体辩论(E7)', e7, None, 3),
    ('人类反馈强(E8)', [d for d in e8 if d['condition'] == 'E8_strong'], None, 2),
    ('人类反馈轻(E8)', [d for d in e8 if d['condition'] == 'E8_light'], None, 2),
]
print('%-18s %-12s %-10s %-8s %-10s %-8s' % ('机制', '基线偏差', '纠偏后', '改善pp', '调用数', '效率pp/调用'))
for name, data, _, calls in mechs:
    qs = np.array([d['order_quantity'] for d in data])
    post_bias = (qs.mean() - Q) / Q * 100
    improve = abs(base_bias) - abs(post_bias)
    eff = improve / calls
    print('%-18s %-12.1f %-10.1f %-8.1f %-10d %-8.2f' % (name, base_bias, post_bias, improve, calls, eff))

print()
print('=== 配对/独立检验（统一基线 E2高锚 113.8±8.1, n=30）===')
for name, data, _, _ in mechs:
    qs = np.array([d['order_quantity'] for d in data])
    t, p = stats.ttest_ind(base_q, qs)
    d_eff = (qs.mean() - base_q.mean()) / np.sqrt((base_q.var(ddof=1) + qs.var(ddof=1)) / 2)
    print('E2高锚 vs %-16s t=%.2f, p=%.4f, d=%.2f' % (name, t, p, d_eff))
