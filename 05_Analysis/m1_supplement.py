#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M1补充：中心异质性 + 反事实分解（宽度效应×中心效应）"""
import json
import numpy as np
from collections import defaultdict

recs = [json.loads(l) for l in open(
    r'e:\博士论文 研究计划以及小论文\ICMSE2026_AlgorithmicBoundedRationality\04_Data\raw\exp5_overconfidence.jsonl',
    encoding='utf-8') if l.strip()]
g = defaultdict(list)
for d in recs:
    g[d['condition']].append(d)

Q = {'E5_easy': 107, 'E5_medium': 122, 'E5_hard': 144}
S = {'E5_easy': 10, 'E5_medium': 30, 'E5_hard': 60}

print('=== 中心偏移分布（试验级异质性）===')
hdr = '{:<10}{:>8}{:>8}{:>9}{:>6}'.format('条件', '偏移均值', '偏移SD', '偏移中位', 'max')
print(hdr)
for c in ['E5_easy', 'E5_medium', 'E5_hard']:
    cs = np.array([(d['Q_high'] + d['Q_low']) / 2 for d in g[c]]) - Q[c]
    print('{:<10}{:>8.1f}{:>8.1f}{:>9.1f}{:>6.1f}'.format(
        c, np.abs(cs).mean(), np.abs(cs).std(), np.median(np.abs(cs)), np.abs(cs).max()))

print()
print('=== 反事实分解 ===')
print('C1 无偏中心(offset=0)+实测宽度: 覆盖率=100%（区间必含自身中心）→ 覆盖率不足的唯一来源是有偏中心')
print('C2 固定easy级宽度缩放(w=0.41σ)+实测中心: 无宽度缩放时覆盖率')
print('C3 实测宽度+实测中心（=观测，校验用）')
print('C4 实测宽度+固定0.435σ中心（M1零模型解析式）')
for c in ['E5_easy', 'E5_medium', 'E5_hard']:
    w = np.array([d['Q_high'] - d['Q_low'] for d in g[c]])
    cs = np.array([(d['Q_high'] + d['Q_low']) / 2 for d in g[c]]) - Q[c]
    wf = 0.41 * S[c]
    cov2 = np.mean(wf / 2 >= np.abs(cs)) * 100
    cov3 = np.mean(w / 2 >= np.abs(cs)) * 100
    cov4 = np.mean(w / 2 >= 0.435 * S[c]) * 100
    print('{:<10} C2: {:>5.1f}% | C3(观测): {:>5.1f}% | C4: {:>5.1f}%'.format(c, cov2, cov3, cov4))

print()
print('=== 覆盖阈值诊断 R=(w/2)/offset，R≥1 才可能覆盖 ===')
for c in ['E5_easy', 'E5_medium', 'E5_hard']:
    w = np.array([d['Q_high'] - d['Q_low'] for d in g[c]])
    cs = np.array([(d['Q_high'] + d['Q_low']) / 2 for d in g[c]]) - Q[c]
    r = (w / 2) / np.abs(cs)
    print('{:<10} R均值={:>5.2f}  R中位={:>5.2f}  达标率(R≥1)={:>5.1f}%'.format(
        c, r.mean(), np.median(r), np.mean(r >= 1) * 100))
