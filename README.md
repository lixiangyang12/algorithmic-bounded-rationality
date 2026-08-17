# 算法有限理性：大语言模型决策偏差与纠偏研究

**Algorithmic Bounded Rationality of LLM Agents: Identifying and Correcting Decision Biases in Operations Management**

本仓库提供论文《算法有限理性：大语言模型决策偏差与纠偏研究》的**实验代码与基本数据**，用于结果复现与后续研究引用。论文拟投稿第30届管理科学与工程国际会议（ICMSE 2026，分论坛3：智能决策与运营管理），正式发表后将在此处补充引用信息。

---

## 一、研究概述

当决策主体从人类变为大语言模型（LLM）Agent时，运营管理的经典偏差模式发生了结构性变化。本研究以**报童问题**为标准化实验平台，以 **DeepSeek V4 Flash** 为代表性LLM Agent，通过 **8个实验（535条API调用记录）** 系统考察了LLM Agent在运营决策中的偏差特征与纠偏机制：

1. **偏差识别（E1-E5）**：锚定效应、幻觉偏差、指令敏感性、过度自信四种偏差的强度排序；
2. **难度-校准反转（E5）**：LLM Agent的90%置信区间"自洽但不命中"——对真实最优解的覆盖率在简单/中等/困难任务中分别为6.7%/30.0%/33.3%，自覆盖率均为100%，校准误差随难度递减（83.3%→56.7%），与人类Hard-Easy Effect方向相反；
3. **纠偏机制比较（E6-E8）**：CoT推理（偏差改善+6.8pp）、多智能体辩论（+5.9pp）、人类反馈校准（强反馈+6.0pp、轻反馈+1.6pp）。

理论贡献：提出"算法有限理性"（Algorithmic Bounded Rationality）概念，将LLM Agent因算法架构、训练数据和推理机制本质特征而产生的系统性决策偏差理论化，并界定其与Simon有限理性的边界。

## 二、仓库结构

```
algorithmic_bounded_rationality/
├── config.py                  # 全局配置（API/实验/报童参数），密钥经环境变量读取
├── run_all.py                 # 8个实验的统一入口（E1-E8，支持 --dry-run）
├── requirements.txt           # Python依赖
├── .env.example               # 环境变量模板（填入API Key后另存为 .env）
├── 03_Experiments/            # 实验代码
│   ├── agents/                # LLM Agent封装、Prompt模板、响应解析
│   └── tasks/                 # 报童问题任务与偏差计算
├── 04_Data/
│   ├── raw/                   # 8个实验的原始API调用记录（JSONL，共535条）
│   └── results/               # 汇总统计与中间分析结果（CSV/JSON）
├── 05_Analysis/
│   └── statistical_tests.py   # 统计检验（t检验/ANOVA/Kruskal-Wallis/二项检验）
└── 06_Writing/figures_scripts/  # 论文图表绘制脚本
```

## 三、环境安装

```bash
pip install -r requirements.txt
```

配置API密钥（DeepSeek等）：将 `.env.example` 复制为 `.env` 并填入密钥，例如：

```
DEEPSEEK_API_KEY=sk-xxxx
```

密钥仅通过环境变量读取（见 `config.py`），仓库中不含任何真实密钥。

## 四、复现步骤

```bash
# 1. 干运行：查看配置与实验计划（不调用API）
python run_all.py --dry-run

# 2. 运行全部实验（E1-E8，约535次API调用，注意费用）
python run_all.py --exp all

# 或运行单个实验 / 指定重复次数
python run_all.py --exp E5 --n 10

# 3. 统计检验（单样本t检验、ANOVA、Kruskal-Wallis、覆盖率二项检验）
python 05_Analysis/statistical_tests.py --all

# 4. 论文图表（需Windows中文字体 SimHei/SimSun）
python 06_Writing/figures_scripts/plot_all_figures.py
python 06_Writing/figures_scripts/plot_fig6_calibration_v4.py
python 06_Writing/figures_scripts/plot_fig7_debiasing_v4.py
```

> 注意：E5（过度自信）各难度条件的理性最优解不同——简单σ=10对应Q\*=107、中等σ=30对应Q\*=122、困难σ=60对应Q\*=144。统计与绘图脚本均以各难度理性最优为参照（`statistical_tests.py` 中 `_optimal_q_for_sigma`）。

## 五、实验设计

| 实验 | 内容 | 条件数 | 重复 | 参照最优 |
|------|------|--------|------|----------|
| E1 | 基准测试 | 1 | 30 | 122 |
| E2 | 锚定效应（低/高/随机锚定） | 3 | 90 | 122 |
| E3 | 幻觉偏差（供应链/政策/竞争虚假信息） | 3 | 90 | 122 |
| E4 | 指令敏感性（正式/口语化/角色/警告/英文） | 5 | 150 | 122 |
| E5 | 过度自信与置信度校准（σ=10/30/60） | 3 | 90 | 107/122/144 |
| E6 | CoT推理纠偏 | 1 | 30 | 122 |
| E7 | 多智能体辩论纠偏 | 1 | 15 | 122 |
| E8 | 人类反馈校准（轻/强反馈） | 2 | 40 | 122 |

实验参数：报童问题 c=5、p=15、v=2、D~N(100,30)，温度 T=0.7；每次调用均为独立会话。

## 六、数据说明

- `04_Data/raw/*.jsonl`：原始API调用记录（含订货量、置信区间、推理文本、token与成本），共535条，成功率与解析成功率均为100%；
- `04_Data/results/all_experiments_summary.csv`：19个实验条件汇总（均值/SD/偏差/统计量），与论文表B.1、表C.1一致；
- `04_Data/results/full_analysis.json`：各实验详细统计结果。

## 七、关键结果（与论文一致）

- 低锚定导致最严重决策扭曲：偏差 **-42.2%**（d=-4.37），锚定效应呈显著不对称性；
- LLM Agent采用近似"均值+0.3σ"启发式（z≈0.32，理论最优z≈0.735），偏差随难度单调放大：**-3.5% / -8.8% / -17.4%**；
- 90%置信区间对真实最优解覆盖率：**6.7% / 30.0% / 33.3%**（自覆盖率均为100%），校准误差随难度递减（83.3%→56.7%）；
- CoT推理为最优纠偏策略（偏差从-6.7%恢复至+0.1%，6.8pp/调用）。

## 八、引用

论文正式发表后，请引用：

```bibtex
@unpublished{yang2026algorithmic,
  title  = {算法有限理性：大语言模型决策偏差与纠偏研究},
  author = {杨理想 and 王晶},
  note   = {第30届管理科学与工程国际会议（ICMSE 2026）投稿},
  year   = {2026}
}
```

## 九、许可

MIT License（见 LICENSE 文件）。实验数据遵循与代码相同的许可条款，引用时请注明出处。

---

## English Summary

This repository provides the experiment code and data for the study *"Algorithmic Bounded Rationality of LLM Agents: Identifying and Correcting Decision Biases in Operations Management"* (submitted to ICMSE 2026, Session 3: Intelligent Decision-Making and Operations Management). Using the newsvendor problem as a standardized testbed and DeepSeek V4 Flash as a representative LLM agent, the study runs 8 experiments (535 API call records) to identify four decision biases (anchoring, hallucination, instruction sensitivity, and overconfidence), reveals a "difficulty–calibration reversal" pattern (90% CI coverage of the true optimum: 6.7%/30.0%/33.3%; self-coverage 100%), and compares three debiasing mechanisms (CoT reasoning, multi-agent debate, and human feedback). See README (Chinese) for reproduction steps.
