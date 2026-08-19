# 算法有限理性：大语言模型决策偏差与纠偏研究

**Algorithmic Bounded Rationality of LLM Agents: Identifying and Correcting Decision Biases in Operations Management**

本仓库提供论文《算法有限理性：大语言模型决策偏差与纠偏研究》的**实验代码与基本数据**，用于结果复现与后续研究引用。论文拟投稿第30届管理科学与工程国际会议（ICMSE 2026，分论坛3：智能决策与运营管理），正式发表后将在此处补充引用信息。

---

## 一、研究概述

当决策主体从人类变为大语言模型（LLM）Agent时，运营管理的经典偏差模式发生了**方向性变化**。本研究以**报童问题**为标准化实验平台，以 **DeepSeek V4 Flash**（模型ID：deepseek-chat，API端点 https://api.deepseek.com，温度 T=0.7，2026-08-10 API 快照）为代表性LLM Agent，通过 **8个实验（535条API调用记录）** 系统考察了LLM Agent在运营决策中的偏差特征与纠偏机制：

1. **偏差识别（E1-E5）**：锚定效应、虚假信息误导、指令敏感性、过度自信四种偏差的强度排序；
2. **难度-校准反转（E5）**：LLM Agent的90%置信区间"自洽但不命中"——对真实最优解的覆盖率在简单/中等/困难任务中分别为6.7%/30.0%/33.3%，自覆盖率均为100%；双零模型蒙特卡洛检验表明该覆盖率的难度梯度完全可由几何机制（点估计方差+宽度缩放）解释；真正与人类方向相反的反转内核是**区间宽度随难度扩张**（w/σ 0.41→0.79，人类困难任务收缩区间）；
3. **纠偏机制比较（E6-E8，统一基线E2高锚定-6.7%）**：CoT推理（+6.7pp）、脚本化外部反馈-强（+5.3pp）、多智能体辩论（+5.2pp）、脚本化外部反馈-轻（+0.8pp）。

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
│   ├── statistical_tests.py   # 统计检验（t检验/ANOVA/Kruskal-Wallis/二项检验）
│   ├── m1_null_model_dcr.py   # 难度-校准反转零模型蒙特卡洛检验（固定有偏中心）
│   ├── m1_null_model_v2.py    # 零模型V2（观测点估计分布重采样中心）
│   ├── m2_analysis.py         # 宽度-难度推断检验 + 纠偏统一基线
│   └── m3_heuristic_test.py   # "均值+0.3σ"启发式回归拟合检验
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

# 4. 稳健性检验（零模型/推断检验）
python 05_Analysis/m1_null_model_dcr.py
python 05_Analysis/m1_null_model_v2.py
python 05_Analysis/m2_analysis.py
python 05_Analysis/m3_heuristic_test.py

# 5. 论文图表（需Windows中文字体 SimHei/SimSun）
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
| E3 | 虚假信息误导（供应链/政策/竞争虚假信息） | 3 | 90 | 122 |
| E4 | 指令敏感性（正式/口语化/角色/警告/英文） | 5 | 150 | 122 |
| E5 | 过度自信与置信度校准（σ=10/30/60） | 3 | 90 | 107/122/144 |
| E6 | CoT推理纠偏 | 1 | 30 | 122 |
| E7 | 多智能体辩论纠偏 | 1 | 15 | 122 |
| E8 | 脚本化外部反馈（轻/强反馈，脚本生成非真实人类评审） | 2 | 40 | 122 |

实验参数：报童问题 c=5、p=15、v=2、D~N(100,30)，温度 T=0.7；每次调用均为独立会话。

## 六、数据说明

- `04_Data/raw/*.jsonl`：原始API调用记录（含订货量、置信区间、推理文本、token与成本），共535条，成功率与解析成功率均为100%；
- `04_Data/results/all_experiments_summary.csv`：19个实验条件汇总（均值/SD/偏差/统计量），与论文表B.1、表C.1一致；
- `04_Data/results/full_analysis.json`：各实验详细统计结果。

## 七、关键结果（与论文一致）

- 低锚定导致最严重决策扭曲：偏差 **-42.2%**（d=-4.37），锚定效应呈显著不对称性；
- LLM Agent采用近似"均值+0.3σ"启发式：回归拟合 Q̂=100.87+0.310σ（n=90，斜率95%CI[0.236,0.383]，显著低于理论最优0.735、与0.3无显著差异），偏差随难度单调放大：**-3.5% / -8.8% / -17.4%**；
- 90%置信区间对真实最优解覆盖率：**6.7% / 30.0% / 33.3%**（自覆盖率均为100%）；双零模型（固定有偏中心 / 观测点估计分布重采样）表明覆盖率难度梯度完全由几何机制解释（零模型Ⅱ p=0.907/0.286/0.844），校准误差随难度递减属几何投影而非结构性机制；
- 宽度-难度关系推断检验：Kruskal-Wallis H(2)=77.70（p<0.001），Spearman ρ=0.934（p<0.001），w/σ 0.41→0.79（与人类困难任务收缩区间方向相反，属跨任务对照假设）；
- 纠偏机制（统一基线E2高锚定-6.7%）：CoT推理 **+6.7pp**（6.7pp/调用）> 脚本化外部反馈-强 **+5.3pp**（2.6pp/调用）> 多智能体辩论 **+5.2pp**（1.7pp/调用）> 脚本化外部反馈-轻 **+0.8pp**（0.4pp/调用）。

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

This repository provides the experiment code and data for the study *"Algorithmic Bounded Rationality of LLM Agents: Identifying and Correcting Decision Biases in Operations Management"* (submitted to ICMSE 2026, Session 3: Intelligent Decision-Making and Operations Management). Using the newsvendor problem as a standardized testbed and DeepSeek V4 Flash (deepseek-chat, temperature 0.7, API snapshot 2026-08-10) as a representative LLM agent, the study runs 8 experiments (535 API call records) to identify four decision biases (anchoring, misinformation susceptibility, instruction sensitivity, and overconfidence), reveals a "difficulty–calibration reversal" pattern (90% CI coverage of the true optimum: 6.7%/30.0%/33.3%; self-coverage 100%; the coverage gradient is fully mechanical per dual zero-model tests; the robust reversal lies in width–difficulty scaling, w/σ 0.41→0.79, as a cross-task comparison hypothesis), and compares three debiasing mechanisms under a unified baseline (CoT: +6.7pp; scripted external feedback-strong: +5.3pp; multi-agent debate: +5.2pp; scripted external feedback-light: +0.8pp). See README (Chinese) for reproduction steps.
