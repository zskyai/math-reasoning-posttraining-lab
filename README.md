# Math Reasoning Post-Training Lab

一个可复现的 verifier-based reasoning 后训练最小实验。仓库把
`Base -> SFT -> DPO -> GRPO` 的关键数据流拆开：答案解析、可验证奖励、偏好损失、组内优势归一化和
`pass@k`/格式指标都可以独立测试。

> 真实完成边界：当前 smoke 使用 12 条手工审计题和一个四分类策略（每个分类对应一个固定候选回答）。
> 它用于验证目标函数和评测代码，不是 Qwen/GSM8K/MATH 的训练结果，也不报告 GPU 提升。把结果写入简历前，
> 必须替换为带版本号的公开数据并完成真实模型、多随机种子和污染审计。

## 为什么做这个实验

数学推理后训练中，字符串奖励很容易被格式投机污染：模型可能学会输出 `\\boxed{}`，但答案仍然错误；
也可能因为一组 rollout 全对或全错而没有有效的相对优势。本仓库用同一个确定性 verifier 同时构造 DPO
偏好对、计算 GRPO reward 和统计评测，便于定位这些失败模式。

## 目录

```text
verifier.py                         # boxed/####/numeric answer extraction and reward
losses.py                           # SFT, DPO and clipped GRPO objectives
metrics.py                          # empirical pass@k, format and length metrics
data/fixture_math.jsonl             # 12 hand-audited offline examples
scripts/build_preference_pairs.py   # verifier-checked chosen/rejected pairs
scripts/run_smoke.py                # deterministic CPU Base/SFT/DPO/GRPO smoke
configs/smoke.json                  # seed and optimisation settings
tests/                              # parser, objective and metric unit tests
```

## Quick start

```bash
python -m pip install -r requirements.txt
python scripts/build_preference_pairs.py
python scripts/run_smoke.py --config configs/smoke.json
python -m unittest discover -s tests -v
```

`run_smoke.py` writes `outputs/smoke/summary.json` and `metrics.json` (ignored by Git). The summary records the
seed, reward components, final losses, entropy, action distribution, zero-variance group count and limitations.

## Smoke experiment

Each prompt has four fixed candidates:

1. correct answer with an explicit `\\boxed{answer}` marker;
2. wrong answer with a marker;
3. correct answer without an explicit marker;
4. an unparsable response.

The base categorical policy intentionally prefers candidate 2. SFT targets candidate 3 to expose the difference
between correctness and machine-readable formatting. DPO prefers candidate 1 over candidate 3, while GRPO uses
the verifier reward (`correctness + format bonus - invalid penalty`) and group-normalised advantages. This is a
controlled objective test, not evidence that DPO or GRPO is better on a language benchmark.

`metrics.py` reports empirical "any of the first k" pass@k over the fixed candidate list. It is not the unbiased
stochastic pass@k estimator used for sampled language-model evaluations.

## Moving to a real model

For a publishable algorithm experiment, keep the objective and verifier interfaces but replace the fixture with a
documented public split such as GSM8K, MATH-500, OpenR1-Math or NuminaMath. Record the dataset revision, token
counts and SHA-256; split by problem; and run at least:

```text
Base | SFT | SFT+DPO | SFT+GRPO (or Dr.GRPO)
```

Use an exact-answer verifier plus a format check, report pass@1/pass@8, cross-dataset generalisation, response
length, KL/entropy, reward and rollout throughput, and retain failed examples. A production implementation can
plug these interfaces into TRL, veRL or Open-R1, but the data construction, reward design, ablations and diagnosis
must remain explicit rather than presenting a default trainer run as an original result.
