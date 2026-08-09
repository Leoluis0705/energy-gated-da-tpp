# CGCNN目标检索轨迹的隐藏DFT可评价性审计

## 结论

本次纠正后的实验中，所有采集策略只使用原始CGCNN目标区间预测、命中概率、不确定性和原有多样性信息。DFT可评价性模型不参与候选选择，只在候选已经被原始策略选出后进行隐藏评分。

因此，Predicted-Target Greedy没有获得DFT可评价性标签或概率。此前按该概率排序的`DFT-evaluable Greedy`与`Joint-qualified Greedy`属于另一个实验问题，不进入本次主比较。

## 隐藏评价模型

- 模型：`shallow_gradient_boosting`。
- 历史DFT尝试：20个，其中严格可评价10个。
- 留一交叉验证ROC-AUC：0.820；平衡准确率：0.700。
- 历史训练候选使用OOF概率，其余候选使用冻结全模型概率。
- 这些结果是ML估计的DFT可评价性，不是新增VASP观测。

## 十个配对seed的正式叠加结果

| 方法 | 查询数 | 目标命中 | 预计可计算目标数 | ML硬标签可计算目标数 | 目标内平均可计算概率 | 目标内ML可计算率 |
|---|---|---|---|---|---|---|
| Energy-Gated DA-TPP | 80 | 26.00 | 16.63 | 20.00 | 0.64 | 0.77 |
| Energy-Gated DA-TPP | 160 | 54.00 | 33.87 | 41.00 | 0.63 | 0.76 |
| Energy-Gated DA-TPP | 240 | 67.00 | 39.93 | 46.00 | 0.60 | 0.69 |
| Energy-Gated DA-TPP | 320 | 78.00 | 46.51 | 53.00 | 0.60 | 0.68 |
| Predicted-Target Greedy | 80 | 19.00 | 10.99 | 12.00 | 0.58 | 0.63 |
| Predicted-Target Greedy | 160 | 54.00 | 31.60 | 35.00 | 0.59 | 0.65 |
| Predicted-Target Greedy | 240 | 69.00 | 40.96 | 47.00 | 0.59 | 0.68 |
| Predicted-Target Greedy | 320 | 78.00 | 46.51 | 53.00 | 0.60 | 0.68 |

## Gate相对Greedy的配对差值

| 查询数 | 指标 | Gate减Greedy | Gate更高seed数 | 相同seed数 | Greedy更高seed数 |
|---|---|---|---|---|---|
| 80 | expected_DFT_evaluable_target_hits | 5.64 | 10 | 0 | 0 |
| 80 | ML_labeled_DFT_evaluable_target_hits | 8.00 | 10 | 0 | 0 |
| 80 | mean_hidden_DFT_evaluable_probability_among_targets | 0.06 | 10 | 0 | 0 |
| 80 | ML_labeled_DFT_evaluable_rate_among_targets | 0.14 | 10 | 0 | 0 |
| 160 | expected_DFT_evaluable_target_hits | 2.28 | 10 | 0 | 0 |
| 160 | ML_labeled_DFT_evaluable_target_hits | 6.00 | 10 | 0 | 0 |
| 160 | mean_hidden_DFT_evaluable_probability_among_targets | 0.04 | 10 | 0 | 0 |
| 160 | ML_labeled_DFT_evaluable_rate_among_targets | 0.11 | 10 | 0 | 0 |
| 240 | expected_DFT_evaluable_target_hits | -1.03 | 0 | 0 | 10 |
| 240 | ML_labeled_DFT_evaluable_target_hits | -1.00 | 0 | 0 | 10 |
| 240 | mean_hidden_DFT_evaluable_probability_among_targets | 0.00 | 10 | 0 | 0 |
| 240 | ML_labeled_DFT_evaluable_rate_among_targets | 0.01 | 10 | 0 | 0 |
| 320 | expected_DFT_evaluable_target_hits | 0.00 | 0 | 10 | 0 |
| 320 | ML_labeled_DFT_evaluable_target_hits | 0.00 | 0 | 10 | 0 |
| 320 | mean_hidden_DFT_evaluable_probability_among_targets | 0.00 | 0 | 10 | 0 |
| 320 | ML_labeled_DFT_evaluable_rate_among_targets | 0.00 | 0 | 10 | 0 |

正差表示Gate在相同查询预算下更早找到更多“目标区间命中且被隐藏模型评为DFT可评价”的候选。该优势集中在早期预算；到240次查询时Greedy追平或略微反超，到320次两者检完同一批78个目标后完全相同。

## seed独立性审计

| 方法 | 查询数 | seed数 | 不同查询前缀数 | 不同目标集合数 |
|---|---|---|---|---|
| Energy-Gated DA-TPP | 80 | 10 | 1 | 1 |
| Energy-Gated DA-TPP | 160 | 10 | 1 | 1 |
| Energy-Gated DA-TPP | 240 | 10 | 1 | 1 |
| Energy-Gated DA-TPP | 320 | 10 | 10 | 1 |
| Predicted-Target Greedy | 80 | 10 | 1 | 1 |
| Predicted-Target Greedy | 160 | 10 | 1 | 1 |
| Predicted-Target Greedy | 240 | 10 | 1 | 1 |
| Predicted-Target Greedy | 320 | 10 | 10 | 1 |

审计显示，80、160和240次查询时，每种方法的10个归档seed连完整查询前缀都相同；到320次时非目标候选顺序开始变化，但目标集合仍完全相同。因此早期差值只有一个独立轨迹，不能写成10次独立重复，也不能据此计算有意义的配对显著性。

## 原六基线单次历史轨迹

这些是旧的单次完整六策略轨迹，仅作描述性补充，不能替代十seedGate–Greedy正式比较。

| 方法 | 查询数 | 目标命中 | 预计可计算目标数 | ML硬标签可计算目标数 | 目标内平均可计算概率 | 目标内ML可计算率 |
|---|---|---|---|---|---|---|
| Energy-Gated DA-TPP | 80 | 33 | 20.20 | 22 | 0.61 | 0.67 |
| Energy-Gated DA-TPP | 160 | 50 | 29.78 | 33 | 0.60 | 0.66 |
| Energy-Gated DA-TPP | 240 | 69 | 40.66 | 46 | 0.59 | 0.67 |
| Energy-Gated DA-TPP | 320 | 78 | 46.51 | 53 | 0.60 | 0.68 |
| Explore | 80 | 29 | 17.32 | 17 | 0.60 | 0.59 |
| Explore | 160 | 44 | 26.05 | 27 | 0.59 | 0.61 |
| Explore | 240 | 60 | 35.44 | 39 | 0.59 | 0.65 |
| Explore | 320 | 74 | 43.60 | 49 | 0.59 | 0.66 |
| MC Dropout | 80 | 8 | 4.44 | 4 | 0.55 | 0.50 |
| MC Dropout | 160 | 27 | 14.42 | 14 | 0.53 | 0.52 |
| MC Dropout | 240 | 36 | 20.43 | 21 | 0.57 | 0.58 |
| MC Dropout | 320 | 49 | 28.56 | 32 | 0.58 | 0.65 |
| Modulus / Gradient-Norm Hybrid | 80 | 10 | 5.54 | 5 | 0.55 | 0.50 |
| Modulus / Gradient-Norm Hybrid | 160 | 20 | 9.22 | 7 | 0.46 | 0.35 |
| Modulus / Gradient-Norm Hybrid | 240 | 39 | 18.83 | 18 | 0.48 | 0.46 |
| Modulus / Gradient-Norm Hybrid | 320 | 58 | 32.40 | 36 | 0.56 | 0.62 |
| Predicted-Target Greedy | 80 | 27 | 16.36 | 20 | 0.61 | 0.74 |
| Predicted-Target Greedy | 160 | 47 | 28.78 | 34 | 0.61 | 0.72 |
| Predicted-Target Greedy | 240 | 69 | 40.96 | 47 | 0.59 | 0.68 |
| Predicted-Target Greedy | 320 | 78 | 46.51 | 53 | 0.60 | 0.68 |
| Random Sampling | 80 | 6 | 4.29 | 6 | 0.72 | 1.00 |
| Random Sampling | 160 | 18 | 11.30 | 13 | 0.63 | 0.72 |
| Random Sampling | 240 | 26 | 15.57 | 19 | 0.60 | 0.73 |
| Random Sampling | 320 | 39 | 22.73 | 28 | 0.58 | 0.72 |

## 论文使用边界

- 可以报告：在旧CGCNN目标区间任务下，Gate早期找到的目标候选中，隐藏ML评价器预测为DFT可评价的数量更多。
- 不可以报告：这些候选已经获得了新的真实DFT验证。
- 不可以把DFT可评价性概率输入任何采集策略后，仍称为同一实验。
- 新的VASP前瞻验证若以后完成，应单独作为真实高保真证据。
