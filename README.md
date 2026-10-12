# 实验一：文本分类（新闻 10 分类）

## 环境

建议使用 Python ≥ 3.10，依赖：

```bash
pip install numpy pandas scikit-learn matplotlib
```

本机验证环境：`D:\ai-stack\venv`（scikit-learn 1.9.0）。

## 数据文件

| 文件 | 说明 |
|------|------|
| `train_data.csv` | 训练集，`text` + `target`（0–9） |
| `test_data_unlabeled.csv` | 无标签测试集，仅 `text` |
| `predictions.csv` | **提交用预测结果**（无表头，一行一个类别） |
| `draft_main.py` | 课程提供的草稿 |
| `TUNING.md` | 课程调参说明 |
| `main.py` | 本实验完整可复现脚本 |

## 一键复现

在 `data-Project1/data-Project1` 目录下执行：

```bash
python main.py
```

脚本会：

1. 分层划分验证集（80/20，`random_state=42`）
2. 对比 MultinomialNB / LinearSVM / LogisticRegression 多组超参
3. 对照「清洗邮件头」与「不清洗」
4. 按验证集 macro-F1 选最优配置，全量重训后写出 `predictions.csv`
5. 将表格与图写入 `results/`

## 当前最优配置（已跑通）

- 模型：`LinearSVC`，`C=1.0`
- 预处理：不清洗邮件头（清洗会损害验证表现，见报告）
- TF-IDF：`max_features=20000`, `ngram_range=(1,2)`, `stop_words='english'`, `sublinear_tf=True`, `min_df=2`
- 验证集：accuracy ≈ **0.9403**，macro-F1 ≈ **0.9404**

## 结果目录

- `results/val_results.csv`：全部验证对照
- `results/summary.json`：最优配置摘要
- `results/val_macro_f1_top.png`：Top 配置柱状图
- `results/best_val_confusion.png`：最优配置混淆矩阵
- `results/best_val_classification_report.txt`：分类报告

## 报告

见同目录 `report.md`（可复制到 Word/PDF，控制在 5 页内）。
