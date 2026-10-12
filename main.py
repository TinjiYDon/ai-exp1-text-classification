# -*- coding: utf-8 -*-
"""
实验一：新闻文本 10 分类
流程：加载 → 清洗对照 → 分层验证划分 → TF-IDF → 多模型调参 → 全量重训 → 导出预测
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# ---------------------------------------------------------------------------
# 全局配置（固定种子，保证可复现）
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
VAL_SIZE = 0.2
DATA_DIR = Path(__file__).resolve().parent
RESULTS_DIR = DATA_DIR / "results"

HEADER_PREFIXES = (
    "from:",
    "subject:",
    "organization:",
    "lines:",
    "reply-to:",
    "nntp-posting-host:",
    "x-newsreader:",
    "x-mailer:",
    "distribution:",
    "keywords:",
    "summary:",
    "article-i.d.:",
    "news-software:",
)


def load_data():
    train_df = pd.read_csv(DATA_DIR / "train_data.csv")
    test_df = pd.read_csv(DATA_DIR / "test_data_unlabeled.csv")
    x_train = train_df["text"].astype(str).tolist()
    y_train = train_df["target"].values
    x_test = test_df["text"].astype(str).tolist()
    return x_train, y_train, x_test


def clean_newsgroup_text(text: str) -> str:
    """去掉新闻组邮件头与引用行，保留正文；作为创新点的轻量预处理。"""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    kept = []
    for line in lines:
        lower = line.strip().lower()
        if any(lower.startswith(p) for p in HEADER_PREFIXES):
            continue
        if lower.startswith(">"):
            continue
        if re.match(r"^in article\s+<", lower):
            continue
        kept.append(line)
    cleaned = "\n".join(kept)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if cleaned else text


def make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
        min_df=2,
    )


def candidate_models():
    """至少覆盖题目要求的 2+ 算法，并给出可解释的超参网格。"""
    configs = []
    for alpha in (0.1, 0.5, 1.0):
        configs.append(
            (
                "MultinomialNB",
                {"alpha": alpha},
                MultinomialNB(alpha=alpha),
            )
        )
    for c in (0.1, 1.0, 10.0):
        configs.append(
            (
                "LinearSVM",
                {"C": c},
                LinearSVC(C=c, random_state=RANDOM_STATE, max_iter=5000),
            )
        )
    for c in (0.1, 1.0, 10.0):
        configs.append(
            (
                "LogisticRegression",
                {"C": c},
                LogisticRegression(
                    C=c,
                    max_iter=2000,
                    solver="saga",
                    random_state=RANDOM_STATE,
                ),
            )
        )
    return configs


def evaluate_split(x_tr, y_tr, x_va, y_va, clean: bool) -> list[dict]:
    x_tr_use = [clean_newsgroup_text(t) for t in x_tr] if clean else list(x_tr)
    x_va_use = [clean_newsgroup_text(t) for t in x_va] if clean else list(x_va)

    vectorizer = make_vectorizer()
    x_tr_vec = vectorizer.fit_transform(x_tr_use)
    x_va_vec = vectorizer.transform(x_va_use)

    rows = []
    for name, params, model in candidate_models():
        t0 = time.time()
        model.fit(x_tr_vec, y_tr)
        pred = model.predict(x_va_vec)
        elapsed = time.time() - t0
        row = {
            "clean": clean,
            "model": name,
            "params": params,
            "accuracy": float(accuracy_score(y_va, pred)),
            "macro_f1": float(f1_score(y_va, pred, average="macro")),
            "train_seconds": round(elapsed, 3),
        }
        rows.append(row)
        print(
            f"[val] clean={clean} {name} {params} "
            f"acc={row['accuracy']:.4f} f1={row['macro_f1']:.4f} "
            f"({row['train_seconds']}s)"
        )
    return rows


def pick_best(rows: list[dict]) -> dict:
    return max(rows, key=lambda r: (r["macro_f1"], r["accuracy"]))


def build_final_estimator(best: dict):
    name = best["model"]
    params = best["params"]
    if name == "MultinomialNB":
        clf = MultinomialNB(alpha=params["alpha"])
    elif name == "LinearSVM":
        clf = LinearSVC(
            C=params["C"], random_state=RANDOM_STATE, max_iter=5000
        )
    elif name == "LogisticRegression":
        clf = LogisticRegression(
            C=params["C"],
            max_iter=2000,
            solver="saga",
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(f"未知模型: {name}")
    return Pipeline(
        [
            ("tfidf", make_vectorizer()),
            ("clf", clf),
        ]
    )


def plot_comparison(df: pd.DataFrame, out_path: Path) -> None:
    pivot = (
        df.sort_values("macro_f1", ascending=False)
        .head(12)
        .assign(
            label=lambda d: d.apply(
                lambda r: f"{r['model']}|{r['params']}|clean={r['clean']}",
                axis=1,
            )
        )
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(pivot["label"], pivot["macro_f1"], color="#4C78A8")
    ax.set_xlabel("Macro-F1 (validation)")
    ax.set_ylabel("Configuration")
    ax.set_title("Top validation configurations by macro-F1")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion(y_true, y_pred, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix (best config on validation)")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    np.random.seed(RANDOM_STATE)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    x_all, y_all, x_test = load_data()
    print("--- 数据加载成功 ---")
    print(f"训练集样本数量: {len(x_all)}")
    print(f"无标签测试集样本数量: {len(x_test)}")

    x_tr, x_va, y_tr, y_va = train_test_split(
        x_all,
        y_all,
        test_size=VAL_SIZE,
        stratify=y_all,
        random_state=RANDOM_STATE,
    )
    print(f"划分后: train={len(x_tr)}, val={len(x_va)} (stratified)")

    all_rows: list[dict] = []
    # 创新对照：清洗 vs 不清洗（其余流程一致）
    for clean in (False, True):
        all_rows.extend(evaluate_split(x_tr, y_tr, x_va, y_va, clean=clean))

    results_df = pd.DataFrame(all_rows)
    results_path = RESULTS_DIR / "val_results.csv"
    results_df.to_csv(results_path, index=False, encoding="utf-8")
    plot_comparison(results_df, RESULTS_DIR / "val_macro_f1_top.png")

    best = pick_best(all_rows)
    print("\n=== 验证集最优配置 ===")
    print(json.dumps(best, ensure_ascii=False, indent=2))

    # 用最优配置在验证划分上重算混淆矩阵与分类报告（仅报告用）
    x_tr_use = (
        [clean_newsgroup_text(t) for t in x_tr]
        if best["clean"]
        else list(x_tr)
    )
    x_va_use = (
        [clean_newsgroup_text(t) for t in x_va]
        if best["clean"]
        else list(x_va)
    )
    pipe = build_final_estimator(best)
    pipe.fit(x_tr_use, y_tr)
    va_pred = pipe.predict(x_va_use)
    report_txt = classification_report(y_va, va_pred, digits=4)
    (RESULTS_DIR / "best_val_classification_report.txt").write_text(
        report_txt, encoding="utf-8"
    )
    plot_confusion(y_va, va_pred, RESULTS_DIR / "best_val_confusion.png")

    # 全量训练集重训后预测无标签测试集
    x_all_use = (
        [clean_newsgroup_text(t) for t in x_all]
        if best["clean"]
        else list(x_all)
    )
    x_test_use = (
        [clean_newsgroup_text(t) for t in x_test]
        if best["clean"]
        else list(x_test)
    )
    final_pipe = build_final_estimator(best)
    final_pipe.fit(x_all_use, y_all)
    test_pred = final_pipe.predict(x_test_use)

    pred_path = DATA_DIR / "predictions.csv"
    pd.DataFrame(test_pred).to_csv(pred_path, index=False, header=False)
    print(f"已写入预测: {pred_path} ({len(test_pred)} 行)")

    summary = {
        "random_state": RANDOM_STATE,
        "val_size": VAL_SIZE,
        "vectorizer": {
            "max_features": 20000,
            "ngram_range": [1, 2],
            "stop_words": "english",
            "sublinear_tf": True,
            "min_df": 2,
        },
        "best": best,
        "n_train_full": len(x_all),
        "n_test": len(x_test),
        "predictions_path": str(pred_path.name),
    }
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"结果明细: {results_path}")
    print("完成。")


if __name__ == "__main__":
    main()
