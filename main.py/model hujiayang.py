from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from spaceship_features import SPEND_COLUMNS, SpaceshipFeatureEngineer


RANDOM_STATE = 42


NUMERIC_FEATURES = [
    "Age",
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
    "PassengerGroup",
    "PassengerNumber",
    "GroupSize",
    "CabinNum",
    "TotalSpend",
    "LogTotalSpend",
    "LogRoomService",
    "LogFoodCourt",
    "LogShoppingMall",
    "LogSpa",
    "LogVRDeck",
    "NameLength",
]

CATEGORICAL_FEATURES = [
    "HomePlanet",
    "CryoSleep",
    "Destination",
    "VIP",
    "Deck",
    "Side",
    "IsSolo",
    "NoSpend",
    "IsChild",
    "HasName",
]


@dataclass
class TrainArtifacts:
    pipeline: Pipeline
    best_params: dict
    best_cv_score: float
    holdout_accuracy: float
    holdout_report: str
    confusion: list[list[int]]
    coefficients: pd.DataFrame
    cv_results: pd.DataFrame


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    model = LogisticRegression(
        solver="liblinear",
        max_iter=4000,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("features", SpaceshipFeatureEngineer()),
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Expected train.csv and test.csv in {data_dir}")
    return pd.read_csv(train_path), pd.read_csv(test_path)


def save_eda_plots(train: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="Set2")
    engineered = SpaceshipFeatureEngineer().fit_transform(train)

    plots: dict[str, str] = {}

    def save_current(name: str) -> None:
        path = figures_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=160, bbox_inches="tight")
        plt.close()
        plots[name] = str(path)

    plt.figure(figsize=(7, 4))
    sns.countplot(data=train, x="Transported")
    plt.title("Transported Distribution")
    plt.xlabel("Transported")
    plt.ylabel("Passenger Count")
    save_current("01_target_distribution.png")

    plt.figure(figsize=(7, 4))
    sns.barplot(data=train, x="HomePlanet", y="Transported", errorbar=None)
    plt.title("Transported Rate by Home Planet")
    plt.xlabel("Home Planet")
    plt.ylabel("Transported Rate")
    save_current("02_homeplanet_transport_rate.png")

    plt.figure(figsize=(7, 4))
    sns.barplot(data=train, x="CryoSleep", y="Transported", errorbar=None)
    plt.title("Transported Rate by CryoSleep")
    plt.xlabel("CryoSleep")
    plt.ylabel("Transported Rate")
    save_current("03_cryosleep_transport_rate.png")

    plt.figure(figsize=(8, 4))
    sns.histplot(data=train, x="Age", hue="Transported", bins=35, kde=True, element="step")
    plt.title("Age Distribution by Transported")
    plt.xlabel("Age")
    plt.ylabel("Passenger Count")
    save_current("04_age_distribution.png")

    plt.figure(figsize=(8, 4))
    sns.boxplot(data=engineered, x="Transported", y="LogTotalSpend")
    plt.title("Log Total Spend by Transported")
    plt.xlabel("Transported")
    plt.ylabel("log(1 + TotalSpend)")
    save_current("05_total_spend_boxplot.png")

    missing = train.isna().mean().sort_values(ascending=False).reset_index()
    missing.columns = ["Feature", "MissingRate"]
    plt.figure(figsize=(8, 5))
    sns.barplot(data=missing[missing["MissingRate"] > 0], x="MissingRate", y="Feature")
    plt.title("Missing Value Rate")
    plt.xlabel("Missing Rate")
    plt.ylabel("Feature")
    save_current("06_missing_values.png")

    plt.figure(figsize=(8, 4))
    sns.barplot(data=engineered, x="Deck", y="Transported", errorbar=None, order=sorted(engineered["Deck"].unique()))
    plt.title("Transported Rate by Cabin Deck")
    plt.xlabel("Deck")
    plt.ylabel("Transported Rate")
    save_current("07_deck_transport_rate.png")

    corr_cols = [
        "Transported",
        "Age",
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck",
        "GroupSize",
        "CabinNum",
        "TotalSpend",
        "LogTotalSpend",
    ]
    corr_df = engineered[corr_cols].copy()
    corr_df["Transported"] = corr_df["Transported"].astype(int)
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr_df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="vlag", center=0)
    plt.title("Numeric Feature Correlation")
    save_current("08_correlation_heatmap.png")

    return plots


def tune_and_train(
    train: pd.DataFrame,
    output_dir: Path,
    cv_folds: int,
    search_iter: int,
) -> TrainArtifacts:
    X = train.drop(columns=["Transported"])
    y = train["Transported"].astype(bool)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    param_distributions = {
        "model__C": np.logspace(-3, 3, 80),
        "model__penalty": ["l1", "l2"],
        "model__class_weight": [None, "balanced"],
        "model__intercept_scaling": [0.5, 1.0, 2.0, 5.0],
    }
    search = RandomizedSearchCV(
        estimator=build_pipeline(),
        param_distributions=param_distributions,
        n_iter=search_iter,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
        return_train_score=True,
    )
    search.fit(X_train, y_train)

    best_pipeline = search.best_estimator_
    valid_predictions = best_pipeline.predict(X_valid)
    holdout_accuracy = accuracy_score(y_valid, valid_predictions)
    holdout_report = classification_report(y_valid, valid_predictions, digits=4)
    confusion = confusion_matrix(y_valid, valid_predictions, labels=[False, True]).tolist()

    final_pipeline = search.best_estimator_
    final_pipeline.fit(X, y)

    cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    cv_results.to_csv(output_dir / "cv_results.csv", index=False)

    coefficients = extract_coefficients(final_pipeline)
    coefficients.to_csv(output_dir / "logistic_regression_coefficients.csv", index=False)
    save_coefficients_plot(coefficients, output_dir / "figures" / "09_top_coefficients.png")

    joblib.dump(final_pipeline, output_dir / "logistic_regression_pipeline.joblib")

    return TrainArtifacts(
        pipeline=final_pipeline,
        best_params=search.best_params_,
        best_cv_score=float(search.best_score_),
        holdout_accuracy=float(holdout_accuracy),
        holdout_report=holdout_report,
        confusion=confusion,
        coefficients=coefficients,
        cv_results=cv_results,
    )


def extract_coefficients(pipeline: Pipeline) -> pd.DataFrame:
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocess"]
    model: LogisticRegression = pipeline.named_steps["model"]

    numeric_names = list(NUMERIC_FEATURES)
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    categorical_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    names = numeric_names + categorical_names

    return (
        pd.DataFrame({"feature": names, "coefficient": model.coef_[0]})
        .assign(abs_coefficient=lambda df: df["coefficient"].abs())
        .sort_values("abs_coefficient", ascending=False)
        .reset_index(drop=True)
    )


def save_coefficients_plot(coefficients: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    top = coefficients.head(20).iloc[::-1].copy()
    top["direction"] = np.where(top["coefficient"] >= 0, "positive", "negative")
    plt.figure(figsize=(8, 7))
    sns.barplot(data=top, x="coefficient", y="feature", hue="direction", dodge=False, palette="Set2")
    plt.axvline(0, color="#333333", linewidth=1)
    plt.title("Top 20 Logistic Regression Coefficients")
    plt.xlabel("Coefficient for Transported=True")
    plt.ylabel("Feature")
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def kaggle_bool_label(values: np.ndarray | pd.Series) -> list[bool]:
    values_array = np.asarray(values).astype(bool)
    return values_array.tolist()


def positive_probabilities(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    model: LogisticRegression = pipeline.named_steps["model"]
    true_index = int(np.where(model.classes_ == True)[0][0])
    return pipeline.predict_proba(X)[:, true_index]


def create_prediction_outputs(
    pipeline: Pipeline,
    train: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    test_predictions = pipeline.predict(test)
    submission = pd.DataFrame(
        {
            "PassengerId": test["PassengerId"],
            "Transported": kaggle_bool_label(test_predictions),
        }
    )
    submission_path = output_dir / "submission.csv"
    submission_true_false_path = output_dir / "submission_true_false.csv"
    submission.to_csv(submission_path, index=False)
    submission.to_csv(submission_true_false_path, index=False)

    train_features = train.drop(columns=["Transported"])
    train_predictions = pipeline.predict(train_features)
    test_probabilities = positive_probabilities(pipeline, test)
    train_probabilities = positive_probabilities(pipeline, train_features)

    train_output = train_features.copy()
    train_output.insert(0, "Dataset", "train")
    train_output.insert(2, "ActualTransported", kaggle_bool_label(train["Transported"]))
    train_output.insert(3, "PredictedTransported", kaggle_bool_label(train_predictions))
    train_output.insert(4, "PredictedTransportedProbability", np.round(train_probabilities, 6))

    test_output = test.copy()
    test_output.insert(0, "Dataset", "test")
    test_output.insert(2, "ActualTransported", "")
    test_output.insert(3, "PredictedTransported", kaggle_bool_label(test_predictions))
    test_output.insert(4, "PredictedTransportedProbability", np.round(test_probabilities, 6))

    all_predictions = pd.concat([train_output, test_output], ignore_index=True, sort=False)
    all_predictions_path = output_dir / "all_predictions_true_false.csv"
    all_predictions.to_csv(all_predictions_path, index=False)
    return submission_path, submission_true_false_path, all_predictions_path


def summarize_dataset(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    engineered = SpaceshipFeatureEngineer().fit_transform(train)
    summary = {
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "target_rate": float(train["Transported"].mean()),
        "target_counts": train["Transported"].value_counts().sort_index().to_dict(),
        "missing_train": train.isna().sum().to_dict(),
        "missing_test": test.isna().sum().to_dict(),
        "transport_by_homeplanet": train.groupby("HomePlanet")["Transported"].mean().round(4).to_dict(),
        "transport_by_cryosleep": train.groupby("CryoSleep")["Transported"].mean().round(4).to_dict(),
        "transport_by_destination": train.groupby("Destination")["Transported"].mean().round(4).to_dict(),
        "transport_by_deck": engineered.groupby("Deck")["Transported"].mean().round(4).to_dict(),
        "age_quantiles": train["Age"].quantile([0, 0.25, 0.5, 0.75, 0.95, 0.99, 1]).round(4).to_dict(),
        "spend_quantiles": engineered["TotalSpend"].quantile([0, 0.25, 0.5, 0.75, 0.95, 0.99, 1]).round(4).to_dict(),
    }
    return summary


def format_missing_table(missing: pd.Series, row_count: int) -> pd.DataFrame:
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        return pd.DataFrame([{"字段": "无", "缺失数量": 0, "缺失率": "0.00%"}])
    table = missing.rename_axis("字段").reset_index(name="缺失数量")
    table["缺失率"] = table["缺失数量"].apply(lambda value: f"{value / row_count:.2%}")
    return table


def write_report(
    train: pd.DataFrame,
    test: pd.DataFrame,
    summary: dict,
    artifacts: TrainArtifacts,
    output_dir: Path,
    cv_folds: int,
    search_iter: int,
) -> Path:
    report_path = output_dir / "spaceship_logistic_regression_report.md"
    missing_train = format_missing_table(pd.Series(summary["missing_train"]), len(train))
    missing_test = format_missing_table(pd.Series(summary["missing_test"]), len(test))
    top_coefficients = artifacts.coefficients.head(12)
    top_coefficient_lines = "\n".join(
        f"| {row.feature} | {row.coefficient:.4f} | {row.abs_coefficient:.4f} |"
        for row in top_coefficients.itertuples(index=False)
    )
    best_params_json = json.dumps(artifacts.best_params, ensure_ascii=False, indent=2)
    confusion = artifacts.confusion

    report = f"""# Spaceship Titanic 逻辑回归建模报告

## 1. 数据概况

- 训练集规模：{summary["train_shape"][0]} 行，{summary["train_shape"][1]} 列。
- 测试集规模：{summary["test_shape"][0]} 行，{summary["test_shape"][1]} 列。
- 训练集 `Transported=True` 占比：{summary["target_rate"]:.2%}。
- 目标：预测测试集中乘客是否被传送，输出 `PassengerId,Transported`。

## 2. 缺失值与预处理

训练集缺失值：

{missing_train.to_markdown(index=False)}

测试集缺失值：

{missing_test.to_markdown(index=False)}

预处理与特征工程：

1. 数值变量使用中位数填补，并使用 `StandardScaler` 标准化；逻辑回归对变量尺度敏感，因此标准化是必要步骤。
2. 分类变量和布尔变量使用众数填补，并使用 One-Hot 编码。
3. 从 `PassengerId` 拆分出乘客组号、组内编号和组规模，并构造 `IsSolo`。
4. 从 `Cabin` 拆分出 `Deck`、`CabinNum`、`Side`。
5. 对消费字段构造 `TotalSpend`、`LogTotalSpend` 和各消费项的 log 特征，用于缓解强右偏和异常高消费值。
6. 构造 `NoSpend`、`IsChild`、`NameLength`、`HasName` 等辅助特征。

## 3. 探索性数据分析（EDA）

### 3.1 目标变量分布

![Target Distribution](figures/01_target_distribution.png)

目标变量接近平衡，`Transported=True` 占比为 {summary["target_rate"]:.2%}，因此 Accuracy 可作为主要评估指标。

### 3.2 出发星球与传送率

![HomePlanet Rate](figures/02_homeplanet_transport_rate.png)

按 `HomePlanet` 的传送率：

{pd.Series(summary["transport_by_homeplanet"]).to_markdown()}

不同出发星球的传送率差异明显，说明该分类变量对预测有价值。

### 3.3 冷冻睡眠与传送率

![CryoSleep Rate](figures/03_cryosleep_transport_rate.png)

按 `CryoSleep` 的传送率：

{pd.Series(summary["transport_by_cryosleep"]).to_markdown()}

处于冷冻睡眠的乘客更容易被传送，这是最强的解释变量之一。

### 3.4 年龄分布

![Age Distribution](figures/04_age_distribution.png)

年龄分布存在缺失但缺失比例不高，因此采用中位数填补。儿童变量 `IsChild` 用于补充非线性年龄影响。

### 3.5 消费分布与异常值

![Total Spend Boxplot](figures/05_total_spend_boxplot.png)

消费字段强右偏，存在高消费异常值。报告图中使用 `log(1 + TotalSpend)` 展示；模型同时保留原始消费与 log 消费，并通过标准化降低尺度影响。

总消费分位数：

{pd.Series(summary["spend_quantiles"]).to_markdown()}

### 3.6 舱位 Deck 与传送率

![Deck Rate](figures/07_deck_transport_rate.png)

按 `Deck` 的传送率：

{pd.Series(summary["transport_by_deck"]).to_markdown()}

舱位甲板与传送概率存在关联，拆分 `Cabin` 比直接丢弃该列更有信息量。

### 3.7 相关性分析

![Correlation Heatmap](figures/08_correlation_heatmap.png)

数值相关性显示消费类字段之间存在明显关联，`LogTotalSpend` 与目标变量呈负相关趋势；这与冷冻睡眠乘客通常消费为 0 的现象一致。

## 4. 建模与验证策略

- 模型：`LogisticRegression`。
- 验证方式：训练集先划分 20% 分层 holdout；调参阶段在其余 80% 上使用 {cv_folds} 折分层交叉验证。
- 调参方式：`RandomizedSearchCV`，共抽样评估 {search_iter} 组参数，搜索 `C`、`penalty`、`class_weight`、`intercept_scaling`。
- 使用分层交叉验证的原因：这是二分类任务，且不是时间序列数据，因此分层 K 折比时间序列划分更合适。
- 评估指标：Accuracy，与 Kaggle Spaceship Titanic 的分类目标一致。

最佳参数：

```json
{best_params_json}
```

交叉验证最佳平均 Accuracy：{artifacts.best_cv_score:.4f}

20% holdout Accuracy：{artifacts.holdout_accuracy:.4f}

Holdout 混淆矩阵（行是真实值，列是预测值）：

|  | 预测 false | 预测 true |
|---|---:|---:|
| 真实 false | {confusion[0][0]} | {confusion[0][1]} |
| 真实 true | {confusion[1][0]} | {confusion[1][1]} |

分类报告：

```text
{artifacts.holdout_report}
```

## 5. 逻辑回归系数解释

![Top Coefficients](figures/09_top_coefficients.png)

| 特征 | 系数 | 绝对值 |
|---|---:|---:|
{top_coefficient_lines}

系数为正表示更倾向预测 `Transported=true`，系数为负表示更倾向预测 `Transported=false`。由于数值变量已标准化、分类变量已 One-Hot，系数可用于方向性解释。

## 6. 输出文件

- `submission.csv`：测试集 4277 条 Kaggle 提交结果，`Transported` 使用 `True`/`False`。
- `submission_true_false.csv`：与 `submission.csv` 内容相同，便于明确识别布尔版本。
- `all_predictions_true_false.csv`：训练集 + 测试集全部 12970 条数据预测结果，标签使用 `True`/`False`。
- `logistic_regression_pipeline.joblib`：完整预处理 + 逻辑回归模型流水线。
- `cv_results.csv`：超参数搜索结果。
- `logistic_regression_coefficients.csv`：逻辑回归系数明细。
- `figures/`：EDA 与模型解释图表。

## 7. 复现命令

```bash
python spaceship_logistic_regression.py --data-dir spaceship_data --output-dir spaceship_outputs
```
"""
    report_path.write_text(report, encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a Spaceship Titanic logistic regression model and create true/false outputs."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("spaceship_data"), help="Directory with train/test CSV.")
    parser.add_argument("--output-dir", type=Path, default=Path("spaceship_outputs"), help="Artifact directory.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of stratified CV folds.")
    parser.add_argument("--search-iter", type=int, default=80, help="Random search iterations.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figures").mkdir(parents=True, exist_ok=True)

    train, test = load_data(args.data_dir)
    summary = summarize_dataset(train, test)
    save_eda_plots(train, args.output_dir)
    artifacts = tune_and_train(train, args.output_dir, args.cv_folds, args.search_iter)
    submission_path, submission_true_false_path, all_predictions_path = create_prediction_outputs(
        artifacts.pipeline,
        train,
        test,
        args.output_dir,
    )
    report_path = write_report(train, test, summary, artifacts, args.output_dir, args.cv_folds, args.search_iter)

    metrics = {
        "best_cv_accuracy": artifacts.best_cv_score,
        "holdout_accuracy": artifacts.holdout_accuracy,
        "best_params": artifacts.best_params,
        "submission_path": str(submission_path),
        "submission_true_false_path": str(submission_true_false_path),
        "all_predictions_path": str(all_predictions_path),
        "report_path": str(report_path),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
