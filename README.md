# Spaceship-Titanic-MLW
## 📝 Project Overview
This repository contains a comprehensive suite of machine learning solutions for the Kaggle [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) competition. The project demonstrates a progression of techniques ranging from basic data imputation to advanced feature engineering, neural network hyperparameter tuning, and a production-ready Object-Oriented Programming (OOP) pipeline.

This repository is designed to meet strict evaluation criteria: well-commented, readable implementation code, and clear instructions for environment setup, training, and inference.

## 📂 Repository Structure & File Descriptions

The repository contains five core files, each representing a different approach or component of the machine learning lifecycle:

1. **`model xiaoqingyang.ipynb` (Baseline Random Forest)**
   * **Description:** A streamlined, readable baseline model. It handles basic missing value imputation (median/mode), boolean transformations, and trains a `RandomForestClassifier`. Perfect for understanding the raw dataset and achieving a quick baseline score.

2. **`model tianxuanpu.ipynb` (Advanced Feature Engineering & Ensemble)**
   * **Description:** A highly optimized solution focusing on data mining. It splits complex strings (like `Cabin` into Deck/Num/Side), engineers group size and total spending features, and utilizes Stratified K-Fold cross-validation. The final predictions are made using a Soft Voting Ensemble of advanced gradient boosting frameworks (`XGBoost`, `LightGBM`, and `CatBoost`).

3. **`model huangjinghua.ipynb` (Neural Network & Hyperparameter Tuning)**
   * **Description:** Focuses on standardizing data (`StandardScaler`) for deep learning. It leverages `RandomizedSearchCV` to aggressively tune a Multi-Layer Perceptron (`MLPClassifier`), optimizing hidden layer sizes, activation functions, and alpha parameters for stable and robust predictions.

4. **`model hujiayang.py` (Production-Ready OOP Script)**
   * **Description:** A fully structured, modular Python script. It utilizes `sklearn.pipeline.Pipeline` and `ColumnTransformer` to create a leak-proof preprocessing workflow. It trains a Logistic Regression model, generates EDA plots (via `seaborn`/`matplotlib`), extracts feature importances, and logs performance metrics automatically.

5. **`model hujiayang2.joblib` (Serialized Pre-trained Pipeline)**
   * **Description:** The compiled, serialized model exported from the Python script. It contains the fully fitted custom transformers, scalers, encoders, and the trained Logistic Regression weights. Ready for immediate inference on unseen data without retraining.

---

## 💻 Environment & Dependencies

To ensure reproducibility, please run this code using **Python 3.10 or higher**. 

Install the required dependencies using `pip`:

```bash
pip install numpy pandas scikit-learn xgboost lightgbm catboost matplotlib seaborn joblib












