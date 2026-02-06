# =========================
# Install
# =========================
!pip install catboost


# =========================
# Upload files
#=========================
from google.colab import files
uploaded = files.upload()


# =========================
# Imports
# =========================
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


# =========================
# Load
# =========================
train = pd.read_csv("/train.csv")
test = pd.read_csv("/test.csv")
sample_sub = pd.read_csv("/sample_submission.csv")

X = train.drop(columns=["y"])
y = train["y"]


# =========================
# Feature Engineering
# =========================

# Example: interaction features if columns exist
if "age" in X.columns and "balance" in X.columns:
    X["age_balance"] = X["age"] * X["balance"]
    test["age_balance"] = test["age"] * test["balance"]

# count missing per row
X["missing_count"] = X.isna().sum(axis=1)
test["missing_count"] = test.isna().sum(axis=1)


# =========================
# Categorical columns
# =========================
cat_features = X.select_dtypes(include=["object"]).columns.tolist()
print("Categoricals:", cat_features)


# =========================
# Handle class imbalance
# =========================
pos_weight = (y == 0).sum() / (y == 1).sum()
print("Scale pos weight:", pos_weight)


# =========================
# CV Setup
# =========================
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof = np.zeros(len(X))
test_preds = np.zeros(len(test))

# =========================
# FAST TRAINING VERSION
# =========================

from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np

folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

oof = np.zeros(len(X))
test_preds = np.zeros(len(test))

for fold,(trn,val) in enumerate(folds.split(X,y)):
    print(f"\n🔥 Fold {fold+1}")

    X_tr,X_val = X.iloc[trn],X.iloc[val]
    y_tr,y_val = y.iloc[trn],y.iloc[val]

    model = CatBoostClassifier(
        iterations=800,          # ↓ from 3000
        depth=8,                # ↓ from 10
        learning_rate=0.05,     # ↑ faster learning
        loss_function="Logloss",
        eval_metric="AUC",
        subsample=0.8,
        random_seed=42,
        early_stopping_rounds=100,
        verbose=200
    )

    model.fit(
        X_tr,y_tr,
        cat_features=cat_features,
        eval_set=(X_val,y_val),
        use_best_model=True
    )

    oof[val] = model.predict_proba(X_val)[:,1]
    test_preds += model.predict_proba(test)[:,1] / folds.n_splits


auc = roc_auc_score(y,oof)
print("\n⚡ FAST CV ROC AUC:",auc)

submission = sample_sub.copy()
submission["y"] = test_preds
submission.to_csv("submission.csv",index=False)

from google.colab import files
files.download("submission.csv")