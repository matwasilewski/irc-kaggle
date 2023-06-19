LGBM_PARAMS = {
    "boosting_type": ["gbdt", "dart"],
    "n_estimators": [20, 50, 100, 150, 200, 250, 300, 350, 400],
    "reg_alpha": [0, 0.001, 0.01, 0.1],
    "reg_lambda": [0, 0.001, 0.01, 0.1],
}

XGB_PARAMS = {
    "booster": ["gbtree", "gblinear", "dart"],
    "max_depth": [2, 4, 6, 8],
    "lambda": [0, 0.001, 0.01, 0.1, 1],
    "alpha": [0, 0.001, 0.01, 0.1, 1],
}

SEED = 42
