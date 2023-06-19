from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier


def tune_hyperparameters(
    df, preprocessing_pipeline, classifier, classifier_name, param_grid
):
    assert classifier_name != "preprocessing"

    X = df.drop("Class", axis=1)
    y = df.Class

    param_grid = {f"{classifier_name}__{k}": v for k, v in param_grid.items()}

    fine_tuninng_pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessing_pipeline),
            (classifier_name, classifier),
        ]
    )
    grid_search = GridSearchCV(
        fine_tuninng_pipeline, param_grid, scoring="neg_log_loss"
    )
    grid_search.fit(X, y)

    print(f"Finished hyperparameter fine-tuning in {classifier_name}.")
    print(f"Best parameter (CV score={grid_search.best_score_}):")
    print(grid_search.best_params_)
    return grid_search


def xgb_model_for_optimization():
    xgb_params = {
        "booster": ["gbtree", "gblinear", "dart"],
        "max_depth": [2, 4, 6, 8],
        "lambda": [0, 0.001, 0.01, 0.1, 1],
        "alpha": [0, 0.001, 0.01, 0.1, 1],
    }
    xgb_clf = XGBClassifier(scale_pos_weight=4.71, random_state=seed)
    xgb_name = "XGBlassifier"
    xgb_arg = {"clf": xgb_clf, "clf_name": xgb_name, "param_grid": xgb_params}
    return xgb_arg


def lgbm_model_for_optimization():
    lgbm_params = {
        "boosting_type": ["gbdt", "dart"],
        "n_estimators": [20, 50, 100, 150, 200, 250, 300, 350, 400],
        "reg_alpha": [0, 0.001, 0.01, 0.1],
        "reg_lambda": [0, 0.001, 0.01, 0.1],
    }
    lgbm_clf = LGBMClassifier(class_weight="balanced")
    lgbm_name = "LGBMClassifier"
    lgbm_arg = {
        "clf": lgbm_clf,
        "clf_name": lgbm_name,
        "param_grid": lgbm_params,
    }
    return lgbm_arg


def models_for_hyperparameter_optimization():
    classifier_params = [
        lgbm_model_for_optimization(),
        xgb_model_for_optimization(),
    ]
    return classifier_params
