class TuningSettings:
    VOTING_GRID_PARAMS = {
        "voting": ["soft"],
        "weights": [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2), (5, 1, 1),
                    (1, 5, 1), (1, 1, 5)],
    }

    LGBM_GRID_PARAMS = {
        "boosting_type": ["gbdt", "dart"],
        "n_estimators": [20, 50, 100, 150, 200, 250, 300, 350, 400],
        "reg_alpha": [0, 0.001, 0.01, 0.1],
        "reg_lambda": [0, 0.001, 0.01, 0.1],
    }

    XGB_GRID_PARAMS = {
        "booster": ["gbtree", "gblinear", "dart"],
        "max_depth": [2, 4, 6, 8],
        "lambda": [0, 0.001, 0.01, 0.1, 1],
        "alpha": [0, 0.001, 0.01, 0.1, 1],
    }

    CATBOOST_GRID_PARAMS = {
        'bagging_temperature': [1.0],  # nonnegative float # Default value: 1.0
        'boosting_type': ['Plain'],  # 'Ordered' or 'Plain'
        'depth': [5, 6],  # integer, range: (1, 16) # Default value: 6
        'grow_policy': [
            'Depthwise'
        ],  # 'SymmetricTree' (default), 'Depthwise' or 'Lossguide'
        'iterations': [200, 1000],  # Positive integer # Default value: 1000
        'l2_leaf_reg': [2, 3],  # Positive integer # Default value: 3
        'learning_rate': [
            0.03,
            0.07,
        ],  # float, range: (0.0, 1.0) # Default value: 0.03
        'max_bin': [65535],  # integers 1 to 65535 # 500
        'max_leaves': [31],  # integer # Default value: 31
        'min_data_in_leaf': [1, 3],  # integer # Default value: 1
        #     'random_seed'        : [0],              # Nonnegative integer # Default value: 1
        'random_strength': [1.0, 5.0],  # positive float # Default value: 1.0
        'rsm': [0.5, 1.0],  # float, range: (0.0, 1.0] # Default value: 1.0
        'sampling_frequency': [
            'PerTreeLevel'
        ],  # 'PerTreeLevel' (default) or 'PerTree'
        'thread_count': [-1],  # -1 or positive integer # Default value: -1
    }

    SEED = 42


class ModelGreekSettings:
    LGBM_PARAMS = {
        "boosting_type": "dart",
        "n_estimators": "300",
        "reg_alpha": 0,
        "reg_lambda": 0.1,
    }

    XGB_PARAMS = {
        "alpha": 1,
        "booster": "gbtree",
        "lambda": 1,
        "max_depth": 2,
    }

    CATBOOST_PARAMS = {
        'bagging_temperature': 1.0,
        'boosting_type': 'Plain',
        'depth': 5,
        'grow_policy': 'Depthwise',
        'iterations': 200,
        'l2_leaf_reg': 3,
        'learning_rate': 0.07,
        'max_bin': 65535,
        'max_leaves': 31,
        'min_data_in_leaf': 1,
        'random_strength': 5.0,
        'rsm': 1.0,
        'sampling_frequency': 'PerTreeLevel',
        'thread_count': -1,
    }
    SEED = 42


class ModelNoGreekSettings:
    LGBM_PARAMS = {
        "boosting_type": "dart",
        "n_estimators": "300",
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
    }

    XGB_PARAMS = {
        "alpha": 0.001,
        "booster": "gbtree",
        "lambda": 1,
        "max_depth": 2,
    }

    CATBOOST_PARAMS = {
        'bagging_temperature': 1.0,
        'boosting_type': 'Plain',
        'depth': 6,
        'grow_policy': 'Depthwise',
        'iterations': 200,
        'l2_leaf_reg': 3,
        'learning_rate': 0.07,
        'max_bin': 65535,
        'max_leaves': 31,
        'min_data_in_leaf': 3,
        'random_strength': 5.0,
        'rsm': 1.0,
        'sampling_frequency': 'PerTreeLevel',
        'thread_count': -1,
    }

    SEED = 42
