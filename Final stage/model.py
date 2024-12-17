import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import optuna
class Model:
    def __init__(self, n_trials=50, timeout=600):
        self.n_trials = n_trials
        self.timeout = timeout

    def optimize_xgb(self, trial, X_train, X_valid, y_valid):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'objective': 'binary:logistic',
            'gpu_id': 0,
            'verbosity': 0,
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3)
        }

        model = xgb.train(
            params=params,
            dtrain=X_train,
            num_boost_round=500,
            evals=[(X_valid, 'valid')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        preds = model.predict(X_valid)
        return roc_auc_score(y_valid, preds)

    def train_and_predict(self, X_train, X_test, y_train, y_test):
        study = optuna.create_study(direction='maximize')
        X_train_split, X_valid, y_train_split, y_valid = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        X_train_split_D = xgb.DMatrix(X_train_split, label=y_train_split)
        X_valid_D = xgb.DMatrix(X_valid, label=y_valid)

        study.optimize(
            lambda trial: self.optimize_xgb(trial, X_train_split_D, X_valid_D, y_valid),
            n_trials=self.n_trials,
            timeout=self.timeout
        )
        best_params = study.best_params
        model = xgb.train(
            params={
                **best_params,
                'objective': 'binary:logistic',
                'gpu_id': 0,
                'verbosity': 0,
            },
            dtrain=xgb.DMatrix(X_train, label=y_train),
            num_boost_round=500
        )
        preds = model.predict(xgb.DMatrix(X_test))
        return preds