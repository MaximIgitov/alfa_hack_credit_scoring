import os
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
import xgboost as xgb
import optuna
import time
from sklearn.model_selection import cross_val_score

class FeatureFiltration:
    def __init__(self, data_train, data_test):
        self.data_train = data_train
        self.data_test = data_test

    def preprocess_data(self):
        start_time = time.time()  # Начало отслеживания времени

        # Проверка наличия целевой переменной и разделение данных
        if 'target' in self.data_train.columns and 'target' in self.data_test.columns:
            X_train = self.data_train.drop(columns=['target'])
            y_train = self.data_train['target']
            X_test = self.data_test.drop(columns=['target'])
            y_test = self.data_test['target']
        else:
            raise ValueError("Ошибка: отсутствует столбец 'target' в train или test данных.")

        # Определение категориальных признаков
        categorical_columns = [col for col in X_train.columns if X_train[col].nunique() < 30]

        # Кодирование категориальных признаков
        for col in categorical_columns:
            X_train[col] = X_train[col].astype('category').cat.codes
            X_test[col] = X_test[col].astype('category').cat.codes

        end_time = time.time()  # Конец отслеживания времени
        print(f"Preprocessing data took {end_time - start_time:.2f} seconds.")
        return X_train, y_train, X_test, y_test

    def filter_features(self, X_train, y_train, X_test, y_test):
        start_time = time.time()  # Начало отслеживания времени

        # Обучение модели с предварительными гиперпараметрами
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=500,
            gpu_id=0,
            verbosity=0,
            use_label_encoder=False
        )
        model.fit(X_train, y_train)

        # Вычисление Permutation Importance
        perm_importance = permutation_importance(model, X_test, y_test, scoring='roc_auc', n_repeats=10, random_state=42)

        # Сохранение названий колонок с отрицательным приростом
        negative_importance_columns = [col for col, importance in zip(X_test.columns, perm_importance.importances_mean) if importance < 0]

        # Удаление этих колонок из обучающего и тестового наборов
        X_train = X_train.drop(columns=negative_importance_columns)
        X_test = X_test.drop(columns=negative_importance_columns)

        # Сохранение обновленных данных
        data_train_filtered = pd.concat([X_train, y_train], axis=1)
        data_test_filtered = pd.concat([X_test, y_test], axis=1)

        end_time = time.time()  # Конец отслеживания времени
        print(f"Feature filtering took {end_time - start_time:.2f} seconds.")
        print(f"Удаленные признаки с отрицательным влиянием: {negative_importance_columns}")
        return data_train_filtered, data_test_filtered


class Model:
    def __init__(self, data_train, data_test, n_trials=500, timeout=1000000):
        self.data_train = data_train
        self.data_test = data_test
        self.n_trials = n_trials
        self.timeout = timeout

    def optimize_xgb(self, trial, X_train, y_train, X_test, y_test):
        start_time = time.time()  # Начало отслеживания времени

        params = {
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0)
        }

        model = xgb.XGBClassifier(
            **params,
            objective='binary:logistic',
            n_estimators=500,
            gpu_id=0,
            verbosity=0,
            use_label_encoder=False
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train)],
            eval_metric='auc',
            verbose=False
        )

        preds = model.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, preds)

        end_time = time.time()  # Конец отслеживания времени
        print(f"Optimization took {end_time - start_time:.2f} seconds.")
        return score

    def run_optuna(self, study_name, optimize_function, X_train, y_train, X_test, y_test):
        start_time = time.time()  # Начало отслеживания времени

        study = optuna.create_study(direction='maximize', study_name=study_name)
        study.optimize(lambda trial: optimize_function(trial, X_train, y_train, X_test, y_test), n_trials=self.n_trials, timeout=self.timeout)

        end_time = time.time()  # Конец отслеживания времени
        print(f"Optuna optimization took {end_time - start_time:.2f} seconds.")
        return study.best_params

    def train_model(self, X_train, y_train, X_test, y_test):
        start_time = time.time()  # Начало отслеживания времени

        best_params_xgb = self.run_optuna('XGBoost', self.optimize_xgb, X_train, y_train, X_test, y_test)

        # Обучение модели с лучшими гиперпараметрами
        xgb_model = xgb.XGBClassifier(
            **best_params_xgb,
            objective='binary:logistic',
            n_estimators=500,
            gpu_id=0,
            verbosity=0,
            use_label_encoder=False
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_train, y_train)], eval_metric='auc', verbose=False)

        # Предсказания модели
        preds_xgb = xgb_model.predict_proba(X_test)[:, 1]

        # Оценка модели
        roc_auc = roc_auc_score(y_test, preds_xgb)
        end_time = time.time()  # Конец отслеживания времени
        print(f"Model training and evaluation took {end_time - start_time:.2f} seconds.")
        print(f"ROC-AUC score: {roc_auc:.4f}")
        return roc_auc, preds_xgb


def fitting(path):
    start_time = time.time()  # Начало отслеживания времени

    # Получим список файлов по выданному пути к папке
    try:
        current_data = os.listdir(path)
    except Exception:
        return "Не папка"

    # В случае отсутствия ошибки - сохраним данные
    else:
        current_data = os.listdir(path)

    # Выделим тренировочный датасет
    train_data = [data for data in current_data if data.endswith('train.parquet')][0]
    # Выделим тестовый датасет
    test_data = [data for data in current_data if data.endswith('test.parquet')][0]
    # Откроем тренировочные данные
    train_data = pd.read_parquet(path + f'/{train_data}')
    # Откроем тестовые данные
    test_data = pd.read_parquet(path + f'/{test_data}')

    # Фильтрация признаков
    filter_class = FeatureFiltration(train_data, test_data)
    X_train, y_train, X_test, y_test = filter_class.preprocess_data()
    data_train_filtered, data_test_filtered = filter_class.filter_features(X_train, y_train, X_test, y_test)

    # Разделим данные на X_train, X_test и y_train, y_test
    X_train_filtered = data_train_filtered.drop(columns=["target"])
    y_train_filtered = data_train_filtered["target"]
    X_test_filtered = data_test_filtered.drop(columns=["target"])
    y_test_filtered = data_test_filtered["target"]

    # Обучение модели
    model_class = Model(data_train_filtered, data_test_filtered)
    roc_auc, predictions = model_class.train_model(X_train_filtered, y_train_filtered, X_test_filtered, y_test_filtered)

    # Сохраним предсказания
    prediction_df = pd.DataFrame(predictions, columns=['prediction'])
    prediction_df['id'] = data_test_filtered['id'].values
    prediction_df.to_csv(f"predictions/{path.split('/')[-1]}.csv", index=False)

    end_time = time.time()  # Конец отслеживания времени
    print(f"Model training and prediction creation took {end_time - start_time:.2f} seconds.")

    return prediction_df


def model():
    start_time = time.time()  # Начало отслеживания времени

    folders = ['folder1', 'folder2', 'folder3']  # Пример папок, для которых будем генерировать предсказания
    for fold in folders:
        # Запишем новый путь к данным
        data_path = f'data/{fold}'
        # Вызовем функцию, передав в нее путь к папке для обучения
        prediction = fitting(path=data_path)
        # Сохраним полученное предсказание
        if type(prediction) is not str:
            # Сохраняем предсказание
            prediction.to_csv(f"predictions/{fold}.csv", index=False)
            print("Предсказание создано!")
        else:
            print("Невозможно создать предсказание!")

    end_time = time.time()  # Конец отслеживания времени
    print(f"Total model execution time: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    model()
