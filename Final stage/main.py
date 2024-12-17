import os
import time
import pandas as pd
from model import Model
from feature_filter import FeatureFiltration


OPTUNA_EARLY_STOPING = 25

def process_directory(data_folder, predictions_folder):
    os.makedirs(predictions_folder, exist_ok=True)
    folders = os.listdir(data_folder)
    for folder in folders:
        try:
            folder_path = os.path.join(data_folder, folder)
            files = os.listdir(folder_path)
            train_file = next(file for file in files if file.endswith("train.parquet"))
            test_file = next(file for file in files if file.endswith("test.parquet"))

            train_data = pd.read_parquet(os.path.join(folder_path, train_file))
            test_data = pd.read_parquet(os.path.join(folder_path, test_file))
        except Exception as e:
            print(f"Ошибка при обработке папки {folder}: {e}")
            continue

        start_time = time.time()
        filtration = FeatureFiltration(train_data, test_data)
        X_train, y_train, X_test = filtration.preprocess_data()
        significant_features = filtration.filter_features(X_train, y_train)
        X_train_filtered = X_train[significant_features]
        X_test_filtered = X_test[significant_features]
        preprocess_time = time.time() - start_time
        print(f"Предобработка данных для папки {folder} заняла {preprocess_time:.2f} секунд.")

        start_time = time.time()
        if preprocess_time < 30:
            significant_features = filtration.filter_perm_score(X_train_filtered, y_train)
            X_train_filtered = X_train_filtered[significant_features]
            X_test_filtered = X_test_filtered[significant_features]
        perm_time = time.time() - start_time
        print(f"Расчет перм скора для {folder} заняло {perm_time:.2f} секунд.")

        start_time = time.time()
        model = Model(n_trials=500, timeout=1200)
        predictions = model.train_and_predict(X_train_filtered, X_test_filtered, y_train, None)
        train_time = time.time() - start_time
        print(f"Обучение модели для папки {folder} заняло {train_time:.2f} секунд.")

        prediction = pd.DataFrame({
            'id': test_data['id'],
            'target': predictions
        })
        prediction = prediction[['id', 'target']].sort_values(by='id', ascending=True)
        output_path = os.path.join(predictions_folder, f"{folder}.csv")
        prediction.to_csv(output_path, index=False)
        print(f"Предсказание для папки {folder} сохранено в {output_path}")


if __name__ == "__main__":
    data_folder = "./data"
    predictions_folder = "./predictions"
    process_directory(data_folder, predictions_folder)