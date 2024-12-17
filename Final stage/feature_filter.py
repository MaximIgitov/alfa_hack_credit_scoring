import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

class FeatureFiltration:
    def __init__(self, data_train, data_test, importance_threshold=0):
        self.data_train = data_train
        self.data_test = data_test
        self.importance_threshold = importance_threshold

    def preprocess_data(self):
        if 'target' not in self.data_train.columns:
            raise ValueError("Ошибка: отсутствует столбец 'target' в train данных.")

        X = self.data_train.drop(columns=['target'])
        y = self.data_train['target']
        X_test = self.data_test

        categorical_columns = [col for col in X.columns if X[col].nunique() < 100]
        for col in categorical_columns:
            X[col] = X[col].astype('category').cat.codes
            if col in X_test.columns:
                X_test[col] = X_test[col].astype('category').cat.codes

        return X, y, X_test

    def filter_features(self, X, y):
        dtrain = xgb.DMatrix(X, label=y)
        model = xgb.train(
            params={
                'objective': 'binary:logistic',
                'gpu_id': 0,
                'verbosity': 0,
                'use_label_encoder': False,
            },
            dtrain=dtrain,
            num_boost_round=500
        )
        feature_importances = model.get_score(importance_type='gain')
        feature_importance_df = pd.DataFrame({
            'feature': feature_importances.keys(),
            'importance': feature_importances.values()
        })
        significant_features = feature_importance_df[
            feature_importance_df['importance'] > self.importance_threshold
            ]['feature'].tolist()
        print(len(X.columns) - len(significant_features))
        return significant_features

    def filter_perm_score(self, X, y):
        X_train_split, X_valid, y_train_split, y_valid = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = xgb.XGBClassifier(
            objective='binary:logistic',
            use_label_encoder=False,
            eval_metric='logloss',
            tree_method='gpu_hist',
            verbosity=0,
            random_state=42
        )
        model.fit(X_train_split, y_train_split)

        result = permutation_importance(
            model,
            X_valid,
            y_valid,
            scoring='roc_auc',
            random_state=42
        )

        significant_features = [
            feature for feature, importance in zip(X.columns, result.importances_mean)
            if importance > 0
        ]
        print(len(X.columns) - len(significant_features))
        return significant_features