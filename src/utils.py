import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
from src.exception import customException
import sys
import os
import dill


class DataPreprocessor:
    def __init__(self):
        # store fitted mappings so test can use them
        self.freq_maps = {}
        self.part_of_day_dict = {
            'night': 1, 
            'morning': 2, 
            'afternoon': 3, 
            'evening': 4
        }

    def _feature_engineering(self, df, fit=False):
        """Feature engineering applied to both train/test"""

        # drop duplicates
        df = df.drop_duplicates()

        # datetime features
        df['click_time'] = pd.to_datetime(df['click_time'])
        df['click_time_year'] = df['click_time'].dt.year
        df['click_time_month'] = df['click_time'].dt.month
        df['click_time_day'] = df['click_time'].dt.day
        df['click_time_hour'] = df['click_time'].dt.hour
        df['click_time_minutes'] = df['click_time'].dt.minute
        df['click_time_seconds'] = df['click_time'].dt.second

        # remove unused col
        if 'attributed_time' in df.columns:
            df = df.drop('attributed_time', axis=1)

        # basic counts
        df['ip_click_count'] = df.groupby('ip')['channel'].transform('count')
        df['app_click_count'] = df.groupby('app')['channel'].transform('count')
        df['ip_app_unique'] = df.groupby('ip')['app'].transform('nunique')
        df['ip_device_unique'] = df.groupby('ip')['device'].transform('nunique')
        df['is_night'] = df['click_time_hour'].isin([0,1,2,3,4,5,6]).astype(int)

        # frequency encoding
        for col in ['ip', 'app', 'device', 'os', 'channel']:
            if fit:
                freq = df[col].value_counts(normalize=True)
                self.freq_maps[col] = freq
            df[col + '_freq'] = df[col].map(self.freq_maps[col]).fillna(0)

        # extra features
        df['day_of_week'] = df['click_time'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
        df['part_of_day'] = pd.cut(
            df['click_time_hour'],
            bins=[-1, 6, 12, 18, 24],
            labels=['night','morning','afternoon','evening']
        )

        # map categories → numbers safely
        df['part_of_day'] = (
            df['part_of_day']
            .astype(str)  # ensure string before mapping
            .map(self.part_of_day_dict)
            .fillna(0)
            .astype(int)
        )

        # sort for deltas
        df = df.sort_values(['ip','click_time'])
        df['time_since_last_click'] = (
            df.groupby('ip')['click_time']
            .diff().dt.total_seconds()
            .fillna(-1)
        )

        # interaction counts
        df['ip_app_count'] = df.groupby(['ip','app'])['channel'].transform('count')
        df['ip_channel_count'] = df.groupby(['ip','channel'])['app'].transform('count')
        df['app_channel_count'] = df.groupby(['app','channel'])['ip'].transform('count')

        # ratios
        df['ip_app_ratio'] = df['ip_app_unique'] / (df['ip_click_count'] + 1e-6)
        df['ip_device_ratio'] = df['ip_device_unique'] / (df['ip_click_count'] + 1e-6)
        df['ip_vs_app_clicks'] = df['ip_click_count'] / (df['app_click_count'] + 1e-6)

        # rolling clicks (per ip, last 60s)
        def rolling_count(g):
            g = g.set_index('click_time')
            return g.rolling('60s')['ip'].count().reset_index(drop=True)

        try:
            df['rolling_clicks_1min'] = (
                df.groupby('ip', group_keys=False)
                .apply(rolling_count)
                .reset_index(drop=True)
            )
        except Exception:
            df['rolling_clicks_1min'] = 0  # fallback if group is too small

        # drop raw timestamp
        df = df.drop('click_time', axis=1)

        return df

    def fit_transform(self, df: pd.DataFrame):
        """Fit mappings on train + transform"""
        return self._feature_engineering(df.copy(), fit=True)

    def transform(self, df: pd.DataFrame):
        """Use stored mappings on test data"""
        return self._feature_engineering(df.copy(), fit=False)

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise customException(e,sys)
    


def evaluate_models(x_train, x_test,y_train, y_test,models):
    try:

        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            model.fit(x_train, y_train)

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.kyes())[i]] = test_model_score


        return report
    except Exception as e:
        raise customException(e,sys)