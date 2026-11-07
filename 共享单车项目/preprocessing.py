"""共享单车：按时间划分和防止数据泄漏的预处理组件。"""
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / '数据集' / '共享单车' / 'hour.csv'
TARGET = 'cnt'
NUMERIC = ['temp', 'atemp', 'hum', 'windspeed']
CATEGORICAL = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
FEATURES = NUMERIC + CATEGORICAL

def load_and_split():
    df = pd.read_csv(SOURCE, parse_dates=['dteday'])
    df = df.sort_values(['dteday', 'hr'], kind='stable').reset_index(drop=True)
    # 整天划分：选择最接近 80% 记录数的日期边界。
    daily = df.groupby('dteday', sort=True).size().cumsum()
    boundary = (daily - len(df) * .8).abs().idxmin()
    train = df.loc[df.dteday <= boundary].copy()
    test = df.loc[df.dteday > boundary].copy()
    assert train.dteday.max() < test.dteday.min()
    return train, test

def make_preprocessor(scale_numeric=True):
    numeric_steps = [('imputer', SimpleImputer(strategy='median'))]
    if scale_numeric:
        numeric_steps.append(('scaler', StandardScaler()))
    categorical_steps = [('imputer', SimpleImputer(strategy='most_frequent')),
                         ('onehot', OneHotEncoder(handle_unknown='ignore'))]
    return ColumnTransformer([
        ('numeric', Pipeline(numeric_steps), NUMERIC),
        ('categorical', Pipeline(categorical_steps), CATEGORICAL),
    ])

def make_pipeline(model, scale_numeric=True):
    return Pipeline([('preprocess', make_preprocessor(scale_numeric)), ('model', model)])
