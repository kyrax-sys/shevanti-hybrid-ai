import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_and_clean_data(path):
    df = pd.read_csv(path)
    df.columns = ['date', 'price']
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['target'] = df['price'].shift(-1)
    return df


def get_splits(df, train_end='2014-12-31', valid_end='2015-12-31'):
    train = df[df['date'] <= train_end]
    valid = df[(df['date'] > train_end) & (df['date'] <= valid_end)]
    test = df[df['date'] > valid_end]
    return train, valid, test


def evaluate(df_slice, pred_col, label, verbose=True):
    y_true = df_slice['target'].values
    y_pred = df_slice[pred_col].values
    today = df_slice['price'].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    smape = np.mean(
        2 * np.abs(y_pred - y_true)
        / (np.abs(y_true) + np.abs(y_pred) + 1e-9)
    ) * 100
    dir_acc = np.mean(
        np.sign(y_true - today) == np.sign(y_pred - today)
    ) * 100

    if verbose:
        print(
            f"{label:20s}  MAE={mae:7.2f}  RMSE={rmse:7.2f}  "
            f"sMAPE={smape:6.2f}%  DirAcc={dir_acc:5.1f}%"
        )

    return {
        'model': label,
        'MAE': mae,
        'RMSE': rmse,
        'sMAPE': smape,
        'DirAcc': dir_acc
    }
