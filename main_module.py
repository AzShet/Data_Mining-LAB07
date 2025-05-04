
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder

def clean_data(df):
    """
    Clean the dataset by handling missing values, converting types, 
    and adjusting the target column.

    input:
    - df: Raw dataframe

    output:
    - df: Cleaned dataframe
    """
    df = df.replace('?', np.nan)
    df = df.dropna()
    df['bare_nuclei'] = df['bare_nuclei'].astype(int)
    df['class'] = df['class'].apply(lambda x: 1 if x == 4 else 0)
    df = df.drop(columns=['id'])
    return df

def calculate_iv_optimized(df: pd.DataFrame, target: str) -> pd.Series:
    """
    Calcula el Information Value (IV) para todas las columnas/características en el dataset.

    Entrada:
    - df: DataFrame (pd.DataFrame) limpio con columnas y variable objetivo binaria (0/1).
    - target: Nombre de la columna objetivo (str).

    Salida:
    - pd.Series: Valores de IV por columna/característica, ordenados descendentemente.
    """
    def woe_iv_optimized_single_feature(df: pd.DataFrame, feature: str, target: str) -> float:
        counts = df.groupby([feature, target]).size()
        count_table = counts.unstack(fill_value=0)

        if 0 not in count_table.columns:
             count_table[0] = 0
        if 1 not in count_table.columns:
             count_table[1] = 1

        count_table = count_table[[0, 1]]
        count_table.columns = ['NonEvent', 'Event']

        total_event = count_table['Event'].sum()
        total_non_event = count_table['NonEvent'].sum()

        if total_event == 0 or total_non_event == 0:
            return 0.0

        epsilon = 1e-6
        count_table['Dist_Event'] = count_table['Event'] / total_event
        count_table['Dist_NonEvent'] = count_table['NonEvent'] / total_non_event

        count_table['Dist_Event'] = count_table['Dist_Event'].replace(0, epsilon)
        count_table['Dist_NonEvent'] = count_table['Dist_NonEvent'].replace(0, epsilon)

        with np.errstate(divide='ignore', invalid='ignore'):
             count_table['WOE'] = np.log(count_table['Dist_Event'] / count_table['Dist_NonEvent'])

        count_table.replace([np.inf, -np.inf], 0, inplace=True)

        count_table['IV'] = (count_table['Dist_Event'] - count_table['Dist_NonEvent']) * count_table['WOE']

        feature_iv = count_table['IV'].sum()

        return feature_iv

    iv_dict = {col: woe_iv_optimized_single_feature(df, col, target)
               for col in df.columns if col != target}

    return pd.Series(iv_dict).sort_values(ascending=False)

def select_strong_predictors(iv_series, threshold=0.02):
    """
    Select predictors with strong IV above the threshold.

    input:
    - iv_series: Series with IV per variable
    - threshold: Minimum IV to consider strong

    output:
    - list: Selected features
    """
    return iv_series[iv_series >= threshold].index.tolist()
