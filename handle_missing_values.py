

from typing import Any, List, Union
from lightgbm import LGBMClassifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

# train_df = pd.read_csv("./song-popularity-prediction/train.csv")
# test_df = pd.read_csv("./song-popularity-prediction/test.csv")
# train_df["is_train"] = True
# test_df["is_train"] = False
# train_test_df = pd.concat([train_df, test_df]).reset_index(drop=True).copy()

'''
Why do we have missing values ???
- Sensor data where the sensor went offline.
- Survey data where some questions were not answered.


Strategies to handle missing values :

DO NOTHING
- LightGBM handles missing values out of the box (https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html)
- XGBoost handles missing values out of the box

DROP THE MISSING VALUES (either rows or columns)

USE PANDAS IMPUTATION

SKLEARN IMPUTATION 
SimpleImputer Similar to pandas fillna
IterativeImputer
KNNImputer

LIGHTGBM IMPUTATION

'''

# Are there any missing values and in which columns ? 
# per column missing values
def plot_bar_chart_missing_values_train_test(train_df, test_df):
    df = pd.DataFrame([train_df.isna().mean(), test_df.isna().mean()]).T
    df = df.rename(columns={0: "train_missing", 1: "test_missing"})

    df.loc[df["train_missing"] > 0, :].plot(
        kind="barh", figsize=(8, 5), title="% of Values Missing"
    )
    plt.show()
# plot_bar_chart_missing_values_train_test(train_df, test_df)

def rows_without_missing_values(df: pd.DataFrame):
    return df.dropna(axis = 0)

def columns_without_missing_values(df: pd.DataFrame):
    return df.dropna(axis = 1)

def impute_na_column_with_specific_value(df:pd.DataFrame, value: Any, columns: Union[List, None] = None):
    if not columns:
        return df.fillna(value)
    else:
        temp_1_df = df.loc[:, columns].fillna(value)
        temp_2_df = df.drop(columns, axis = 1)
        return pd.concat([temp_1_df, temp_2_df], axis = 1)

def impute_na_column_with_strategy(df: pd.DataFrame, column: str, strategy = "mean"):
    if strategy == "mean":
        df[column] = df[column].fillna(df[column].mean())
    if strategy == "median":
        df[column] = df[column].fillna(df[column].median())
    return df

def impute_na_column_with_groupby_other_column(df: pd.DataFrame, column_to_fill: str, column_to_groupby: str, strategy: str = "mean"):
    mapping = {}
    if strategy == "mean":
        mapping = df.groupby(column_to_groupby)[column_to_fill].mean().to_dict()
    if strategy ==  "median":
        mapping = df.groupby(column_to_groupby)[column_to_fill].median().to_dict()
    mapped_series = df[column_to_groupby].map(mapping)
    df[column_to_fill] = df[column_to_fill].fillna(mapped_series)
    return df

def impute_na_column_with_strategy_sklearn_simple_imputer(df: pd.DataFrame, column: str, strategy : str = "mean", add_indicator: bool = True, constant_value: Any = None):
    '''
    strategy can be one of : ["mean", "median", "most_frequent", "constant"]
    add_indicator : adds new column indicating whether imputation has taken place for the value in the row or not
    '''
    if strategy == "constant" and constant_value:
        imputer = SimpleImputer(strategy=strategy, add_indicator=add_indicator, fill_value=constant_value)    
    else:
        imputer = SimpleImputer(strategy=strategy, add_indicator=add_indicator)
    imputed_columns = imputer.fit_transform(df.loc[:, [column]])
    if add_indicator:
        imputed_columns_df = pd.DataFrame(imputed_columns, columns = [column, f"{column}_is_imputed"])
    else:
        imputed_columns_df = pd.DataFrame(imputed_columns, columns = [column])
    return pd.concat([df.drop([column], axis = 1), imputed_columns_df], axis = 1)
# impute_na_column_with_stratege_sklearn_simple_imputer(train_test_df, "energy", "mean", False)
# impute_na_column_with_stratege_sklearn_simple_imputer(train_test_df, "energy", "constant", True, 100000)

def impute_knn_algorithm(df: pd.DataFrame, n_neighbors: int, add_indicator: bool = True):
    imputer = KNNImputer(n_neighbors=n_neighbors, add_indicator=add_indicator)
    imputed_df = imputer.fit_transform(df)
    return imputed_df
# impute_knn_algorithm(train_test_df, n_neighbors=3, add_indicator=True)

# # Using an algorithm that handles missing values out of the box
# lgb_classifier = LGBMClassifier(use_missing = True)
# lgb_classifier.fit(train_df.drop(columns = ["song_popularity"]), train_df["song_popularity"])




