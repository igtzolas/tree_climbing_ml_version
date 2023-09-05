
from typing import List, Union
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.compose import make_column_selector as selector

'''
Useful links:
- https://inria.github.io/scikit-learn-mooc/python_scripts/03_categorical_pipeline.html
- https://github.com/itzzmeakhi/medium-blog-posts


Some notes:
In general OneHotEncoder is the encoding strategy used when the downstream models are linear models while OrdinalEncoder is often a good strategy with tree-based models.

Using an OrdinalEncoder will output ordinal categories. This means that there is an order in the resulting categories (e.g. 0 < 1 < 2). 
The impact of violating this ordering assumption is really dependent on the downstream models. 
Linear models will be impacted by misordered categories while tree-based models will not.

You can still use an OrdinalEncoder with linear models but you need to be sure that:
- the original categories (before encoding) have an ordering;
- the encoded categories follow the same ordering than the original categories.
'''

# df = pd.read_csv("./cat-in-the-dat/train.csv")

def get_types(df: pd.DataFrame, columns: Union[List, None] = None):
    if not columns:
        return df.dtypes.to_dict()
    return df.dtypes[columns].to_dict()

def get_categorical_variables(df: pd.DataFrame):
    categorical_columns_selector = selector(dtype_include=object)
    categorical_columns = categorical_columns_selector(df)
    return categorical_columns    

def one_hot_encode_column(df: pd.DataFrame, column: str, drop_first = True):
    '''
    The column is removed from the dataframe
    new columns are added to the dataframe with names : column + _ + column_value
    '''
    return pd.get_dummies(df, columns = [column], drop_first=drop_first)
# result_1 = one_hot_encode_column(df, "nom_1")
# print(result_1.loc[:, [col for col in result_1.columns if col.startswith("nom_1")]].sum())

def one_hot_encode_column_sklearn(df: pd.DataFrame, column: str):
    categories = df[column].unique().tolist()
    one_hot_encoder = OneHotEncoder()
    # one_hot_encoder.set_params(categories = [f"{column}_{category}" for category in categories]).to
    result = one_hot_encoder.fit_transform(df.loc[:, [column]])
    categories = [f"{column}_{category}" for category in one_hot_encoder.categories_[0].tolist()]
    return pd.DataFrame(result.toarray(), columns=categories)
# result_2 = one_hot_encode_column_sklearn(df, "nom_1")
# print(result_2.sum())

def label_encode(df: pd.DataFrame, column: str):
    label_encoder = LabelEncoder()
    result = label_encoder.fit_transform(df.loc[:, column])
    return  pd.DataFrame(result, columns = [f"{column}_label_encoded"]), label_encoder
# result_3 = label_encode(df, "ord_1")

def ordinal_encode(df: pd.DataFrame, column: str, categories: Union[List, None] = None):
    if categories:
        ordinal_encoder = OrdinalEncoder(categories = [categories]) # Should be a list of lists
    else:
        ordinal_encoder = OrdinalEncoder()
    result = ordinal_encoder.fit_transform(df.loc[:, [column]])
    return  pd.DataFrame(result, columns = [f"{column}_ordinal_encoded"])
# df["ord_1"].unique()
# array(['Grandmaster', 'Expert', 'Novice', 'Contributor', 'Master'],
#       dtype=object)
# categories = ["Contributor", "Novice", "Expert", "Master", "Grandmaster"]
# result_4 = ordinal_encode(df, "ord_1", categories=categories)