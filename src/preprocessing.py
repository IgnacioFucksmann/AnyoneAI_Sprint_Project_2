from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre processes data for modeling. Receives train, val and test dataframes
    and returns numpy ndarrays of cleaned up dataframes with feature engineering
    already performed.

    Arguments:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        train : np.ndarrary
        val : np.ndarrary
        test : np.ndarrary
    """
    # Print shape of input data
    print("Input train data shape: ", train_df.shape)
    print("Input val data shape: ", val_df.shape)
    print("Input test data shape: ", test_df.shape, "\n")

    # Make a copy of the dataframes
    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    # 1. Correct outliers/anomalous values in numerical
    # columns (`DAYS_EMPLOYED` column).
    working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_val_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)

    # 2. TODO Encode string categorical features (dytpe `object`):
    #     - If the feature has 2 categories encode using binary encoding,
    #       please use `sklearn.preprocessing.OrdinalEncoder()`. Only 4 columns
    #       from the dataset should have 2 categories.
    #     - If it has more than 2 categories, use one-hot encoding, please use
    #       `sklearn.preprocessing.OneHotEncoder()`. 12 columns
    #       from the dataset should have more than 2 categories.
    # Take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the OrdinalEncoder and
    #     OneHotEncoder classes, then use the fitted models to transform all the
    #     datasets.
    binary_cols = [
        "NAME_CONTRACT_TYPE",
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY",
        "EMERGENCYSTATE_MODE",
    ]
    ordinal_encoder = OrdinalEncoder()
    ordinal_encoder.fit(working_train_df[binary_cols])

    working_train_df[binary_cols] = ordinal_encoder.transform(
        working_train_df[binary_cols]
    )
    working_val_df[binary_cols] = ordinal_encoder.transform(working_val_df[binary_cols])
    working_test_df[binary_cols] = ordinal_encoder.transform(
        working_test_df[binary_cols]
    )

    multi_category_cols = [
        "CODE_GENDER",
        "NAME_TYPE_SUITE",
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE",
        "OCCUPATION_TYPE",
        "WEEKDAY_APPR_PROCESS_START",
        "ORGANIZATION_TYPE",
        "FONDKAPREMONT_MODE",
        "HOUSETYPE_MODE",
        "WALLSMATERIAL_MODE",
    ]
    onehot_encoder = OneHotEncoder(sparse_output=False)
    onehot_encoder.fit(working_train_df[multi_category_cols])

    train_encoded = pd.DataFrame(
        onehot_encoder.transform(working_train_df[multi_category_cols]),
        columns=onehot_encoder.get_feature_names_out(multi_category_cols),
        index=working_train_df.index,
    )
    val_encoded = pd.DataFrame(
        onehot_encoder.transform(working_val_df[multi_category_cols]),
        columns=onehot_encoder.get_feature_names_out(multi_category_cols),
        index=working_val_df.index,
    )
    test_encoded = pd.DataFrame(
        onehot_encoder.transform(working_test_df[multi_category_cols]),
        columns=onehot_encoder.get_feature_names_out(multi_category_cols),
        index=working_test_df.index,
    )

    working_train_df = working_train_df.drop(columns=multi_category_cols)
    working_val_df = working_val_df.drop(columns=multi_category_cols)
    working_test_df = working_test_df.drop(columns=multi_category_cols)

    working_train_df = pd.concat([working_train_df, train_encoded], axis=1)
    working_val_df = pd.concat([working_val_df, val_encoded], axis=1)
    working_test_df = pd.concat([working_test_df, test_encoded], axis=1)

    # 3. TODO Impute values for all columns with missing data or, just all the columns.
    # Use median as imputing value. Please use sklearn.impute.SimpleImputer().
    # Again, take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the SimpleImputer and then use the fitted
    #     model to transform all the datasets.
    imputer = SimpleImputer(strategy="median")
    imputer.fit(working_train_df)
    working_train_df = imputer.transform(working_train_df)
    working_val_df = imputer.transform(working_val_df)
    working_test_df = imputer.transform(working_test_df)

    # 4. TODO Feature scaling with Min-Max scaler. Apply this to all the columns.
    # Please use sklearn.preprocessing.MinMaxScaler().
    # Again, take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the MinMaxScaler and then use the fitted
    #     model to transform all the datasets.
    scaler = MinMaxScaler()
    scaler.fit(working_train_df)

    train = scaler.transform(working_train_df)
    val = scaler.transform(working_val_df)
    test = scaler.transform(working_test_df)

    return train, val, test
