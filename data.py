import pandas as pd
import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence


def load_data(x_path):
    # Your code here
    df = pd.read_csv(x_path)
    df = df.drop(df.columns[0], axis=1)
    return df


def split_data(x, y, split=0.8):
    # Your code here
    from sklearn.model_selection import train_test_split

    test_y, train_y = train_test_split(y, test_size=split)

    train_ids = train_y['patientunitstayid'].tolist()
    test_ids = test_y['patientunitstayid'].tolist()

    train_x = x[x['patientunitstayid'].isin(train_ids)]
    test_x = x[x['patientunitstayid'].isin(test_ids)]

    train_x = train_x.reset_index(drop=True)
    test_x = test_x.reset_index(drop=True)

    return train_x, train_y, test_x, test_y


def preprocess_x(df):
    # Your code here

    # cellattributevalue - 5 and null
    # celllabel - 1 and null
    # ethnicity - 6 and null
    # gender - 2 and null
    # labmeasurenamesystem - 1 and null
    # labname - 2 and null
    # nursingchartcelltypevalname - 10 and null

    # some columns have a mixture of numerical and categorical data
    df['age'] = df['age'].apply(lambda x: float(x) if x != '> 89' else 90.0)
    df['nursingchartvalue'] = df['nursingchartvalue'].apply(lambda x: None if type(x) != int else int(x))

    # derive helpful feature
    df['bmi'] = df['admissionweight'] / (df['admissionheight'] ** 2)

    # remove redundant data
    df = df.drop(['celllabel', 'labmeasurenamesystem'], axis=1)

    # binary encode columns with two classes
    df['gender'] = df['gender'].replace({'Female': 0, 'Male': 1})

    def pivot(df, column, values):
        pivot_df = df.pivot(columns=column, values=values)
        pivot_df = pivot_df.drop(pivot_df.columns[0], axis=1)
        pivot_df.columns = [f"{values}_{i}" for i in pivot_df.columns]
        df = pd.concat([df, pivot_df], axis=1)
        df = df.drop(column, axis=1)
        df = df.drop(values, axis=1)
        return df

    # pivot each type of test and its result to its own column
    df = pivot(df, 'nursingchartcelltypevalname', 'nursingchartvalue')
    df = pivot(df, 'labname', 'labresult')

    # categorical features that need to become dummies
    encode_features = ['ethnicity', 'cellattributevalue']

    # numerical features that need normalizzation
    # patient id should not be normalized as its an index
    normalize_features = ['age', 'admissionheight', 'admissionweight', 'bmi', 'offset', 'unitvisitnumber']

    # pivot creates new columns, these columns need to be normalized too
    for c in df.columns:
        if "nursingchartvalue" in c or "labresult" in c:
            normalize_features.append(c)

    # encoding
    dummy_df = pd.get_dummies(df[encode_features])
    dummy_df = dummy_df.isin([1.0]).mask(~dummy_df.isin([1.0]), np.nan).astype(float)
    df = pd.concat([df, dummy_df], axis=1)
    df = df.drop(encode_features, axis=1)

    # normalizing
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # normalized_data = scaler.fit_transform(df[normalize_features])
    # normalized_df = pd.DataFrame(normalized_data, columns=normalize_features)
    # df[normalize_features] = normalized_df

    return df
