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

    # def pivot(df, column, values):
    #     pivot_df = df.pivot(columns=column, values=values)
    #     pivot_df = pivot_df.drop(pivot_df.columns[0], axis=1)
    #     pivot_df.columns = [f"{values}_{i}" for i in pivot_df.columns]

    #     df = pd.concat([df, pivot_df], axis=1)

    #     df = df.drop(column, axis=1)
    #     df = df.drop(values, axis=1)

    #     return df

    # some columns have a mixture of numerical and categorical data
    df['age'] = df['age'].apply(lambda x: float(x) if x != '> 89' else 90.0)
    df = df[df['nursingchartvalue'].apply(lambda x: str(x).isdigit())]

    # derive helpful feature
    df['bmi'] = df['admissionweight'] / (df['admissionheight'] ** 2)

    # remove redundant data
    df = df.drop(['celllabel', 'labmeasurenamesystem'], axis=1)

    # binary encode columns with two classes
    df['gender'] = df['gender'].replace({'Female': 0, 'Male': 1})

    # pivot each type of test and its result to its own column
    # df = pivot(df, 'nursingchartcelltypevalname', 'nursingchartvalue')
    # df = pivot(df, 'labname', 'labresult')

    df_demographic = df[df['offset'] == 0.0]
    df_sequential = df[df['offset'] != 0.0]

    df_sequential = df_sequential.drop(['bmi','admissionheight','admissionweight','age','ethnicity','gender','unitvisitnumber'], axis=1)
    df_demographic = df_demographic.drop(['cellattributevalue','labname','labresult','nursingchartcelltypevalname','nursingchartvalue','offset'], axis=1)

    # categorical features that need to become dummies
    encode_features_demographic = ['ethnicity']
    encode_features_sequential = ['cellattributevalue', 'nursingchartcelltypevalname', 'labname']

    # encoding
    def encoding(df, encode_features):
        dummy_df = pd.get_dummies(df[encode_features])
        dummy_df = dummy_df.isin([1.0]).mask(~dummy_df.isin([1.0]), np.nan).astype(float)
        df = pd.concat([df, dummy_df], axis=1)
        df = df.drop(encode_features, axis=1)
        return df

    df_demographic = encoding(df_demographic, encode_features_demographic)
    df_sequential = encoding(df_sequential, encode_features_sequential)

    # sort by column name
    df_demographic = df_demographic.sort_index(axis=1)
    df_sequential = df_sequential.sort_index(axis=1)

    print(df_demographic.columns)
    print(df_sequential.columns)

    combined_data = pd.merge(df_demographic, df_sequential, on='patientunitstayid')
    combined_data.to_csv("combined.csv", index=False, float_format='%.14f')

    return df
