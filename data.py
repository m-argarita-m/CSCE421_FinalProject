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
    df = df[(df['nursingchartvalue'].isnull()) | (df['nursingchartvalue'].apply(lambda x: str(x).isdigit()))]

    # derive helpful feature
    df['bmi'] = df['admissionweight'] / (df['admissionheight'] ** 2)

    # remove redundant data
    df = df.drop(['celllabel', 'labmeasurenamesystem'], axis=1)

    # encode columns
    gender_map = {'Female': 1, 'Male': 2, '': 0}
    df['gender'] = df['gender'].apply(lambda x: gender_map.get(x, 0))

    ethnicity_map = {'Asian': 1, 'African American': 2, 'Caucasian': 3, 'Hispanic': 4, 'Native American': 5, 'Other/Unknown': 0, '': 0}
    df['ethnicity'] = df['ethnicity'].apply(lambda x: ethnicity_map.get(x, 0))

    cellattributevalue_map = {'< 2 seconds': 1, '> 2 seconds': 2, 'feets': 3, 'hands': 4, 'normal': 5, '': 0}
    df['cellattributevalue'] = df['cellattributevalue'].apply(lambda x: cellattributevalue_map.get(x, 0))

    nursingchartcelltypevalname_map = {'GCS Total': 1, 'Heart Rate': 2, 'Invasive BP Diastolic': 3, 'Invasive BP Mean': 4, 'Invasive BP Systolic': 5, 'Non-Invasive BP Diastolic': 6, 'Non-Invasive BP Mean': 7, 'Non-Invasive BP Systolic': 8, 'O2 Saturation': 9, 'Respiratory Rate': 10, '': 0}
    df['nursingchartcelltypevalname'] = df['nursingchartcelltypevalname'].apply(lambda x: nursingchartcelltypevalname_map.get(x, 0))

    labname_map = {'glucose': 1, 'pH': 2, '': 0}
    df['labname'] = df['labname'].apply(lambda x: labname_map.get(x, 0))

    df_demographic = df[df['unitvisitnumber'].notnull()]
    df_sequential = df[df['unitvisitnumber'].isnull()]

    df_sequential = df_sequential.drop(['bmi','admissionheight','admissionweight','age','ethnicity','gender','unitvisitnumber'], axis=1)
    df_demographic = df_demographic.drop(['cellattributevalue','labname','labresult','nursingchartcelltypevalname','nursingchartvalue','offset'], axis=1)

    # sort by column name
    df_demographic = df_demographic.sort_index(axis=1)
    df_sequential = df_sequential.sort_index(axis=1)

    combined_data = pd.merge(df_demographic, df_sequential, on='patientunitstayid')
    combined_data.to_csv("combined.csv", index=False, float_format='%.14f')

    return df
