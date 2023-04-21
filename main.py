# from project_utils import create_data_for_project

# data = create_data_for_project(".")

import itertools

import torch
import pandas as pd
from sklearn.metrics import roc_auc_score


from data import load_data, preprocess_x, split_data
from parser import parse
from model import Model


def main():
    args = parse()

    x = load_data("data/train_x.csv")
    y = load_data("data/train_y.csv")

    train_x, train_y, test_x, test_y = split_data(x, y)

    ###### Your Code Here #######
    # Add anything you want here


    test_x = load_data("data/test_x.csv")

    ############################

    processed_x_train = preprocess_x(train_x)
    processed_x_test = preprocess_x(test_x)

    ###### Your Code Here #######
    # Add anything you want here

    import numpy as np

    processed_x_train = processed_x_train.rename(columns={'ethnicity_Native American': 'ethnicity_Native_American'})
    processed_x_test = processed_x_test.assign(ethnicity_Native_American=np.zeros(len(processed_x_test)))

    processed_x_train = processed_x_train.sort_index(axis=1)
    processed_x_test = processed_x_test.sort_index(axis=1)

    x_train_agg = processed_x_train.groupby('patientunitstayid', as_index=False).agg(np.nanmean)
    x_train_agg = x_train_agg.reset_index(drop=True)
    # x_train_agg.to_csv("testinglol.csv")

    x_test_agg = processed_x_test.groupby('patientunitstayid', as_index=False).agg(np.nanmean)
    x_test_agg = x_test_agg.reset_index(drop=True)
    # x_test_agg.to_csv("testinglol.csv")

    train_y = train_y.sort_values('patientunitstayid')
    train_y = train_y.drop(train_y.columns[0], axis=1)
    test_y = test_y.sort_values('patientunitstayid')
    test_y = test_y.drop(test_y.columns[0], axis=1)

    from sklearn.ensemble import HistGradientBoostingClassifier
    model = HistGradientBoostingClassifier()

    model.fit(X=x_train_agg, y=train_y.values.ravel())

    from sklearn.metrics import roc_auc_score

    # class_probs = model.predict_proba(x_test_agg)
    y_pred_proba = model.predict_proba(x_test_agg)
    print(y_pred_proba)
    df_combined = pd.concat([x_test_agg['patientunitstayid'], pd.Series(y_pred_proba[:,1])], axis=1)

    cols = df_combined.columns.tolist()
    cols[1] = 'hospitaldischargestatus'
    df_combined.columns = cols

    df_combined.to_csv("submission.csv", index=False, float_format='%.14f')




    # auc_roc = roc_auc_score(test_y, y_pred_proba[:,1])
    # print("Taadaa: ", auc_roc)
    exit()

    # # x_train_agg = arr[np.argsort(arr[:, 1])]

    ############################

    model = Model(args)  # you can add arguments as needed
    model.fit(processed_x_train, train_y)
    x = load_data("test_x.csv")

    ###### Your Code Here #######
    # Add anything you want here

    ############################

    processed_x_test = preprocess_x(x)

    prediction_probs = model.predict_proba(processed_x_test)

    #### Your Code Here ####
    # Save your results

    ########################


if __name__ == "__main__":
    main()
