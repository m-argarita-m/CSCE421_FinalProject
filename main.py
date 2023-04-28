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

    # train_x, train_y, val_x, val_y = split_data(x, y, 2015)
    train_x, train_y = x, y

    ###### Your Code Here #######
    # Add anything you want here

    print("print(train_x.shape)", train_x.shape)
    print("print(train_y.shape)", train_y.shape)


    test_x = load_data("data/test_x.csv")

    ############################

    processed_x_train = preprocess_x(train_x)
    processed_x_test = preprocess_x(test_x)

    ###### Your Code Here #######
    # Add anything you want here

    # processed_x_val = preprocess_x(val_x)

    ############################

    model = Model(args)  # you can add arguments as needed
    model.fit(processed_x_train, train_y)
    x = load_data("data/test_x.csv")

    ###### Your Code Here #######
    # Add anything you want here

    # model.score(processed_x_val, val_y)

    ############################

    processed_x_test = preprocess_x(x)

    prediction_probs = model.predict_proba(processed_x_test)

    #### Your Code Here ####
    # Save your results

    ########################


if __name__ == "__main__":
    main()
