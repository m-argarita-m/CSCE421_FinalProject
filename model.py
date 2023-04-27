from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.metrics import roc_auc_score

import pandas as pd
import numpy as np

class Model():
    def __init__(self, args):
        ############################ Your Code Here ############################
        # Initialize your model in this space
        # You can add arguements to the initialization as needed
        self.scaler = None
        self.normalize_features = None
        self.columns = None

        self.encoded_columns = ['ethnicity', 'cellattributevalue']
        self.normalized_columns = ['age', 'admissionheight', 'admissionweight', 'bmi', 'offset', 'unitvisitnumber']
        self.pivot_columns = ['nursingchartvalue', 'labresult']

        self.model = None
        ########################################################################

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        ############################ Your Code Here ############################
        # Fit your model to the training data here

        self.columns = x_train.columns

        x_train = self.train_normalize(x_train)
        x_train = self.aggregate(x_train)

        x_train = x_train.sort_values('patientunitstayid')
        y_train = y_train.sort_values('patientunitstayid')

        x_train = x_train.drop('patientunitstayid', axis=1)

        self.model = HistGradientBoostingClassifier()
        self.model.fit(X=x_train, y=y_train['hospitaldischargestatus'])

        ########################################################################

    def predict_proba(self, x):
        ############################ Your Code Here ############################
        # Predict the probability of in-hospital mortaility for each x

        x = self.fix_columns(self.columns, x)
        x = self.test_normalize(x)
        x = self.aggregate(x)

        ids = x['patientunitstayid']
        x = x.drop('patientunitstayid', axis=1)

        preds = self.model.predict_proba(x)

        df_combined = pd.concat([ids, pd.Series(preds[:,1])], axis=1)

        cols = df_combined.columns.tolist()
        cols[1] = 'hospitaldischargestatus'
        df_combined.columns = cols

        df_combined.to_csv("submission.csv", index=False, float_format='%.14f')

        ########################################################################
        return preds

    def score(self, x_val, y_val):
        x_val = self.fix_columns(self.columns, x_val)
        x_val = self.test_normalize(x_val)
        x_val = self.aggregate(x_val)

        x_val = x_val.sort_values('patientunitstayid')
        y_val = y_val.sort_values('patientunitstayid')

        x_val.to_csv("x_val.csv", index=False)

        x_val = x_val.drop('patientunitstayid', axis=1)

        pred_proba = self.model.predict_proba(x_val)

        score = roc_auc_score(y_val['hospitaldischargestatus'], pred_proba[:,1])
        print('ROC-AOC:', score)

    def train_normalize(self, df):
        # pivot creates new columns, these columns need to be normalized too
        for c in df.columns:
            if "nursingchartvalue" in c or "labresult" in c:
                self.normalized_columns.append(c)

        # normalizing
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        normalized_data = self.scaler.fit_transform(df[self.normalized_columns])
        normalized_df = pd.DataFrame(normalized_data, columns=self.normalized_columns)
        df[self.normalized_columns] = normalized_df

        return df

    def test_normalize(self, df):
        # normalizing
        normalized_data = self.scaler.transform(df[self.normalized_columns])
        normalized_df = pd.DataFrame(normalized_data, columns=self.normalized_columns)
        df[self.normalized_columns] = normalized_df

        return df

    def fix_columns(self, train_x_columns, test_df):
        if not set(train_x_columns) == set(test_df.columns):
            missing_columns = set(train_x_columns) - set(test_df.columns)

            for column in missing_columns:
                # if self.is_column_encoded(column):
                #     test_df[column] = 0
                # else:
                #     test_df[column] = np.nan

                test_df[column] = np.nan

            additional_columns = set(test_df.columns) - set(train_x_columns)
            test_df = test_df.drop(columns=additional_columns)
            test_df = test_df.sort_index(axis=1)

        return test_df

    def is_column_encoded(self, column_name):
        for c in self.encoded_columns:
            if column_name in c:
                return True
        return False

    def aggregate(self, df):
        df = df.groupby('patientunitstayid', as_index=False).agg(np.nanmean)
        df = df.reset_index(drop=True)
        return df
