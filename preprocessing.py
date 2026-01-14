# preprocessing

import numpy as np
import pandas as pd

def clean_data(df):
    # Numerical missing values
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Categorical missing values
    for col in ['Gender', 'Married', 'Dependents', 'Self_Employed']:
        if col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Outlier treatment
    if 'Loan_Amount_Term' in df.columns:
        df['Loan_Amount_Term'] = np.log(df['Loan_Amount_Term'])

    return df


def prepare_features(train, test):
    train = train.drop('Loan_ID', axis=1)
    test = test.drop('Loan_ID', axis=1)

    X = train.drop('Loan_Status', axis=1)
    y = train['Loan_Status']

    X = pd.get_dummies(X)
    test = pd.get_dummies(test)

    X, test = X.align(test, join='left', axis=1, fill_value=0)

    return X, y, test
