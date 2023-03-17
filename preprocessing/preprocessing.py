#TODO add import statements

import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce
from category_encoders import woe
import time

import warnings
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)    #TODO: solve warning instead of suppressing it

"""Contents:
import statements
3 generic preprocessing functions:
    def convert_categorical_variables
    def standardize
    def handle_missing_data

dataset-specific functions:
    def preprocess_acerta()
        return covariates, labels, amounts, cost_matrix, categorical_variables
    def preprocess_babushkin()
    def preprocess_eds()
    def preprocess_ibm()

"""

# Set random seed
random.seed(42)

"""3 GENERIC PREPROCESSING FUNCTIONS"""

def convert_categorical_variables(x_train, y_train, x_val, x_test, categorical_variables, cat_encoder):

    if cat_encoder == 1:
        """
        onehot_encoder = OneHotEncoder() #TODO: verder aanvullen
#        onehot_encoder.fit(x_train, y_train)
        onehot_encoder.fit(x_train[categorical_variables])
        
        x_train = onehot_encoder.transform(x_train)
        x_val = onehot_encoder.transform(x_val)
        x_test = onehot_encoder.transform(x_test)
        """
        """
        encoder = OneHotEncoder(handle_unknown='ignore')
        x_train_cat_encoded = encoder.fit_transform(x_train[:, categorical_variables])
        x_val_cat_encoded = encoder.transform(x_val[:, categorical_variables])
        x_test_cat_encoded = encoder.transform(x_test[:, categorical_variables])

        x_train_encoded = np.concatenate((x_train[:, [0]], x_train_cat_encoded.toarray()), axis=1)
        x_val_encoded = np.concatenate((x_val[:, [0]], x_val_cat_encoded.toarray()), axis=1)
        x_test_encoded = np.concatenate((x_test[:, [0]], x_test_cat_encoded.toarray()), axis=1)
        """
        """
        numerical_variables = list(set(range(x_train.shape[1])) - set(categorical_variables))
#        x_num = x_train[:, numerical_variables]  # numerical features
#        x_cat = x_train[:, categorical_variables]  # categorical features

        # encode categorical features using one-hot encoding
        encoder = OneHotEncoder(handle_unknown='ignore')
        x_train_cat_encoded = encoder.fit_transform(x_train[:, categorical_variables])
        x_val_cat_encoded = encoder.transform(x_val[:, categorical_variables])
        x_test_cat_encoded = encoder.transform(x_test[:, categorical_variables])

        # concatenate numerical and encoded categorical features
        x_train = np.concatenate((x_train[:, numerical_variables], x_train_cat_encoded.toarray()), axis=1)
        x_val = np.concatenate((x_val[:, numerical_variables], x_val_cat_encoded.toarray()), axis=1)
        x_test = np.concatenate((x_test[:, numerical_variables], x_test_cat_encoded.toarray()), axis=1)
        """
        ce_one_hot = ce.OneHotEncoder(cols=categorical_variables)   #TODO: question : what happens if categorical values exist in val or test, but not in train? How is this encoded?
        ce_one_hot.fit(x_train)
        x_train = ce_one_hot.transform(x_train)
        x_val = ce_one_hot.transform(x_val)
        x_test = ce_one_hot.transform(x_test)

    elif cat_encoder == 2:
        # Use weight of evidence encoding: WOE = ln (p(1) / p(0))
        woe_encoder = woe.WOEEncoder(verbose=1, cols=categorical_variables)
        woe_encoder.fit(x_train, y_train)
        x_train = woe_encoder.transform(x_train)
        x_val = woe_encoder.transform(x_val)
        x_test = woe_encoder.transform(x_test)

    return x_train, x_val, x_test


def standardize(x_train, x_val, x_test):

    #print(x_train.dtypes)
    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train_scaled = scaler.transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    # For compatibility with xgboost package: make contiguous arrays
    x_train_scaled = np.ascontiguousarray(x_train_scaled)
    x_val_scaled = np.ascontiguousarray(x_val_scaled)
    x_test_scaled = np.ascontiguousarray(x_test_scaled)

    return x_train_scaled, x_val_scaled, x_test_scaled


def handle_missing_data(df_train, df_val, df_test, categorical_variables):
    #print(str(df_train.keys()))
    for key in df_train.keys():
        # If variable has > 90% missing values: delete
        if df_train[key].isna().mean() > 0.9:
            df_train = df_train.drop(key, 1)
            df_val = df_val.drop(key, 1)
            df_test = df_test.drop(key, 1)

            if key in categorical_variables:
                categorical_variables.remove(key)
            continue

        # Handle other missing data:
        #   Categorical variables: additional category '-1'
        if key in categorical_variables:
            df_train[key].fillna('-1', inplace = True)
            df_val[key].fillna('-1', inplace = True)
            df_test[key].fillna('-1', inplace = True)

        #   Continuous variables: median imputation
        else:
            median = df_train[key].median()
#            median = df_train.loc[key].median()

#            df_train.loc[key] = df_train.loc[key].fillna(median)
#            df_train[key] = df_train[key].fillna(median)
            df_train[key].fillna(median, inplace=True)
#            df_train.loc[:, key] = df_train[key].fillna(median)

#            df_val.loc[key] = df_val.loc[key].fillna(median)
#            df_val[key] = df_val[key].fillna(median)
            df_val[key].fillna(median, inplace=True)
#            df_val.loc[:, key] = df_val[key].fillna(median)

#            df_test.loc[key] = df_test.loc[key].fillna(median)
#            df_test[key] = df_test[key].fillna(median)
            df_test[key].fillna(median, inplace=True)
#            df_test.loc[:, key] = df_test[key].fillna(median)

    assert df_train.isna().sum().sum() == 0 and df_val.isna().sum().sum() == 0 and df_test.isna().sum().sum() == 0

    return df_train, df_val, df_test, categorical_variables

"""DATASET-SPECIFIC PREPROCESSING FUNCTIONS"""

# read data
# Drop ID and useless columns
# fill meaningful NaN
# transform Turnover feature to 0/1
# Split into covariates, labels
# Create cost matrix
# List categorical variables


def preprocess_acerta(): #TODO
    return covariates, labels, amounts, cost_matrix, categorical_variables


def preprocess_ibm():

    try:
        df = pd.read_csv('data/ibm.csv', sep=',')
    except FileNotFoundError:
        df = pd.read_csv('../data/ibm.csv', sep=',')

    # Drop ID and useless columns
    df = df.drop('EmployeeNumber', axis=1)
    df = df.drop('EmployeeCount', axis=1)
    df = df.drop('Over18', axis=1)
    df = df.drop('StandardHours', axis=1)

    # Transform 'Attrition' from Yes/No to 1/0
    df['Attrition'] = df['Attrition'].replace({'Yes': 1, 'No': 0})

    # Split into covariates, labels
    labels = df['Attrition'].values.astype(int)
    covariates = df.drop('Attrition', axis=1)

    # Create cost matrix
    income = covariates['MonthlyIncome'].values

    n_samples = income.shape[0]
    cost_matrix = np.zeros((n_samples, 2, 2))     # cost_matrix [[TN, FN], [FP, TP]]
    cost_matrix[:, 0, 0] = 0.0
    cost_matrix[:, 0, 1] = income #not detected, lose employee, cost is 6 months salary
    cost_matrix[:, 1, 0] = income   #predicted, not lose employee, cost is fixed cost 500 (intervention)
    cost_matrix[:, 1, 1] = 0   #predicted, lost employee anyway, cost is fixed cost 500 (intervention)

    # List categorical variables
    categorical_variables = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']

    amounts = income

    return covariates, labels, amounts, cost_matrix, categorical_variables

def preprocess_kaggle1():
    #read data
    try:
        df = pd.read_csv('data/kaggle1.csv', sep=',')
    except FileNotFoundError:
        df = pd.read_csv('../data/kaggle1.csv', sep=',')

    # Drop ID and useless columns

    # fill meaningful NaN
    df['filed_complaint'].fillna(0, inplace=True)
    df['recently_promoted'].fillna(0, inplace=True)

    # Transform 'Attrition' from Yes/No to 1/0
    df['status'] = df['status'].replace({'Left': 1, 'Employed': 0})

    # Split into covariates, labels
    labels = df['status'].values.astype(int)
    covariates = df.drop('status', axis=1)


    # Create cost matrix
    n_samples = labels.shape[0]
    cost_matrix = np.zeros((n_samples, 2, 2))     # cost_matrix [[TN, FN], [FP, TP]]
    cost_matrix[:, 0, 0] = 0
    cost_matrix[:, 0, 1] = 1
    cost_matrix[:, 1, 0] = 1
    cost_matrix[:, 1, 1] = 0

    # List categorical variables
    categorical_variables = ['department','salary']

    amounts = np.ones((n_samples))

    return covariates, labels, amounts, cost_matrix, categorical_variables


def preprocess_kaggle2(): #old name: Babushkin
    #read data
    try:
        df = pd.read_csv('data/kaggle2.csv', sep=',')
    except FileNotFoundError:
        df = pd.read_csv('../data/kaggle2.csv', sep=',')

    # Drop ID and useless columns

    # Transform 'gender' (f/m), 'coach' (yes/no), 'greywage' (grey/white), 'head_gender' (f/m) to 1/0
    df['gender'] = df['gender'].replace({'f': 1, 'm': 0})
    df['head_gender'] = df['head_gender'].replace({'f': 1, 'm': 0})
    df['greywage'] = df['greywage'].replace({'grey':1, 'white':0})


    # Split into covariates, labels
    labels = df['event'].values.astype(int)
    covariates = df.drop('event', axis=1)

    # Create cost matrix
    n_samples = labels.shape[0]
    cost_matrix = np.zeros((n_samples, 2, 2))     # cost_matrix [[TN, FN], [FP, TP]]
    cost_matrix[:, 0, 0] = 0
    cost_matrix[:, 0, 1] = 1
    cost_matrix[:, 1, 0] = 1
    cost_matrix[:, 1, 1] = 0

    # List categorical variables
    categorical_variables = ['industry','profession','traffic','coach','way']

    # define amounts
    amounts = np.ones((n_samples))

    return covariates, labels, amounts, cost_matrix, categorical_variables

def preprocess_kaggle3(): #TODO
    # read data
    try:
        df = pd.read_csv('data/kaggle3.csv', sep=',')
    except FileNotFoundError:
        df = pd.read_csv('../data/kaggle3.csv', sep=',')

    #remove time element (features do not change over time)
    df_grouped = df.groupby('EmployeeID').last().reset_index()

    # Drop ID and useless columns
    df = df_grouped.drop(
        columns=['EmployeeID', 'recorddate_key', 'birthdate_key', 'orighiredate_key', 'terminationdate_key',
                 'gender_full', 'terminationdate_key', 'termtype_desc','termreason_desc', 'STATUS_YEAR'])

    # Transform binary vars to 1/0
    df['gender_short'] = df['gender_short'].replace({'F': 1, 'M': 0})
    df['STATUS'] = df['STATUS'].replace({'TERMINATED': 1, 'ACTIVE': 0})

    # Split into covariates, labels
    labels = df['STATUS'].values.astype(int)
    covariates = df.drop('STATUS', axis=1)

    # Create cost matrix
    n_samples = labels.shape[0]
    cost_matrix = np.zeros((n_samples, 2, 2))     # cost_matrix [[TN, FN], [FP, TP]]
    cost_matrix[:, 0, 0] = 0
    cost_matrix[:, 0, 1] = 1
    cost_matrix[:, 1, 0] = 1
    cost_matrix[:, 1, 1] = 0

    # List categorical variables
    categorical_variables = ['city_name','department_name','job_title','BUSINESS_UNIT']

    # define amounts
    amounts = np.ones((n_samples))

    return covariates, labels, amounts, cost_matrix, categorical_variables

def preprocess_kaggle4(): # TODO
    # read data
    try:
        df = pd.read_csv('data/kaggle4.csv', sep=',')
    except FileNotFoundError:
        df = pd.read_csv('../data/kaggle4.csv', sep=',')

    # feature engineering, based on temporal data (diff with prev period)
    # rename columns
    df = df.rename(columns={'MMM-YY': 'timestamp'})
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d')
    df['Dateofjoining'] = pd.to_datetime(df['Dateofjoining'], format='%Y-%m-%d')
    # calculate the time difference in seconds and replace negative values with 0
    df['tenure'] = (df['timestamp'] - df['Dateofjoining']).dt.days.astype(int).clip(lower=0)
    df['y'] = np.where(df['LastWorkingDate'].isnull(), 0, 1)
    df['salary_diff'] = df.groupby('Emp_ID')['Salary'].diff().fillna(0)
    df['Q_rating_diff'] = df.groupby('Emp_ID')['Quarterly Rating'].diff().fillna(0)
    df['TBV_diff'] = df.groupby('Emp_ID')['Total Business Value'].diff().fillna(0)

    # Transform binary vars to 1/0
    df['Gender'] = df['Gender'].replace({'Male': 1, 'Female': 0})

    # Drop ID and useless columns
    df = df.drop(columns=['Emp_ID','Dateofjoining','LastWorkingDate'])

    # List categorical variables
    df['timestamp'] = df['timestamp'].astype('object')
    categorical_variables = ['City', 'Education_Level', 'timestamp']

    # Split into covariates, labels
    labels = df['y'].values.astype(int)
    covariates = df.drop('y', axis=1)

    # Create cost matrix
    n_samples = labels.shape[0]
    cost_matrix = np.zeros((n_samples, 2, 2))     # cost_matrix [[TN, FN], [FP, TP]]
    cost_matrix[:, 0, 0] = 0
    cost_matrix[:, 0, 1] = 1
    cost_matrix[:, 1, 0] = 1
    cost_matrix[:, 1, 1] = 0

    # define amounts
    amounts = np.ones((n_samples))

    #print(df.dtypes)

    return covariates, labels, amounts, cost_matrix, categorical_variables

def preprocess_kaggle5(): #TODO
    # read data

    # Drop ID and useless columns

    # Transform binary vars to 1/0

    # Split into covariates, labels

    # Create cost matrix

    # List categorical variables

    # define amounts

    return covariates, labels, amounts, cost_matrix, categorical_variables

def preprocess_kaggle6(): #TODO
    # read data

    # Drop ID and useless columns

    # Transform binary vars to 1/0

    # Split into covariates, labels

    # Create cost matrix

    # List categorical variables

    # define amounts

    return covariates, labels, amounts, cost_matrix, categorical_variables

def preprocess_kaggle7(): #TODO
    # read data

    # Drop ID and useless columns

    # Transform binary vars to 1/0

    # Split into covariates, labels

    # Create cost matrix

    # List categorical variables

    # define amounts

    return covariates, labels, amounts, cost_matrix, categorical_variables

