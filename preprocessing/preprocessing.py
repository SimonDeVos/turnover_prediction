#TODO add import statements



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

def convert_categorical_variables(x_train, y_train, x_val, x_test, categorical_variables):

    # Use weight of evidence encoding: WOE = ln (p(1) / p(0))
    woe_encoder = woe.WOEEncoder(verbose=1, cols=categorical_variables)
    woe_encoder.fit(x_train, y_train)
    x_train = woe_encoder.transform(x_train)
    x_val = woe_encoder.transform(x_val)
    x_test = woe_encoder.transform(x_test)

    return x_train, x_val, x_test


def standardize(x_train, x_val, x_test):

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
            df_train[key] = df_train[key].fillna('-1')
            df_val[key] = df_val[key].fillna('-1')
            df_test[key] = df_test[key].fillna('-1')
        #   Continuous variables: median imputation
        else:
            median = df_train[key].median()
            df_train[key] = df_train[key].fillna(median)
            df_val[key] = df_val[key].fillna(median)
            df_test[key] = df_test[key].fillna(median)

    assert df_train.isna().sum().sum() == 0 and df_val.isna().sum().sum() == 0 and df_test.isna().sum().sum() == 0

    return df_train, df_val, df_test, categorical_variables

"""DATASET-SPECIFIC PREPROCESSING FUNCTIONS"""

def preprocess_acerta():



    return covariates, labels, amounts, cost_matric, categorical_variables



