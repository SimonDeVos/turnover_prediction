#Overview to configure experiments.
#Run main.py to launch
import os
import json
import datetime
from experiments import experiment

# Set project directory:
DIR = r'C:\Users\u0148775\PycharmProjects\turnover_prediction\workdir'

# Specify experimental configuration
"""
settings
datasets
methodologies
thresholding
evaluators
"""

settings = {
#    'class_costs': False,
    'folds': 2,
    'repeats': 2,
    'val_ratio': 0.25,  # Relative to training set only (excluding test set) #TODO: outdated (now implemented as nr of folds in CV)
#    'l1_regularization': False,
#    'lambda1_options': [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
#    'l2_regularization': False,
#    'lambda2_options': [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
#    'neurons_options': [2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10],
    'cat_encoder': 2,   #1: one-hot, 2: WOE,
    'scaler': 1,        #1: standardscaler, 2: minmax scaler, 3... #TODO: add more?
    'CV_hyperpara_tuning': 2,  # The nr of CV for hyperpara tuning (splits train_val into train and val)
    'oversampling': 3   #0: none, 1: SMOTE, 2: RandomOversampler, 3: ADASYN
}

datasets = {
    'acerta': False,    # not implemented
    'cegeka': False,    # ok
    'ds': False,        # ok
    'ibm': False,
    'imec': False,      # ok
    'kaggle1': False,    # ok
    'kaggle2': False,   # ok
    'kaggle3': False,   # ok
    'kaggle4': False,   # ok
    'kaggle5': False,   # ok
    'kaggle6': False,   # ok --> to be deleted (perfect predictions, data leakage in features)
    'kaggle7': False,   # ok
    'medium': True,    # ok
    'rhuebner': False,  # ok (same as kaggle5, but more extensive)
    'techco': False,    # ok
}

methodologies = {
    'ab': False,    # AdaBoost (AB) -                           implemented
    'ann': False,   # Artificial Neural Networks (ANN) - sklearn.neural_network.MLPClassifier #Todo
    'bnb': False,   # Bernoulli Naive Bayes (BNB) -             implemented
    'cb': False,     # CatBoost (CB) -                          implemented #TODO: takes long to train
    'dt': False,    # Decision Tree (DT)-                       implemented
    'gnb': False,   # Gaussian Naive Bayes (GNB)-               implemented
    'gb': False,    # Gradient Boosting (GB)-                   implemented
    'knn': False,    # K-Nearest Neighbors (KNN) -              implemented
    'lgbm': True,   # LightGBM (LGBM) -                        implemented
    'lda': False,   # Linear Discriminant Analysis (LDA) -      implemented
    'lr': False,    # Logistic Regression (LR) -                implemented
    'mnb': False,   # Multinomial Naive Bayes (MNB) -           implemented
    'pac': False,   # Passive Aggressive Classifier (PAC) -     implemented
    'per': True,   # Perceptron (Per) -                        implemented
    'qda': False,   # Quadratic Discriminant Analysis (QDA) -   implemented #TODO: might give "UserWarning: Variables are collinear"
    'rf': False,    # Random Forest (RF) -                      implemented
    'rc': False,    # Ridge Classifier (RC) -                   implemented
    'sgd': False,   # Stochastic Gradient Descent (SGD) -       implemented
    'svm': False,   # Support Vector Machine (SVM) -             implemented #TODO: takes long to train
    'xgb': False    # Extreme Gradient Boosting (XGBoost) -     implemented
}

thresholding = {
    't_idcs': True,        # Instance-dependent cost-sensitive threshold
    't_idcs_cal': False,    # Instance-dependent cost-sensitive threshold with calibrated probabilities
    't_cdcs': False,        # Class-dependent cost-sensitive threshold
    't_cdcs_cal': False,    # Class-dependent cost-sensitive threshold with calibrated probabilities
    't_class_imb': False,   # Class imbalance Threshold
    't_ins': False           # Class insensitive threshold (0.5)
}

evaluators = {
    # Cost-insensitive
    'traditional': True,    # acc, F1, prec, recall
    'ROC': False,           # not implemented
    'AUC': True,
    'PR': False, #TODO: redundant, included in 'traditional'
    'H_measure': True,     #takes into account mix of precision and recall
    'brier': True,
    'recall_overlap': False,        #not implemented
    'recall_correlation': False,    #not implemented

    # Cost-sensitive
    'savings': True,
    'AEC': True,
    'ROCIV': False,     #not implemented
    'PRIV': False,      #not implemented
    'rankings': False,  #not implemented

    # Other
    'time': True, #TODO: there might be a bug in checking significance of time as performance metric (does not detect large differences)

    'stat_hypothesis_testing': True #Perform tests on H0: Model performance follows the same distribution
        #In the context of the Friedman test, the null hypothesis is that there are no significant differences between the groups or treatments. The alternative hypothesis is that at least one group differs significantly from the others.
        #If the p-value is small (e.g., less than 0.05), it suggests that there is strong evidence against the null hypothesis and that at least one group differs significantly from the others. If the p-value is large (e.g., greater than 0.05), it suggests that there is weak or no evidence against the null hypothesis, and that the differences between the groups may be due to chance.
        #However, it's important to note that the p-value alone does not tell you which groups differ significantly from the others. To determine this, you would need to perform post-hoc tests, such as the Nemenyi test or Dunn's test, to compare pairs of groups and identify significant differences. These tests can also adjust for multiple comparisons, which is important when comparing multiple groups.
}

hyperparameters = {
    'ab': {
        'n_estimators': [50, 100, 200],   # [50, 100, 200],
        'learning_rate': [1],   # [0.1, 0.5, 1],
        'algorithm': ['SAMME']  # ['SAMME', 'SAMME.R']
    },

    'bnb': {
        'alpha': [0.1, 1.0],         # [0.01, 0.1, 1.0],
        'binarize': [0.0, 0.5, 1.0],      # [0.0, 0.5, 1.0],
        'fit_prior': [True, False]     # [True, False]
    },

    'cb': {
        'iterations': [100, 200, 300],            # [100, 200, 300],
        'depth': [5],                   # [3, 5, 7, 9],
        'learning_rate': [0.05],        # [0.01, 0.05, 0.1],
        'l2_leaf_reg': [3],             # [1, 3, 5, 7],
        'border_count': [64],           # [32, 64, 128],
        'bagging_temperature': [1]},    # [0, 1, 5, 10]},

    'dt': {
        'criterion': ['entropy'],   # ['gini', 'entropy'],
        'max_depth': [5],           # [None, 5, 10, 15],
        'min_samples_split': [10],  # [2, 5, 10],
        'min_samples_leaf': [2],    # [1, 2, 4],
        'max_features': ['log2']},  # [None, 'sqrt', 'log2']},

    'gb': {
        'n_estimators':  [200],    # [50, 100, 200],
        'learning_rate': [0.1],  # [0.01, 0.1, 0.5],
        'max_depth': [5],             # [3, 5, 7],
        'min_samples_split': [2],    # [2, 5, 10],
        'min_samples_leaf': [1],      # [1, 2, 4],
        'max_features': ['sqrt']},  # ['sqrt', 'log2']},

    'gnb': {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]},   # [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]},

    'knn': {
        'n_neighbors': [5],         # [3, 5, 7, 9],
        'weights': ['uniform'],     # ['uniform', 'distance'],
        'algorithm': ['auto'],      # ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [10]},         # [10, 20, 30, 40]},

    'lda': {
        'solver': ['lsqr'],         # ['svd','lsqr', 'eigen'],    #     #TODO: some combinations of hyperparas are not supported - might give Warnings
        'shrinkage': ['auto'],      # [None, 'auto', 0.1, 0.5, 1.0],
        'n_components': [None]      # [None, 2, 5, 10]
        },

    'lgbm': {
        'learning_rate': [0.05],    # [0.01, 0.05, 0.1],
        'n_estimators': [100],      # [50, 100, 200],
        'max_depth': [5],           # [3, 5, 7],
        'num_leaves': [15],         # [15, 31, 63],
        'subsample': [0.8],         # [0.8, 0.9, 1],
        'colsample_bytree': [0.9]}, # [0.8, 0.9, 1]},

    'lr': {
        'penalty': ['l2'],          # ['l1', 'l2'],                 # regularization penalty
        'C': [0.1],                 # [0.01, 0.1, 1, 10, 100],      # regularization strength
        'solver': ['liblinear'],    # ['liblinear'],                # optimization algorithm
        'max_iter': [100]},         # [100, 500, 1000]},            # maximum number of iterations

    'mnb': {
        'alpha': [0.5],         # [0.1, 0.5, 1.0, 2.0],
        'fit_prior': [True]},   # [True, False]},

    'pac': {
        'C': [0.5],                 # [0.1, 0.5, 1.0, 2.0],
        'max_iter': [2000],         # [1000, 2000, 5000],
        'tol': [1e-3],              # [1e-3, 1e-4, 1e-5],
        'early_stopping': [True]},   # [True, False]},

    'per': {
        'penalty': ['l2'],      # [None, 'l2', 'l1', 'elasticnet'],
        'alpha': [0.0001],      # [0.0001, 0.001, 0.01],
        'max_iter': [1000],     # [1000, 2000, 5000],
        'tol': [1e-3]},         # [1e-3, 1e-4, 1e-5]},

    'qda': {
        'reg_param': [1.0],         # [0.0, 0.1, 0.5, 1.0],
        'store_covariance': [True], # [True, False],
        'tol': [1e-3]},             # [1e-3, 1e-4, 1e-5]},

    'rf': {
        'n_estimators': [50],       # [50, 100, 200],
        'criterion': ['gini'],      # ['gini', 'entropy'],
        'max_depth': [10],          # [None, 5, 10, 15],
        'min_samples_split': [2],   # [2, 5, 10],
        'min_samples_leaf': [1],    # [1, 2, 4],
        'max_features': ['sqrt']},  # [None, 'sqrt', 'log2']},

    'rc': {
        'alpha': [0.01],        # [0.01, 0.1, 1.0, 10.0],
        'solver': ['saga'],     # ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
        'tol': [0.1]},          # [0.0001, 0.001, 0.01, 0.1]},

    'sgd': {
        'loss': ['log_loss'],       # ['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron'],
        'penalty': ['elasticnet'],  # ['l2', 'l1', 'elasticnet'],
        'alpha': [0.0001],          # [0.0001, 0.001, 0.01, 0.1, 1],
        'max_iter': [15000],        # [1000, 5000, 10000],
        'learning_rate': ['optimal'] # ['constant', 'optimal', 'invscaling', 'adaptive']
    },

    'svm': {
        'C': [1],               # [0.1, 1, 10, 100],
        'kernel': ['linear'],   # ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['auto']       # ['scale', 'auto']
    },

    'xgb': {
        'n_estimators': [50],       # [50, 100, 150],
        'max_depth': [4],           # [3, 4, 5],
        'learning_rate': [0.05],    # [0.01, 0.05, 0.1],
        'subsample': [0.7],         # [0.5, 0.7, 1],
        'colsample_bytree': [0.7],  # [0.5, 0.7, 1],
        'gamma': [0],               # [0, 0.1, 0.2],
        'reg_lambda': [0, 1]}       # [0, 1, 10]}
}



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('\n' + datetime.datetime.now().strftime('Experiment started at:  %d-%m-%y  |  %H:%M'))

    experiment = experiment.Experiment(settings, datasets, methodologies, thresholding, evaluators, hyperparameters)
    experiment.run(directory=DIR)

    now = datetime.datetime.now()
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

    directory = str(DIR + '\summary_' + date_time + '.txt')

    # Create txt file for summary of results
    with open(directory, 'w') as file:
        file.write(str(datetime.datetime.now().strftime('Experiment done at:  %d-%m-%y  |  %H:%M') + '\n'))
        file.write('\nSettings: ')
        file.write(json.dumps(settings, indent=3))
        file.write('\nDatasets: ')
        file.write(json.dumps(datasets, indent=3))
        file.write('\nMethodologies: ')
        file.write(json.dumps(methodologies, indent=3))
        file.write('\nThresholding: ')
        file.write(json.dumps(thresholding, indent=3))
        file.write('\nEvaluators: ')
        file.write(json.dumps(evaluators, indent=3))
        file.write('\nHyperparameters: ')

        # hyperara_dict: only contains hyperparameters for the methods that are set to True
        hyperpara_dict = {}
        for key, value in methodologies.items():
            if value:  # if the value is True (literally)
                if key in hyperparameters:  # if the key exists in the nested dictionary
                    hyperpara_dict[key] = hyperparameters[key]
        file.write(json.dumps(hyperpara_dict, indent=3))
        #file.write(json.dumps(hyperparameters, indent=3)) #Todo: line can be removed (rendundant with code above)

        file.write('\n\n_____________________________________________________________________\n\n')

    experiment.evaluate(directory=directory)
