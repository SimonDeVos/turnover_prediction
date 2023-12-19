
import datetime
from experiments import experiment
import json

# Set project directory:
DIR = r'C:\Users\u0148775\PycharmProjects\turnover_prediction\workdir\_results20231219' #todo: anonymize

# Specify experiment configuration
settings = {
    'folds': 2,                 # 2
    'repeats': 1,               # 5
    'cat_encoder': 1,           # 1: one-hot encoding; 2: WoE
    'CV_hyperpara_tuning': 2,   # CV folds (internal) for hyperpara tuning
    'oversampling': 0,          # 0: none; 1: SMOTE; 2: ROS; 3: ADASYN
    'feature_selection': 1,     # 0: none; 1: selectkbest
    'nr_features': 2,           # Amount of features selected
    'setting': "Label123"       # Add label to output file
}

datasets = {
    'real1': False,     # Not publicly available
    'real2': False,     # Not publicly available
    'ds': False,
    'ibm': True,
    'real3': False,     # Not publicly available
    'kaggle1': False,
    'kaggle2': False,   # Not used in the paper
    'kaggle3': False,   # this is kaggle2 in the paper
    'kaggle4': False,   # this is kaggle3 in the paper
    'kaggle5': False,   # this is kaggle4 in the paper
    'kaggle7': False,   # Not used in the paper
    'medium': False,    # Not used in the paper
    'rhuebner': False,  # Not used in the paper
    'techco': False     # Not used in the paper
    }

methodologies = {
    'ab': True,    # AdaBoost (AB)
    'ann': True,   # Artificial Neural Networks (ANN)
    'bnb': True,   # Bernoulli Naive Bayes (BNB)
    'cb': False,   # CatBoost (CB)
    'dt': True,    # Decision Tree (DT)
    'gnb': True,   # Gaussian Naive Bayes (GNB)
    'gb': True,    # Gradient Boosting (GB)
    'knn': True,   # K-Nearest Neighbors (KNN)
    'lgbm': True,  # LightGBM (LGBM)
    'lda': True,   # Linear Discriminant Analysis (LDA)
    'lr': True,    # Logistic Regression (LR)
    'mnb': False,  # Multinomial Naive Bayes (MNB)
    'pac': False,  # Passive Aggressive Classifier (PAC)
    'per': False,  # Perceptron (Per)
    'qda': True,   # Quadratic Discriminant Analysis (QDA)
    'rf': True,    # Random Forest (RF)
    'rc': False,   # Ridge Classifier (RC)
    'sgd': False,  # Stochastic Gradient Descent (SGD)
    'svm': True,   # Support Vector Machine (SVM)
    'xgb': True    # Extreme Gradient Boosting (XGBoost)
}

thresholding = {
    't_idcs': False,        # Instance-dependent cost-sensitive threshold
    't_idcs_cal': False,    # Instance-dependent cost-sensitive threshold with calibrated probabilities
    't_cdcs': False,        # Class-dependent cost-sensitive threshold
    't_cdcs_cal': False,    # Class-dependent cost-sensitive threshold with calibrated probabilities
    't_class_imb': False,   # Class imbalance Threshold
    't_ins': True           # Class insensitive threshold (0.5)
}

evaluators = {
    # Cost-insensitive
    'traditional': False,   # acc, F1, prec, recall
    'Accuracy':True,
    'Recall':True,
    'Precision':True,
    'F1':True,
    'Specificity':True,
    'AUC-PR': True,
    'AUC-ROC': True,
    'PR': False,            # Redundant, included in 'traditional'
    'H_measure': True,
    'brier': True,
    'recall_overlap': False,        # not implemented
    'recall_correlation': False,    # not implemented

    # Cost-sensitive
    'savings': False,
    'AEC': False,
    'ROCIV': False,         # not implemented
    'PRIV': False,          # not implemented
    'rankings': False,      # not implemented

    # Other
    'time': True,

    # Stat testing
    'stat_hypothesis_testing': False # Requires >=3 observations/runs. Perform tests on H0: Model performance follows the same distribution (Friedman test). Needs post-hoc test for pairwise comparison.
}

hyperparameters = {
    'ab': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1],
        'algorithm': ['SAMME', 'SAMME.R']
    },

    'ann': {
        'solver': ['adam', 'lbfgs'],
        'alpha': [0.01], # [0.01, 0.1, 1],
        'hidden_layer_sizes': [(10,10), (20,20), (50,50), (100,100), (200,200)], #Change this to test for other n_hidden_layers
        'activation': ['relu', 'logistic'],
        'max_iter': [100, 200, 500, 1000]
    },

    'bnb': {
        'alpha': [0.01, 0.1, 1.0],
        'fit_prior': [True, False]
    },

    'cb': {
#        'iterations': [50, 100, 200],
        'depth': [5, 10, 15],
#        'learning_rate': [0.1],#[0.01, 0.1, 1],
#        'l2_leaf_reg': [0.1, 0.5, 1.0],
#        'border_count': [32, 64, 128],
#        'bagging_temperature': [0, 1, 5, 10]
    },

    'dt': {
        'criterion': ['gini', 'entropy'],
        'max_depth': [5, 10, 20, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [2, 5, 10],
        },

    'gb': {
        'n_estimators': [20, 50, 100, 200],
        'learning_rate': [0.1],#[0.01, 0.1, 1],
        'max_depth': [5, 10, 20, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [2, 5, 10],
#        'max_features': ['sqrt', 'log2']
    },

    'gnb': {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]},

    'knn': {
        'n_neighbors': [1, 5, 10], # 20, 50],
        'weights': ['uniform', 'distance'],
 #       'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [10],# 30, 50],
        'p': [1, 2]
        },

    'lda': {
#        'solver': ['lsqr'],                  #['svd','lsqr', 'eigen'],   #Caution: certain combinations of hyperparas are not supported - might give Warnings
        'shrinkage': [None, 'auto', 0.1, 0.5, 1.0],
        'n_components': [None, 2, 5, 10]
        },

    'lgbm': {
        'learning_rate': [0.01, 0.1, 1],
        'n_estimators':[20, 50, 100, 200],
        'max_depth': [5, 10, 20, 50],
#        'num_leaves':  [15, 31, 63],
#        'subsample':[0.8, 0.9, 1],
#        'colsample_bytree':  [0.8, 0.9, 1]
        },

    'lr': {
        'penalty': ['none','l1', 'l2'],
        'C':  [0.001, 0.01, 0.1, 1],# [0.01, 0.1, 1, 10],
#        'solver':  ['liblinear'],
        'max_iter': [50, 100, 500, 1000]
        },

    'mnb': {
        'alpha':  [0.1, 0.5, 1.0, 2.0],
        'fit_prior': [True, False]
        },

    'pac': {
        'C':  [0.1, 0.5, 1.0, 2.0],
        'max_iter': [1000, 2000, 5000],
        'tol': [1e-3, 1e-4, 1e-5],
        'early_stopping': [True, False]},

    'per': {
        'penalty': [None, 'l2', 'l1', 'elasticnet'],
        'alpha':  [0.0001, 0.001, 0.01],
        'max_iter':  [1000, 2000, 5000],
        'tol':  [1e-3, 1e-4, 1e-5]},

    'qda': {
        'reg_param':  [0.0, 0.001, 0.1, 0.5, 1.0],
        'store_covariance': [True, False],
        'tol':  [1e-3, 1e-4, 1e-5]},

    'rf': {
        'n_estimators': [20, 50, 100, 200],
        'criterion': ['entropy'], #['gini', 'entropy'],
        'max_depth': [5, 10, 15],
#        'min_samples_split': [2, 10],
#        'min_samples_leaf': [1, 5, 10],
#        'max_features': [None, 'sqrt', 'log2']
        },

    'rc': {
        'alpha': [10.0],
        'solver':  ['sag'],
        'tol': [0.1]},

    'sgd': {
        'loss':  ['hinge', 'log_loss', 'modified_huber', 'perceptron'],
        'penalty':  ['l2', 'l1', 'elasticnet'],
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
        'max_iter':  [1000, 5000, 10000],
        'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive']
    },

    'svm': {
        'C': [0.01, 0.1, 1, 10],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma':['scale', 'auto']
    },

    'xgb': {
        'n_estimators':[50, 100, 200],
        'max_depth': [5, 10, 20, 50],
        'learning_rate': [0.01, 0.1, 1],
#        'subsample': [0.5, 0.7, 1],
#        'colsample_bytree': [0.5, 0.7, 1],
#        'gamma': [0, 0.1, 0.2],
#        'reg_lambda': [0, 1]
    },
}

if __name__ == '__main__':

    print('\n' + datetime.datetime.now().strftime('Experiment started at:  %d-%m-%y  |  %H:%M'))

    experiment_obj = experiment.Experiment(settings, datasets, methodologies, thresholding, evaluators, hyperparameters)
    experiment_obj.run(directory=DIR)

    now = datetime.datetime.now()
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

    dataset_name = next((key for key, value in datasets.items() if value), None)
    directory = str(DIR + '\summary_' + dataset_name + '_' + str(settings['setting']) + '_' + date_time + '.txt')

    # Create output file (.txt) for summary of results
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

        file.write('\n\n_____________________________________________________________________\n\n')

    experiment_obj.evaluate(directory=directory)

print("All experiments completed.")
