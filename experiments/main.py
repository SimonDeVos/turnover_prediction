#Overview to configure experiments.
#Run main.py to launch
import os
import json
import datetime
from experiments import experiment

# Set project directory:
DIR = r'C:\Users\u0148775\PycharmProjects\turnover_prediction\workdir'


#import torch
#print('CUDA available?')
#print(torch.cuda.is_available())
#if torch.cuda.is_available():
#    print(torch.cuda.get_device_name(0))

# Specify experimental configuration
"""
settings
datasets
methodologies
thresholding
evaluators
"""

settings = {
    'class_costs': False,
    'folds': 5,
    'repeats': 2,
    'val_ratio': 0.25,  # Relative to training set only (excluding test set)
    'l1_regularization': False,
    'lambda1_options': [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    'l2_regularization': False,
    'lambda2_options': [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    'neurons_options': [2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10]
}

datasets = {
    'acerta': False,
    'babushkin': False,
    'eds': False,
    'ibm': True,
    'imec': False
}

methodologies = {
    'ab': False,    # AdaBoost (AB)
    'ann': False,   # Artificial Neural Networks (ANN) - sklearn.neural_network.MLPClassifier
    'bnb': False,   # Bernoulli Naive Bayes (BNB) - sklearn.naive_bayes.BernoulliNB
    'cb': False,    # CatBoost (CB) - catboost.CatBoostClassifier
    'dt': False,    # Decision Tree (DT) - sklearn.tree.DecisionTreeClassifier
    'gnb': False,   # Gaussian Naive Bayes (GNB) - sklearn.naive_bayes.GaussianNB
    'gb': False,    # Gradient Boosting (GB) - sklearn.ensemble.GradientBoostingClassifier
    'knn': False,   # K-Nearest Neighbors (KNN) - sklearn.neighbors.KNeighborsClassifier
    'lgbm': False,  # LightGBM (LGBM) - lightgbm.LGBMClassifier
    'lda': False,   # Linear Discriminant Analysis (LDA) - sklearn.discriminant_analysis.LinearDiscriminantAnalysis
    'lr': True,    # Logistic Regression (LR) - sklearn.linear_model.LogisticRegression
    'mnb': False,   # Multinomial Naive Bayes (MNB) - sklearn.naive_bayes.MultinomialNB
    'pac': False,   # Passive Aggressive Classifier (PAC) - sklearn.linear_model.PassiveAggressiveClassifier
    'p': False,     # Perceptron (P) - sklearn.linear_model.Perceptron
    'qda': False,   # Quadratic Discriminant Analysis (QDA) - sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis
    'rf': False,    # Random Forest (RF) - sklearn.ensemble.RandomForestClassifier
    'rc': False,    # Ridge Classifier (RC) - sklearn.linear_model.RidgeClassifier
    'sgd': False,   # Stochastic Gradient Descent (SGD) - sklearn.linear_model.SGDClassifier
    'svm': False,   # Support Vector Machine (SVM) - sklearn.svm.SVC
    'xgb': False  # Extreme Gradient Boosting (XGBoost) - xgboost.XGBClassifier
}

thresholding = {
    't_idcs': False,    # Instance-dependent cost-sensitive threshold
    't_idcs_cal': False,# Instance-dependent cost-sensitive threshold with calibrated probabilities
    't_cdcs': False,    # Class-dependent cost-sensitive threshold
    't_cdcs_cal': False,# Class-dependent cost-sensitive threshold with calibrated probabilities
    't_class_imb': False,# Class imbalance Threshold
    't_ins': True      # Class insensitive threshold (0.5)
}

evaluators = {
    # Cost-insensitive
    'traditional': True,
    'ROC': True,
    'AUC': True,
    'PR': True,
    'H_measure': True,
    'brier': True,
    'recall_overlap': True,
    'recall_correlation': True,

    # Cost-sensitive
    'savings': True,
    'AEC': True,
    'ROCIV': True,
    'PRIV': True,
    'rankings': True,

    # Other
    'time': True
}


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('\n' + datetime.datetime.now().strftime('Experiment started at:  %d-%m-%y  |  %H:%M'))

    experiment = experiment.Experiment(settings, datasets, methodologies, thresholding, evaluators)
    experiment.run(directory=DIR)

    # Create txt file for summary of results
    with open(str(DIR + 'summary.txt'), 'w') as file:
        file.write(str(datetime.datetime.now().strftime('Experiment done at:  %d-%m-%y  |  %H:%M') + '\n'))
        file.write('\nSettings: ')
        file.write(json.dumps(settings, indent=3))
        file.write('\nDatasets: ')
        file.write(json.dumps(datasets, indent=3))
        file.write('\nMethodologies: ')
        file.write(json.dumps(methodologies, indent=3))
        file.write('\nEvaluators: ')
        file.write(json.dumps(evaluators, indent=3))
        file.write('\n\n_____________________________________________________________________\n\n')

    experiment.evaluate(directory=DIR)
