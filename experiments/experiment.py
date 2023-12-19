
import numpy as np
import time

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, Perceptron, RidgeClassifier, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler

import xgboost
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN

from performance_metrics.performance_metrics import get_performance_metrics, evaluate_experiments
from preprocessing.preprocessing import preprocess_ibm, handle_missing_data, convert_categorical_variables, standardize, feature_select, \
    preprocess_kaggle1, preprocess_kaggle2, preprocess_kaggle3, preprocess_kaggle4, preprocess_kaggle5, \
    preprocess_kaggle6, preprocess_kaggle7, preprocess_ds, preprocess_medium, preprocess_rhuebner, \
    preprocess_techco, add_prefix_to_hyperparams

from imblearn.pipeline import make_pipeline

import warnings

# This code is not publicly available:
# from private_code.preprocessing_private import preprocess_real1, preprocess_real2, preprocess_real3

warnings.filterwarnings('ignore')

class Experiment:
    def __init__(self, settings, datasets, methodologies, thresholding, evaluators, hyperparameters):

        self.settings = settings
        self.datasets = datasets
        if sum(self.datasets.values()) != 1:
            raise ValueError('Select only one dataset!')

        self.methodologies = methodologies
        self.thresholding = thresholding
        if sum(self.thresholding.values()) != 1:
            raise ValueError('Select only one thresholding strategy!')

        self.hyperparameters = hyperparameters
        self.evaluators = evaluators
        self.results = {}

    def run(self, directory):
        """
        LOAD AND PREPROCESS DATA
        """

        print('\n\n************** LOADING DATA **************\n')

        if self.datasets['ibm']:
            print('ibm')
            covariates, labels, amounts, cost_matrix, categorical_variables = preprocess_ibm()
#        elif self.datasets['real1']:
#            print('real1')
#            covariates, labels, amounts, cost_matrix, categorical_variables = preprocess_real1() # Not publicly available
#        elif self.datasets['real2']:
#            print('real2')
#            covariates, labels, amounts, cost_matrix, categorical_variables = preprocess_real2() # Not publicly available
        elif self.datasets['ds']:
            print('data scientists dataset')
            covariates, labels, amounts, cost_matrix, categorical_variables = preprocess_ds()
#        elif self.datasets['real3']:
#            covariates, labels, amounts, cost_matrix, categorical_variables = preprocess_real3() # Not publicly available
        elif self.datasets['kaggle1']:
            print('kaggle1')
            covariates, labels, amounts, cost_matrix, categorical_variables = preprocess_kaggle1()
        elif self.datasets['kaggle2']:
            print('kaggle2')
            covariates, labels, amounts, cost_matrix, categorical_variables = preprocess_kaggle2()
        elif self.datasets['kaggle3']:
            print('kaggle3')
            covariates, labels, amounts, cost_matrix, categorical_variables = preprocess_kaggle3()
        elif self.datasets['kaggle4']:
            print('kaggle4')
            covariates, labels, amounts, cost_matrix, categorical_variables = preprocess_kaggle4()
        elif self.datasets['kaggle5']:
            print('kaggle5')
            covariates, labels, amounts, cost_matrix, categorical_variables = preprocess_kaggle5()

        elif self.datasets['kaggle7']:
            print('kaggle7')
            covariates, labels, amounts, cost_matrix, categorical_variables = preprocess_kaggle7()

        elif self.datasets['medium']:
            print('medium dataset')
            covariates, labels, amounts, cost_matrix, categorical_variables = preprocess_medium()
        elif self.datasets['rhuebner']:
            print('rhuebner dataset')
            covariates, labels, amounts, cost_matrix, categorical_variables = preprocess_rhuebner()
        elif self.datasets['techco']:
            print('techco dataset')
            covariates, labels, amounts, cost_matrix, categorical_variables = preprocess_techco()
        else:
            raise Exception('No dataset specified')

        # print properties of dataset:
        print('\tnr of features: '+str(covariates.shape[1]) +' +1')
        print('\tnr of observations: ' + str(covariates.shape[0]))
        imbalance_ratio = (labels == 1).sum() / covariates.shape[0]
        print(f"\t{imbalance_ratio*100:.2f}% of observations are churn")

        """
        RUN EXPERIMENTS
        """

        print('\n\n***** BUILDING CLASSIFICATION MODELS *****')

        # Prepare the cross-validation procedure
        folds = self.settings['folds']
        repeats = self.settings['repeats']
        rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=42)
        prepr = labels

        # Prepare the evaluation matrices
        n_methodologies = sum(self.methodologies.values())
        for key in self.evaluators.keys():
            if self.evaluators[key]:
                self.results[key] = np.empty(shape=(n_methodologies, folds * repeats), dtype='object')

        for i, (train_val_index, test_index) in enumerate(rskf.split(covariates, prepr)):
            print('\nCross validation: ' + str(i + 1))

            index = 0

            x_train_val, x_test = covariates.iloc[train_val_index], covariates.iloc[test_index]
            y_train_val, y_test = labels[train_val_index], labels[test_index]
            amounts_train_val, amounts_test = amounts[train_val_index], amounts[test_index]
            cost_matrix_train_val, cost_matrix_test = cost_matrix[train_val_index, :], cost_matrix[test_index, :]

            # Split training and validation set
            train_ratio = 1 - (1/self.settings['CV_hyperpara_tuning'])
            skf = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio, random_state=42)
            prepr_val = y_train_val

            for train_index, val_index in skf.split(x_train_val, prepr_val):
                x_train, x_val = x_train_val.iloc[train_index], x_train_val.iloc[val_index]
                y_train, y_val = y_train_val[train_index], y_train_val[val_index]
                cost_matrix_train, cost_matrix_val = cost_matrix_train_val[train_index, :], cost_matrix_train_val[val_index, :]

            # Preprocessing: Handle missing data, convert categorical variables, standardize, convert to numpy
            x_train, x_val, x_test, categorical_variables = handle_missing_data(x_train, x_val, x_test, categorical_variables) #TODO: debug
            cat_encoder = self.settings['cat_encoder']
            x_train, x_val, x_test = convert_categorical_variables(x_train, y_train, x_val, x_test, categorical_variables, cat_encoder)
            x_train, x_val, x_test = standardize(x_train=x_train, x_val=x_val, x_test=x_test)
            feat_selector = self.settings['feature_selection']
            x_train, x_val, x_test = feature_select(x_train, y_train, x_val, x_test,  feat_selector, self.settings['nr_features'])

            #After preprocessing properly, concatenate train and val into train_val
            x_train_val = np.concatenate((x_train, x_val))
            y_train_val = np.concatenate((y_train, y_val))
            """Assign thresholds for the different strategies:"""
            #   Instance-dependent cost-sensitive threshold
            if self.thresholding['t_idcs']:
                threshold = (cost_matrix_test[:, 1, 0] - cost_matrix_test[:, 0, 0]) / (cost_matrix_test[:, 1, 0] - cost_matrix_test[:, 0, 0] + cost_matrix_test[:, 0,1] - cost_matrix_test[:, 1, 1])

            #   Instance-dependent cost-sensitive threshold, calibrated
            if self.thresholding['t_idcs_cal']: #TODO
                raise Exception('t_idcs_cal not implemented yet')

            #   Class-dependent cost-sensitive threshold
            if self.thresholding['t_cdcs']:
                threshold_class = (cost_matrix_test[:, 1, 0].mean() - cost_matrix_test[:, 0, 0].mean()) / (cost_matrix_test[:, 1, 0].mean() - cost_matrix_test[:, 0, 0].mean() + cost_matrix_test[:, 0, 1].mean() - cost_matrix_test[:, 1, 1].mean())
                threshold = np.repeat(threshold_class, len(y_test))

            #   Class-dependent cost-sensitive threshold, calibrated
            if self.thresholding['t_cdcs_cal']: #TODO
                raise Exception('t_cdcs_cal not implemented yet')

            #   Class imbalance threshold
            if self.thresholding['t_class_imb']:
                threshold = y_train.mean()

            #   Cost-insensitive threshold
            if self.thresholding['t_ins']:
                threshold = np.repeat(0.5, len(y_test))

            # Define evaluation procedure for different thresholding strategies
            def evaluate_model(proba_val, proba_test, i, index, info):

                #round proba to predictions {0,1}, dependent on defined threshold.
                pred_test = (proba_test > threshold).astype(int)
#                self.results_tr_instance = get_performance_metrics(self.evaluators, self.results_tr_instance, j, index, cost_matrix_test, y_test, proba, pred, info)
                self.results = get_performance_metrics(self.evaluators, self.results, i, index, cost_matrix_test, y_test, proba_test, pred_test, info)

            def create_pipeline_and_param_grid(self, param_grid, prefix: str, method):
                if self.settings['oversampling'] == 0:
                    pipeline = method
                elif self.settings['oversampling'] == 1 or self.settings['oversampling'] == 2 or self.settings['oversampling'] == 3:
                    oversampler = {
                        1: SMOTE(random_state=0),
                        2: RandomOverSampler(random_state=0),
                        3: ADASYN(random_state=0)
                    }[self.settings['oversampling']]
                    pipeline = make_pipeline(oversampler, method)
                    param_grid = add_prefix_to_hyperparams(param_grid, prefix)

                return pipeline, param_grid

            cv_folds = self.settings['CV_hyperpara_tuning']

            # AdaBoost
            if self.methodologies['ab']:
                param_grid = self.hyperparameters['ab']
                method = AdaBoostClassifier()
                pipeline, param_grid = create_pipeline_and_param_grid(self, param_grid, 'adaboostclassifier__', method)

                # Create a GridSearchCV object
                gs = GridSearchCV(pipeline, param_grid=param_grid, scoring='accuracy', cv=cv_folds)
                #gs = RandomizedSearchCV(pipeline, param_distributions=param_grid, scoring='accuracy', cv=cv_folds, n_iter=5)
                start = time.perf_counter()

                gs.fit(x_train_val, y_train_val)
                end = time.perf_counter()

                print('\tab - best hyperparameters:', gs.best_params_)


                proba_test = gs.predict_proba(x_test)[:, 1]
                proba_val = gs.predict_proba(x_val)[:, 1]

                info = {'time': end - start}

                evaluate_model(proba_val, proba_test, i, index, info)

                index += 1

            # Artifical Neural Network
            if self.methodologies['ann']:
                param_grid = self.hyperparameters['ann']
                method = MLPClassifier()
                pipeline, param_grid = create_pipeline_and_param_grid(self, param_grid, "mlpclassifier__", method)

                gs = GridSearchCV(pipeline, param_grid=param_grid, scoring='accuracy', cv=cv_folds)
                #gs = RandomizedSearchCV(pipeline, param_distributions=param_grid, scoring='accuracy', cv=cv_folds, n_iter=5)

                start = time.perf_counter()
                gs.fit(x_train_val, y_train_val)
                end = time.perf_counter()

                print('\tann - best hyperparameters:', gs.best_params_)

                proba_test = gs.predict_proba(x_test)[:, 1]
                proba_val = gs.predict_proba(x_val)[:, 1]

                info = {'time': end - start}

                evaluate_model(proba_val, proba_test, i, index, info)

                index += 1

            # Bernoulli Naive Bayes
            if self.methodologies['bnb']:
                param_grid = self.hyperparameters['bnb']
                method = BernoulliNB()
                pipeline, param_grid = create_pipeline_and_param_grid(self, param_grid, "bernoullinb__", method)

                gs = GridSearchCV(pipeline, param_grid=param_grid, scoring='accuracy', cv=cv_folds)
                #gs = RandomizedSearchCV(pipeline, param_distributions=param_grid, scoring='accuracy', cv=cv_folds, n_iter=5)

                start = time.perf_counter()
                gs.fit(x_train_val, y_train_val)
                end = time.perf_counter()

                print('\tbnb - best hyperparameters:', gs.best_params_)

                proba_test = gs.predict_proba(x_test)[:, 1]
                proba_val = gs.predict_proba(x_val)[:, 1]

                info = {'time': end - start}

                evaluate_model(proba_val, proba_test, i, index, info)

                index += 1

            # CatBoost
            if self.methodologies['cb']:
                method = CatBoostClassifier(silent=True)
                param_grid = self.hyperparameters['cb']

                pipeline, param_grid = create_pipeline_and_param_grid(self, param_grid, "catboostclassifier__", method)

                gs = GridSearchCV(pipeline, param_grid=param_grid, scoring='accuracy', cv=cv_folds)
                #gs = RandomizedSearchCV(pipeline, param_distributions=param_grid, scoring='accuracy', cv=cv_folds, n_iter=5)

                start = time.perf_counter()
                gs.fit(x_train_val, y_train_val)
                end = time.perf_counter()

                print('\tcb - best hyperparameters:', gs.best_params_)

                proba_test = gs.predict_proba(x_test)[:, 1]
                proba_val = gs.predict_proba(x_val)[:, 1]

                info = {'time': end - start}

                evaluate_model(proba_val, proba_test, i, index, info)

                index += 1

            # Decision Tree
            if self.methodologies['dt']:
                method = DecisionTreeClassifier()
                param_grid = self.hyperparameters['dt']

                pipeline, param_grid = create_pipeline_and_param_grid(self, param_grid, "decisiontreeclassifier__", method)

                gs = GridSearchCV(pipeline, param_grid=param_grid, scoring='accuracy', cv=cv_folds)
                #gs = RandomizedSearchCV(pipeline, param_distributions=param_grid, scoring='accuracy', cv=cv_folds, n_iter=5)

                start = time.perf_counter()
                gs.fit(x_train_val, y_train_val)
                end = time.perf_counter()

                print('\tdt - best hyperparameters:', gs.best_params_)

                proba_test = gs.predict_proba(x_test)[:, 1]
                proba_val = gs.predict_proba(x_val)[:, 1]

                info = {'time': end - start}

                evaluate_model(proba_val, proba_test, i, index, info)

                index += 1

            # Gaussian Naive Bayes
            if self.methodologies['gnb']:
                method = GaussianNB()
                param_grid = self.hyperparameters['gnb']

                pipeline, param_grid = create_pipeline_and_param_grid(self, param_grid, "gaussiannb__", method)

                gs = GridSearchCV(pipeline, param_grid=param_grid, scoring='accuracy', cv=cv_folds)
                #gs = RandomizedSearchCV(pipeline, param_distributions=param_grid, scoring='accuracy', cv=cv_folds, n_iter=5)

                start = time.perf_counter()
                gs.fit(x_train_val, y_train_val)
                end = time.perf_counter()

                print('\tgnb - best hyperparameters:', gs.best_params_)

                proba_test = gs.predict_proba(x_test)[:, 1]
                proba_val = gs.predict_proba(x_val)[:, 1]

                info = {'time': end - start}

                evaluate_model(proba_val, proba_test, i, index, info)

                index += 1

            # Gradient Goosting
            if self.methodologies['gb']:
                method = GradientBoostingClassifier()
                param_grid = self.hyperparameters['gb']
                pipeline, param_grid = create_pipeline_and_param_grid(self, param_grid, "gradientboostingclassifier__", method)

                gs = GridSearchCV(pipeline, param_grid=param_grid, scoring='accuracy', cv=cv_folds)
                #gs = RandomizedSearchCV(pipeline, param_distributions=param_grid, scoring='accuracy', cv=cv_folds, n_iter=5)

                start = time.perf_counter()
                gs.fit(x_train_val, y_train_val)
                end = time.perf_counter()

                print('\tgb - best hyperparameters:', gs.best_params_)

                proba_test = gs.predict_proba(x_test)[:, 1]
                proba_val = gs.predict_proba(x_val)[:, 1]

                info = {'time': end - start}

                evaluate_model(proba_val, proba_test, i, index, info)

                index += 1

            # K-Nearest Neighbors
            if self.methodologies['knn']:
                method = KNeighborsClassifier()
                param_grid = self.hyperparameters['knn']
                pipeline, param_grid = create_pipeline_and_param_grid(self, param_grid, "kneighborsclassifier__", method)

                gs = GridSearchCV(pipeline, param_grid=param_grid, scoring='accuracy', cv=cv_folds)
                #gs = RandomizedSearchCV(pipeline, param_distributions=param_grid, scoring='accuracy', cv=cv_folds, n_iter=5)

                start = time.perf_counter()
                gs.fit(x_train_val, y_train_val)
                end = time.perf_counter()

                print('\tknn - best hyperparameters:', gs.best_params_)

                proba_test = gs.predict_proba(x_test)[:, 1]
                proba_val = gs.predict_proba(x_val)[:, 1]

                info = {'time': end - start}

                evaluate_model(proba_val, proba_test, i, index, info)

                index += 1

            # LightGBM
            if self.methodologies['lgbm']:
                method = LGBMClassifier(random_state=42)
                param_grid = self.hyperparameters['lgbm']
                pipeline, param_grid = create_pipeline_and_param_grid(self, param_grid, "lgbmclassifier__", method)

                gs = GridSearchCV(pipeline, param_grid=param_grid, scoring='accuracy', cv=cv_folds)
                #gs = RandomizedSearchCV(pipeline, param_distributions=param_grid, scoring='accuracy', cv=cv_folds, n_iter=5)

                start = time.perf_counter()
                gs.fit(x_train_val, y_train_val)
                end = time.perf_counter()

                print('\tlgbm - best hyperparameters:', gs.best_params_)

                proba_test = gs.predict_proba(x_test)[:, 1]
                proba_val = gs.predict_proba(x_val)[:, 1]

                info = {'time': end - start}

                evaluate_model(proba_val, proba_test, i, index, info)

                index += 1

            # Linear Discriminant Analysis
            if self.methodologies['lda']:
                method = LinearDiscriminantAnalysis()
                param_grid = self.hyperparameters['lda']
                pipeline, param_grid = create_pipeline_and_param_grid(self, param_grid, "lineardiscriminantanalysis__", method)

                gs = GridSearchCV(pipeline, param_grid=param_grid, scoring='accuracy', cv=cv_folds)
                #gs = RandomizedSearchCV(pipeline, param_distributions=param_grid, scoring='accuracy', cv=cv_folds, n_iter=5)

                start = time.perf_counter()
                gs.fit(x_train_val, y_train_val)
                end = time.perf_counter()

                print('\tlda - best hyperparameters:', gs.best_params_)

                proba_test = gs.predict_proba(x_test)[:, 1]
                proba_val = gs.predict_proba(x_val)[:, 1]

                info = {'time': end - start}

                evaluate_model(proba_val, proba_test, i, index, info)

                index += 1

            # Logistic regression
            if self.methodologies['lr']:
                method = LogisticRegression()
                param_grid = self.hyperparameters['lr']
                pipeline, param_grid = create_pipeline_and_param_grid(self, param_grid, "logisticregression__", method)

                gs = GridSearchCV(pipeline, param_grid=param_grid, scoring='accuracy', cv=cv_folds)
                #gs = RandomizedSearchCV(pipeline, param_distributions=param_grid, scoring='accuracy', cv=cv_folds, n_iter=5)

                # Fitting the grid search object to the train_val data
                start = time.perf_counter()
                gs.fit(x_train_val, y_train_val)
                end = time.perf_counter()

                print('\tlr - best hyperparameters:', gs.best_params_)

                lr_proba = gs.predict_proba(x_test)[:, 1]
                lr_proba_val = gs.predict_proba(x_val)[:, 1]

                info = {'time': end - start}

                evaluate_model(lr_proba_val, lr_proba, i, index, info)

                index += 1

            # Multinomial Naive Bayes
            if self.methodologies['mnb']:
                method = MultinomialNB()
                param_grid = self.hyperparameters['mnb']
                pipeline, param_grid = create_pipeline_and_param_grid(self, param_grid, "multinomialnb__", method)

                gs = GridSearchCV(pipeline, param_grid=param_grid, scoring='accuracy', cv=cv_folds)
                #gs = RandomizedSearchCV(pipeline, param_distributions=param_grid, scoring='accuracy', cv=cv_folds, n_iter=5)

                scaler = MinMaxScaler()
                x_train_val_non_neg = scaler.fit_transform(x_train_val)
                x_val_non_neg = scaler.fit_transform(x_val)
                x_test_non_neg = scaler.fit_transform(x_test)

                start = time.perf_counter()
                gs.fit(x_train_val_non_neg, y_train_val)
                end = time.perf_counter()

                print('\tmnb - best hyperparameters:', gs.best_params_)

                proba_test = gs.predict_proba(x_test_non_neg)[:, 1]
                proba_val = gs.predict_proba(x_val_non_neg)[:, 1]

                info = {'time': end - start}

                evaluate_model(proba_val, proba_test, i, index, info)

                index += 1

            # Passive Aggressive Classifier
            if self.methodologies['pac']:
                method = PassiveAggressiveClassifier()
                param_grid = self.hyperparameters['pac']
                pipeline, param_grid = create_pipeline_and_param_grid(self, param_grid, "passiveaggressiveclassifier__", method)

                gs = GridSearchCV(pipeline, param_grid=param_grid, scoring='accuracy', cv=cv_folds)
                #gs = RandomizedSearchCV(pipeline, param_distributions=param_grid, scoring='accuracy', cv=cv_folds, n_iter=5)

                start = time.perf_counter()
                gs.fit(x_train_val, y_train_val)
                end = time.perf_counter()

                print('\tpac - best hyperparameters:', gs.best_params_)

                d_test = gs.decision_function(x_test)
                proba_test = np.exp(d_test) / (1 + np.exp(d_test))

                d_val = gs.decision_function(x_val)
                proba_val = np.exp(d_val) / (1 + np.exp(d_val))

                info = {'time': end - start}

                evaluate_model(proba_val, proba_test, i, index, info)

                index += 1

            # Perceptron
            if self.methodologies['per']:
                method = Perceptron()
                param_grid = self.hyperparameters['per']

                pipeline, param_grid = create_pipeline_and_param_grid(self, param_grid, "perceptron__",method)

                gs = GridSearchCV(pipeline, param_grid=param_grid, scoring='accuracy', cv=cv_folds)
                #gs = RandomizedSearchCV(pipeline, param_distributions=param_grid, scoring='accuracy', cv=cv_folds, n_iter=5)

                start = time.perf_counter()
                gs.fit(x_train_val, y_train_val)
                end = time.perf_counter()

                print('\tper - best hyperparameters:', gs.best_params_)

                d_test = gs.decision_function(x_test)
                proba_test = np.exp(d_test) / (1 + np.exp(d_test))
                #predictions contain nan values, so impute
                proba_test[np.isnan(proba_test)] = np.nanmedian(proba_test)

                d_val = gs.decision_function(x_val)
                proba_val = np.exp(d_val) / (1 + np.exp(d_val))
                # predictions contain nan values, so impute
                proba_val[np.isnan(proba_val)] = np.nanmedian(proba_val)

                info = {'time': end - start}

                evaluate_model(proba_val, proba_test, i, index, info)

                index += 1

            # Quadratic Discriminant Analysis
            if self.methodologies['qda']:
                method = QuadraticDiscriminantAnalysis()
                param_grid = self.hyperparameters['qda']
                pipeline, param_grid = create_pipeline_and_param_grid(self, param_grid, "quadraticdiscriminantanalysis__", method)

                gs = GridSearchCV(pipeline, param_grid=param_grid, scoring='accuracy', cv=cv_folds)
                #gs = RandomizedSearchCV(pipeline, param_distributions=param_grid, scoring='accuracy', cv=cv_folds, n_iter=5)

                start = time.perf_counter()
                gs.fit(x_train_val, y_train_val)
                end = time.perf_counter()

                print('\tqda - best hyperparameters:', gs.best_params_)

                proba_test = gs.predict_proba(x_test)[:, 1]
                proba_val = gs.predict_proba(x_val)[:, 1]

                info = {'time': end - start}

                evaluate_model(proba_val, proba_test, i, index, info)

                index += 1

            # Random Forest
            if self.methodologies['rf']:
                method = RandomForestClassifier()
                param_grid = self.hyperparameters['rf']
                pipeline, param_grid = create_pipeline_and_param_grid(self, param_grid, "randomforestclassifier__", method)

                gs = GridSearchCV(pipeline, param_grid=param_grid, scoring='accuracy', cv=cv_folds)
                #gs = RandomizedSearchCV(pipeline, param_distributions=param_grid, scoring='accuracy', cv=cv_folds, n_iter=5)

                start = time.perf_counter()
                gs.fit(x_train_val, y_train_val)
                end = time.perf_counter()

                print('\trf - best hyperparameters:', gs.best_params_)

                proba_test = gs.predict_proba(x_test)[:, 1]
                proba_val = gs.predict_proba(x_val)[:, 1]

                info = {'time': end - start}

                evaluate_model(proba_val, proba_test, i, index, info)

                index += 1

            # Ridge Classifier
            if self.methodologies['rc']:
                method = RidgeClassifier()
                param_grid = self.hyperparameters['rc']
                pipeline, param_grid = create_pipeline_and_param_grid(self, param_grid, "ridgeclassifier__", method)

                gs = GridSearchCV(pipeline, param_grid=param_grid, scoring='accuracy', cv=cv_folds)
                #gs = RandomizedSearchCV(pipeline, param_distributions=param_grid, scoring='accuracy', cv=cv_folds, n_iter=5)

                start = time.perf_counter()
                gs.fit(x_train_val, y_train_val)
                end = time.perf_counter()

                print('\trc - best hyperparameters:', gs.best_params_)

                # decision_function outputs scores between -1 and 1. Transform these to [0,1] with softmax
                d_test = gs.decision_function(x_test)
                proba_test = np.exp(d_test) / (1 + np.exp(d_test))
                d_val = gs.decision_function(x_val)
                proba_val = np.exp(d_val) / (1 + np.exp(d_val))

                info = {'time': end - start}

                evaluate_model(proba_val, proba_test, i, index, info)

                index += 1

            # Stochastic Gradient Descent
            if self.methodologies['sgd']:
                method = SGDClassifier()
                param_grid = self.hyperparameters['sgd']
                pipeline, param_grid = create_pipeline_and_param_grid(self, param_grid, "sgdclassifier__", method)

                gs = GridSearchCV(pipeline, param_grid=param_grid, scoring='accuracy', cv=cv_folds)
                #gs = RandomizedSearchCV(pipeline, param_distributions=param_grid, scoring='accuracy', cv=cv_folds, n_iter=5)

                start = time.perf_counter()
                gs.fit(x_train_val, y_train_val)
                end = time.perf_counter()

                print('\tsgd - best hyperparameters:', gs.best_params_)

                proba_test = gs.predict_proba(x_test)[:, 1]
                proba_val = gs.predict_proba(x_val)[:, 1]

                info = {'time': end - start}

                evaluate_model(proba_val, proba_test, i, index, info)

                index += 1

            # Support Vector Machine
            if self.methodologies['svm']:
                method = SVC()
                param_grid = self.hyperparameters['svm']
                pipeline, param_grid = create_pipeline_and_param_grid(self, param_grid, "svc__", method)

                gs = GridSearchCV(pipeline, param_grid=param_grid, scoring='accuracy', cv=cv_folds)
                #gs = RandomizedSearchCV(pipeline, param_distributions=param_grid, scoring='accuracy', cv=cv_folds, n_iter=5)

                start = time.perf_counter()
                gs.fit(x_train_val, y_train_val)
                end = time.perf_counter()

                print('\tsvm - best hyperparameters:', gs.best_params_)

                proba_test = gs.decision_function(x_test)
                proba_val = gs.decision_function(x_val)

                info = {'time': end - start}

                evaluate_model(proba_val, proba_test, i, index, info)

                index += 1

            #XGBoosting
            if self.methodologies['xgb']:
                method = xgboost.XGBClassifier()
                param_grid = self.hyperparameters['xgb']
                pipeline, param_grid = create_pipeline_and_param_grid(self, param_grid, "xgbclassifier__", method)

                gs = GridSearchCV(pipeline, param_grid=param_grid, scoring='accuracy', cv=cv_folds)
                #gs = RandomizedSearchCV(pipeline, param_distributions=param_grid, scoring='accuracy', cv=cv_folds, n_iter=5)

                # Fitting the grid search object to the train_val data
                start = time.perf_counter()
                gs.fit(x_train_val, y_train_val)
                end = time.perf_counter()

                print('\txgb - best hyperparameters:', gs.best_params_)

                xgb_proba = gs.predict_proba(x_test)[:, 1]
                xgb_proba_val = gs.predict_proba(x_val)[:, 1]

                info = {'time': end - start}

                evaluate_model(xgb_proba_val, xgb_proba, i, index, info)

                index += 1

            print('\n----------------------------------------------------------------')

    def evaluate(self, directory):
        """
        EVALUATION
        """
        print('\n\n********* EVALUATING CLASSIFIERS *********')

        print('\n*** Results ***')
        print('Thresholding method: '+str(self.thresholding)+'\n')

        evaluate_experiments(evaluators=self.evaluators,
                             methodologies=self.methodologies,
                             thresholding=self.thresholding,
                             evaluation_matrices=self.results,
                             settings=self.settings,
                             dataset=self.datasets,
                             directory=directory,
                             name='id')














