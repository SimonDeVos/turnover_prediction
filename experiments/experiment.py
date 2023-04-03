
#TODO add imports and organize them properly
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, Perceptron, RidgeClassifier, \
    SGDClassifier
import xgboost
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
import time
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from performance_metrics.performance_metrics import get_performance_metrics, evaluate_experiments
from preprocessing.preprocessing import preprocess_ibm, handle_missing_data, convert_categorical_variables, standardize, \
    preprocess_kaggle1, preprocess_kaggle2, preprocess_kaggle3, preprocess_kaggle4, preprocess_kaggle5, \
    preprocess_kaggle6, preprocess_kaggle7, preprocess_acerta, preprocess_ds, preprocess_medium, preprocess_rhuebner, \
    preprocess_techco, add_prefix_to_hyperparams
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier

from imblearn.pipeline import Pipeline, make_pipeline

import warnings
warnings.filterwarnings('ignore')


"""Contents:
define class
    def run
        load and preprocess
            with dataset-specific functions
        run experiments
            Build classification models
                prepare CV procedure
                prepare evaluation matrices
                per CV:
                    split in train/val
                    preprocess:
                        handle missing
                        convert cat var
                        standardize
                        convert to np
                    assign thresholds for each strategy
                        t_idcs
                        t_idcs_cal
                        t_cdcs
                        t_cdcs_cal
                        t_class_imb
                        t_ins
                    Define evaluation procedure for different thresholding strategies (def evaluate_model)

                    Run through different methods:
                        ab
                        ann
                        bnb
                        ...
                        lr
                            initialize model
                                hyperpara tuning through CV
                                    per Cross-fold: oversample training set
                                    tune one validation set
                                    return best hyperparas                               
                            train model
                                oversample train_val set
                                use best hyperparas
                            evaluate: 'evaluate_model(proba_val,proba,i,index,info)'
                        ...
    def evaluate
        evaluate_experiments (call: from performance_metrics.performance_metrics import evaluate_experiments)
"""


class Experiment:
    def __init__(self, settings, datasets, methodologies, thresholding, evaluators, hyperparameters):

        self.settings = settings

#        self.l1 = self.settings['l1_regularization']
#        self.lambda1_list = self.settings['lambda1_options']
#        self.l2 = self.settings['l2_regularization']
#        self.lambda2_list = self.settings['lambda2_options']
#        self.neurons_list = self.settings['neurons_options']

#        if self.l1 and self.l2:
#            raise ValueError('Only l1 or l2 regularization allowed, not both!')

        self.datasets = datasets
        # Verify that only one dataset is selected
        if sum(self.datasets.values()) != 1:
            raise ValueError('Select only one dataset!')
        self.methodologies = methodologies
        self.thresholding = thresholding
        if sum(self.thresholding.values()) != 1:
            raise ValueError('Select only one thresholding strategy!')

        self.hyperparameters = hyperparameters

        self.evaluators = evaluators

        self.results = {}   #TODO: others will be redundant (if only one .results is used - only one thresholding strategy at the time)
#        self.results_t_idcs = {}
#        self.results_t_idcs_cal = {}
#        self.results_t_cdcs = {}
#        self.results_t_cdcs_cal = {}
#        self.results_t_class_imb = {}
#        self.results_t_ins = {}


    def run(self, directory):
        """
        LOAD AND PREPROCESS DATA
        """
#        global xgb #TODO: no idea what this line does, currently commented

        print('\n\n************** LOADING DATA **************\n')



        if self.datasets['ibm']:
            print('ibm')
            covariates, labels, amounts, cost_matrix, categorical_variables = preprocess_ibm() #TODO: implement function
        elif self.datasets['acerta']:
            print('acerta')
            covariates, labels, amounts, cost_matrix, categorical_variables = preprocess_acerta() #TODO: implement function
        elif self.datasets['ds']:
            print('data scientists dataset')
            covariates, labels, amounts, cost_matrix, categorical_variables = preprocess_ds() #TODO: implement function
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
        elif self.datasets['kaggle6']:
            print('kaggle6')
            covariates, labels, amounts, cost_matrix, categorical_variables = preprocess_kaggle6()
        elif self.datasets['kaggle7']:
            print('kaggle7')
            covariates, labels, amounts, cost_matrix, categorical_variables = preprocess_kaggle7()
#        elif self.datasets['kaggle8']:
#            print('kaggle8')
#            covariates, labels, amounts, cost_matrix, categorical_variables = preprocess_kaggle8()
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
        print('nr of features: '+str(covariates.shape[1]) +' +1')
        print('nr of observations: ' + str(covariates.shape[0]))
        imbalance_ratio = (labels == 1).sum() / covariates.shape[0]
        print(f"The imbalance ratio is {imbalance_ratio:.2f}")

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
            if self.evaluators[key]: #TODO: add here constraints that matrices are only made if thresholding strategy is selected?
                self.results[key] = np.empty(shape=(n_methodologies, folds * repeats), dtype='object') #TODO: other self.results_t_....lines can be deleted when only one thresholding strat is chosen
#                self.results_t_idcs[key] =      np.empty(shape=(n_methodologies, folds * repeats), dtype='object')
#                self.results_t_idcs_cal[key] =  np.empty(shape=(n_methodologies, folds * repeats), dtype='object')
#                self.results_t_cdcs[key] =      np.empty(shape=(n_methodologies, folds * repeats), dtype='object')
#                self.results_t_cdcs_cal[key] =  np.empty(shape=(n_methodologies, folds * repeats), dtype='object')
#                self.results_t_class_imb[key] = np.empty(shape=(n_methodologies, folds * repeats), dtype='object')
#                self.results_t_ins[key] =       np.empty(shape=(n_methodologies, folds * repeats), dtype='object')

        for i, (train_val_index, test_index) in enumerate(rskf.split(covariates, prepr)):
            print('\nCross validation: ' + str(i + 1))

            index = 0

            x_train_val, x_test = covariates.iloc[train_val_index], covariates.iloc[test_index]
            y_train_val, y_test = labels[train_val_index], labels[test_index]
            amounts_train_val, amounts_test = amounts[train_val_index], amounts[test_index]
            cost_matrix_train_val, cost_matrix_test = cost_matrix[train_val_index, :], cost_matrix[test_index, :]

            # Split training and validation set
            train_ratio = 1 - self.settings['val_ratio']
            skf = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio, random_state=42)
            prepr_val = y_train_val

            for train_index, val_index in skf.split(x_train_val, prepr_val):
                x_train, x_val = x_train_val.iloc[train_index], x_train_val.iloc[val_index]
                y_train, y_val = y_train_val[train_index], y_train_val[val_index]
                cost_matrix_train, cost_matrix_val = cost_matrix_train_val[train_index, :], cost_matrix_train_val[
                                                                                            val_index, :]

        #        # Setting: instance or class-dependent costs?
        #        if self.settings['class_costs']:
        #            cost_matrix_train = np.tile(cost_matrix_train.mean(axis=0)[None, :], (len(y_train), 1, 1))
        #            cost_matrix_val = np.tile(cost_matrix_val.mean(axis=0)[None, :], (len(y_val), 1, 1))

            # Preprocessing: Handle missing data, convert categorical variables, standardize, convert to numpy

            x_train, x_val, x_test, categorical_variables = handle_missing_data(x_train, x_val, x_test, categorical_variables) #TODO: debug
            cat_encoder = self.settings['cat_encoder']
            x_train, x_val, x_test = convert_categorical_variables(x_train, y_train, x_val, x_test, categorical_variables, cat_encoder)
            x_train, x_val, x_test = standardize(x_train=x_train, x_val=x_val, x_test=x_test)

            #After preprocessing properly, concatenate train and val into train_val
            x_train_val = np.concatenate((x_train, x_val))
            y_train_val = np.concatenate((y_train, y_val))


            """Assign thresholds for the different strategies:"""
            #   Instance-dependent cost-sensitive threshold
            if self.thresholding['t_idcs']:
                threshold = (cost_matrix_test[:, 1, 0] - cost_matrix_test[:, 0, 0]) / (cost_matrix_test[:, 1, 0] - cost_matrix_test[:, 0, 0] + cost_matrix_test[:, 0,1] - cost_matrix_test[:, 1, 1])

            #   Instance-dependent cost-sensitive threshold, calibrated
            if self.thresholding['t_idcs_cal']: #TODO
                print("todo")

                """
                # ID CS Threshold with calibrated probabilities (using isotonic regression):
                isotonic = IsotonicRegression(out_of_bounds='clip')
                isotonic.fit(proba_val, y_val)     # Fit on validation set!
                proba_calibrated = isotonic.transform(proba)
                """

            #   Class-dependent cost-sensitive threshold
            if self.thresholding['t_cdcs']:
                threshold_class = (cost_matrix_test[:, 1, 0].mean() - cost_matrix_test[:, 0, 0].mean()) / (cost_matrix_test[:, 1, 0].mean() - cost_matrix_test[:, 0, 0].mean() + cost_matrix_test[:, 0, 1].mean() - cost_matrix_test[:, 1, 1].mean())
                threshold = np.repeat(threshold_class, len(y_test))

            #   Class-dependent cost-sensitive threshold, calibrated
            if self.thresholding['t_cdcs_cal']: #TODO
                print("todo")

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


            """
            if .... --> select classifier
                print classifier
                create calssifier object
                assign parameter grid
                create gridsearch cross-validation object
                Fit the grid search object to the train_val data (which is timed)
                print best hyperparas
                predict probabilities for x_test and x_val
                evaluate model
                increase index counter
            """

            def create_pipeline_and_param_grid(self, param_grid, prefix: str, method):
                if self.settings['oversampling'] == 0:
                    pipeline = method
                else:
                    oversampler = {
                        1: SMOTE(random_state=0),
                        2: RandomOverSampler(random_state=0),
                        3: ADASYN(random_state=0)
                    }[self.settings['oversampling']]
                    pipeline = make_pipeline(oversampler, method)
                    param_grid = add_prefix_to_hyperparams(param_grid, prefix)
                return pipeline, param_grid

            # AdaBoost
            if self.methodologies['ab']:
                param_grid = self.hyperparameters['ab']
                method = AdaBoostClassifier()
                pipeline, param_grid = create_pipeline_and_param_grid(self, param_grid, 'adaboostclassifier__', method)

                #if self.settings['oversampling'] == 0:
                #    pipeline = AdaBoostClassifier()
                #else:
                #    oversampler = {
                #        1: SMOTE,
                #        2: RandomOverSampler,
                #        3: ADASYN
                #    }[self.settings['oversampling']]
                #    pipeline = make_pipeline(oversampler(random_state=0), AdaBoostClassifier())
                #    param_grid = add_prefix_to_hyperparams(param_grid, 'adaboostclassifier')

                #if self.settings['oversampling'] == 0:
                #    pipeline = AdaBoostClassifier()
                #if self.settings['oversampling'] == 1:
                #    # Create a pipeline with SMOTE and AdaBoostClassifier
                #    pipeline = make_pipeline(SMOTE(random_state=0), AdaBoostClassifier())
                #    param_grid = add_prefix_to_hyperparams(param_grid, 'adaboostclassifier')
                #if self.settings['oversampling'] == 2:
                #    pipeline = make_pipeline(RandomOverSampler(random_state=0), AdaBoostClassifier())
                #    param_grid = add_prefix_to_hyperparams(param_grid, 'adaboostclassifier')
                #if self.settings['oversampling'] == 3:
                #    pipeline = make_pipeline(ADASYN(random_state=0), AdaBoostClassifier())
                #    param_grid = add_prefix_to_hyperparams(param_grid, 'adaboostclassifier')

                # Create a GridSearchCV object
                gs = GridSearchCV(pipeline, param_grid=param_grid, scoring='accuracy', cv=5)

                start = time.perf_counter()

                gs.fit(x_train_val, y_train_val)    #oversampling is included in pipeline object, which is included in the gs object
                end = time.perf_counter()

                print('\tab - best hyperparameters:', gs.best_params_)


                proba_test = gs.predict_proba(x_test)[:, 1]
                proba_val = gs.predict_proba(x_val)[:, 1]

                info = {'time': end - start}

                evaluate_model(proba_val, proba_test, i, index, info)

                index += 1

            # Artifical Neural Network  #TODO

            # Bernoulli Naive Bayes
            if self.methodologies['bnb']:
                param_grid = self.hyperparameters['bnb']
                method = BernoulliNB()
                pipeline, param_grid = create_pipeline_and_param_grid(self, param_grid, "bernoullinb__", method)

                gs = GridSearchCV(pipeline, param_grid=param_grid, scoring='accuracy', cv=5)

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

                gs = GridSearchCV(pipeline, param_grid=param_grid, scoring='accuracy', cv=5)

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

                gs = GridSearchCV(pipeline, param_grid=param_grid, scoring='accuracy', cv=5)

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

                gs = GridSearchCV(pipeline, param_grid=param_grid, scoring='accuracy', cv=5)

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

                gs = GridSearchCV(pipeline, param_grid=param_grid, scoring='accuracy', cv=5)

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

                gs = GridSearchCV(pipeline, param_grid=param_grid, scoring='accuracy', cv=5)

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

                gs = GridSearchCV(pipeline, param_grid=param_grid, scoring='accuracy', cv=5)

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

                gs = GridSearchCV(pipeline, param_grid=param_grid, scoring='accuracy', cv=5)

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

                gs_lr = GridSearchCV(pipeline, param_grid=param_grid, scoring='accuracy', cv=5)

                # Fitting the grid search object to the train_val data
                start = time.perf_counter()
                gs_lr.fit(x_train_val, y_train_val)
                end = time.perf_counter()

                print('\tlr - best hyperparameters:', gs_lr.best_params_)

                lr_proba = gs_lr.predict_proba(x_test)[:, 1]
                lr_proba_val = gs_lr.predict_proba(x_val)[:, 1]

                info = {'time': end - start}

                evaluate_model(lr_proba_val, lr_proba, i, index, info)

                index += 1

            # Multinomial Naive Bayes
            if self.methodologies['mnb']:
                method = MultinomialNB()
                param_grid = self.hyperparameters['mnb']
                pipeline, param_grid = create_pipeline_and_param_grid(self, param_grid, "multinomialnb__", method)

                gs = GridSearchCV(pipeline, param_grid=param_grid, scoring='accuracy', cv=5)

                #TODO: mnb can only handle non-negative input data
                #extra preprocessing step:
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

                gs = GridSearchCV(pipeline, param_grid=param_grid, scoring='accuracy', cv=5)

                start = time.perf_counter()
                gs.fit(x_train_val, y_train_val)
                end = time.perf_counter()

                print('\tpac - best hyperparameters:', gs.best_params_)

               # proba_test = gs.decision_function(x_test) #Todo: remove line - dec_funct does not output probabilities
               # proba_val = gs.decision_function(x_val)    #Todo: remove line - dec_funct does not output probabilities

                d_test = gs.decision_function(x_test)
                proba_test = np.exp(d_test) / (1 + np.exp(d_test))

                d_val = gs.decision_function(x_val)
                proba_val = np.exp(d_val) / (1 + np.exp(d_val))

                info = {'time': end - start}

                evaluate_model(proba_val, proba_test, i, index, info)

                index += 1

            """
            # Perceptron
            if self.methodologies['per']:
                model = Perceptron()
                param_grid = self.hyperparameters['per']
                gs = GridSearchCV(model, param_grid=param_grid, scoring='accuracy', cv=5)

                start = time.perf_counter()
                gs.fit(x_train_val, y_train_val)
                end = time.perf_counter()

                print('\tper - best hyperparameters:', gs.best_params_)

                proba_test = gs.decision_function(x_test)
                proba_val = gs.decision_function(x_val)

                info = {'time': end - start}

                evaluate_model(proba_val, proba_test, i, index, info)

                index += 1
            """

            # Quadratic Discriminant Analysis
            if self.methodologies['qda']:
                method = QuadraticDiscriminantAnalysis()
                param_grid = self.hyperparameters['qda']
                pipeline, param_grid = create_pipeline_and_param_grid(self, param_grid, "quadraticdiscriminantanalysis__", method)

                gs = GridSearchCV(pipeline, param_grid=param_grid, scoring='accuracy', cv=5)

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

                gs = GridSearchCV(pipeline, param_grid=param_grid, scoring='accuracy', cv=5)

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

                gs = GridSearchCV(pipeline, param_grid=param_grid, scoring='accuracy', cv=5)

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

                gs = GridSearchCV(pipeline, param_grid=param_grid, scoring='accuracy', cv=5)

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

                gs = GridSearchCV(pipeline, param_grid=param_grid, scoring='accuracy', cv=5)

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

                gs_xgb = GridSearchCV(pipeline, param_grid=param_grid, scoring='accuracy', cv=5)

                # Fitting the grid search object to the train_val data
                start = time.perf_counter()
                gs_xgb.fit(x_train_val, y_train_val)
                end = time.perf_counter()

                print('\txgb - best hyperparameters:', gs_xgb.best_params_)

                xgb_proba = gs_xgb.predict_proba(x_test)[:, 1]
                xgb_proba_val = gs_xgb.predict_proba(x_val)[:, 1]

                info = {'time': end - start}

                evaluate_model(xgb_proba_val, xgb_proba, i, index, info)

                index += 1

            print('\n----------------------------------------------------------------')

    def evaluate(self, directory):
        """
        EVALUATION
        """
        print('\n\n********* EVALUATING CLASSIFIERS *********')


        #TODO - strongly simplified:
        #write to file
            #jfkdlsm

        print('\n*** Results ***')
        #print('Thresholding method: '+str(self.thresholding)+'\n')

        evaluate_experiments(evaluators=self.evaluators,
                             methodologies=self.methodologies,
                             thresholding=self.thresholding,
                             evaluation_matrices=self.results,
                             directory=directory,
                             name='id')














