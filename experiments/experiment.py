
#TODO add imports
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import xgboost
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
import time
from performance_metrics.performance_metrics import get_performance_metrics, evaluate_experiments
from preprocessing.preprocessing import preprocess_ibm, handle_missing_data, convert_categorical_variables, standardize
from sklearn.model_selection import GridSearchCV

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
                            train model
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
        global xgb
        print('\n\n************** LOADING DATA **************\n')



        if self.datasets['ibm']:
            print('ibm')
            covariates, labels, amounts, cost_matrix, categorical_variables = preprocess_ibm() #TODO: implement function
        elif self.datasets['acerta']:
            print('acerta')
            covariates, labels, amounts, cost_matrix, categorical_variables = preprocess_acerta() #TODO: implement function
        elif self.datasets['babushkin']:
            print('babushkin')
            covariates, labels, amounts, cost_matrix, categorical_variables = preprocess_babushkin() #TODO: implement function
        elif self.datasets['eds']:
            print('eds')
            covariates, labels, amounts, cost_matrix, categorical_variables = preprocess_eds() #TODO: implement function
        else:
            raise Exception('No dataset specified')


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
#            x_train, x_val, x_test, categorical_variables = handle_missing_data(x_train, x_val, x_test, categorical_variables) #TODO: debug
            x_train, x_val, x_test = convert_categorical_variables(x_train, y_train, x_val, x_test, categorical_variables)
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
            def evaluate_model(proba_val, proba, j, index, info):

                #round proba to predictions {0,1}, dependent on defined threshold.
                pred = (proba > threshold).astype(int)
#                self.results_tr_instance = get_performance_metrics(self.evaluators, self.results_tr_instance, j, index, cost_matrix_test, y_test, proba, pred, info)
                self.results = get_performance_metrics(self.evaluators, self.results, j, index, cost_matrix_test, y_test, proba, pred, info)


            # Logistic regression
            if self.methodologies['lr']:
                print('\tlr')

                lr = LogisticRegression()

                param_grid = self.hyperparameters['lr']

                gs_lr = GridSearchCV(lr, param_grid=param_grid, scoring='accuracy', cv=5)

                # Fitting the grid search object to the train_val data
                start = time.perf_counter()
                gs_lr.fit(x_train_val, y_train_val)
                end = time.perf_counter()

                print("Best hyperparameters:", gs_lr.best_params_)

                lambda1 = None #TODO: can be deleted
                lambda2 = None

                lr_proba = gs_lr.predict_proba(x_test)[:, 1]
                lr_proba_val = gs_lr.predict_proba(x_val)[:, 1]

                #TODO: update 'info': only time is relevant
                info = {'time': end - start, 'lambda1': lambda1, 'lambda2': lambda2, 'n_neurons': 0}

                evaluate_model(lr_proba_val, lr_proba, i, index, info)

                index += 1

            #XGBoosting
            if self.methodologies['xgb']:
                print('\txgb')

                xgb = xgboost.XGBClassifier()

                param_grid = self.hyperparameters['xgb']

                gs_xgb = GridSearchCV(xgb, param_grid=param_grid, scoring='accuracy', cv=5)

                # Fitting the grid search object to the train_val data
                start = time.perf_counter()
                gs_xgb.fit(x_train_val, y_train_val)
                end = time.perf_counter()

                print("Best hyperparameters:", gs_xgb.best_params_)

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
        print('Thresholding method: '+str(self.thresholding)+'\n')
        evaluate_experiments(evaluators=self.evaluators,
                             methodologies=self.methodologies,
                             thresholding=self.thresholding,
                             evaluation_matrices=self.results,
                             directory=directory,
                             name='id')














