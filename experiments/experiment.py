
#TODO add imports


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
    def __init__(self, settings, datasets, methodologies, thresholding, evaluators):

        self.settings = settings

        self.l1 = self.settings['l1_regularization']
        self.lambda1_list = self.settings['lambda1_options']
        self.l2 = self.settings['l2_regularization']
        self.lambda2_list = self.settings['lambda2_options']
        self.neurons_list = self.settings['neurons_options']

        if self.l1 and self.l2:
            raise ValueError('Only l1 or l2 regularization allowed, not both!')

        self.datasets = datasets
        self.methodologies = methodologies
        self.thresholding = thresholding
        self.evaluators = evaluators

        self.results_t_idcs = {}
        self.results_t_idcs_cal = {}
        self.results_t_cdcs = {}
        self.results_t_cdcs_cal = {}
        self.results_t_class_imb = {}
        self.results_t_ins = {}


    def run(self, directory):
        """
        LOAD AND PREPROCESS DATA
        """
        print('\n\n************** LOADING DATA **************\n')

        # Verify that only one dataset is selected
        if sum(self.datasets.values()) != 1:
            raise ValueError('Select only one dataset!')

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









