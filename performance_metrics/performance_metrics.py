#TODO: imports
import numpy as np
from tabulate import tabulate
from scipy.stats import spearmanr, combine_pvalues, friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman

def get_performance_metrics(evaluators, evaluation_matrices, i, index, cost_matrix, labels, probabilities, predictions, info):
    if evaluators['traditional']:
        true_pos = (predictions * labels).sum()
        true_neg = ((1 - predictions) * (1 - labels)).sum()
        false_pos = (predictions * (1 - labels)).sum()
        false_neg = ((1 - predictions) * labels).sum()

        accuracy = (true_pos + true_neg) / len(labels)
        recall = true_pos / (true_pos + false_neg)
        # Make sure no division by 0!
        if (true_pos == 0) and (false_pos == 0):
            precision = 0
            print('\t\tWARNING: No positive predictions!')
        else:
            precision = true_pos / (true_pos + false_pos)
        if precision == 0:
            f1_score = 0
            print('\t\tWARNING: Precision = 0!')
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)
        #TODO: add sensitivity and specificity?

        evaluation_matrices['traditional'][index, i] = np.array([accuracy, recall, precision, f1_score])

    #TODO: alle andere metrics aanvullen met if evaluators['..']: statements

    return evaluation_matrices

def evaluate_experiments(evaluators, methodologies, thresholding, evaluation_matrices, directory, name):    #TODO: hele functie grondig bekijken
    table_evaluation = []
    n_methodologies = sum(methodologies.values())

    names = []
    for key in methodologies.keys():
        if methodologies[key]:
            names.append(key)

    if evaluators['traditional']:

        table_traditional = [['Method', 'Accuracy', 'Recall', 'Precision', 'F1-score', 'AR', 'sd']]

        # Compute F1 rankings (- as higher is better)
        all_f1s = []
        for i in range(evaluation_matrices['traditional'].shape[0]):
            method_f1s = []
            for j in range(evaluation_matrices['traditional'][i].shape[0]):
                f1 = evaluation_matrices['traditional'][i][j][-1]
                method_f1s.append(f1)
            all_f1s.append(np.array(method_f1s))

        ranked_args = np.argsort(-np.array(all_f1s), axis=0)
        rankings = np.arange(len(ranked_args))[ranked_args.argsort(axis=0)]
        rankings = rankings + 1
        avg_rankings = np.mean(rankings, axis=1)
        sd_rankings = np.sqrt(rankings.var(axis=1))

        # Summarize all per method
        index = 0
        for item, value in methodologies.items():
            if value:
                averages = evaluation_matrices['traditional'][index, :].mean()

                table_traditional.append([item, averages[0], averages[1], averages[2], averages[3],
                                          avg_rankings[index], sd_rankings[index]])

                index += 1

        print(tabulate(table_traditional, headers="firstrow", floatfmt=("", ".4f", ".4f", ".4f", ".4f", ".4f", ".4f")))
        table_evaluation.append(table_traditional)

        # Do tests if enough measurements are available (at least 3)
        if np.array(all_f1s).shape[1] > 2:
            friedchisq = friedmanchisquare(*np.transpose(all_f1s))
            print('\nF1 - Friedman test')
            print('H0: Model performance follows the same distribution')
            print('\tChi-square:\t%.4f' % friedchisq[0])
            print('\tp-value:\t%.4f' % friedchisq[1])
            if friedchisq[1] < 0.05:  # If p-value is significant, do Nemenyi post hoc test
                # Post-hoc Nemenyi Friedman: Rows are blocks, columns are groups
                nemenyi = posthoc_nemenyi_friedman(np.array(all_f1s).T.astype(dtype=np.float32))
                print('\nNemenyi post hoc test:')
                print(nemenyi)

        print('_________________________________________________________________________')











