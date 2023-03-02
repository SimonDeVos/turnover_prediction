#TODO: imports
import numpy as np
from sklearn import metrics
from tabulate import tabulate
from scipy.stats import spearmanr, combine_pvalues, friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman


def savings(cost_matrix, labels, predictions):
    cost_without = cost_without_algorithm(cost_matrix, labels)
    cost_with = cost_with_algorithm(cost_matrix, labels, predictions)
    savings = 1 - cost_with / cost_without

    return savings


def cost_with_algorithm(cost_matrix, labels, predictions):

    cost_tn = cost_matrix[:, 0, 0][np.logical_and(predictions == 0, labels == 0)].sum()
    cost_fn = cost_matrix[:, 0, 1][np.logical_and(predictions == 0, labels == 1)].sum()
    cost_fp = cost_matrix[:, 1, 0][np.logical_and(predictions == 1, labels == 0)].sum()
    cost_tp = cost_matrix[:, 1, 1][np.logical_and(predictions == 1, labels == 1)].sum()

    return sum((cost_tn, cost_fn, cost_fp, cost_tp))


def cost_without_algorithm(cost_matrix, labels):

    # Predict everything as the default class that leads to minimal cost
    # Also include cost of TP/TN!
    cost_neg = cost_matrix[:, 0, 0][labels == 0].sum() + cost_matrix[:, 0, 1][labels == 1].sum()
    cost_pos = cost_matrix[:, 1, 0][labels == 0].sum() + cost_matrix[:, 1, 1][labels == 1].sum()

    return min(cost_neg, cost_pos)


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


    if evaluators['ROC']:
        fpr, tpr, roc_thresholds = metrics.roc_curve(y_true=labels, y_score=probabilities)

        evaluation_matrices['ROC'][index, i] = np.array([fpr, tpr, roc_thresholds])

    if evaluators['AUC']:
        auc = metrics.roc_auc_score(y_true=labels, y_score=probabilities)

        evaluation_matrices['AUC'][index, i] = auc

    if evaluators['savings']:
        # To do: function - savings
        cost_without = cost_without_algorithm(cost_matrix, labels)
        cost_with = cost_with_algorithm(cost_matrix, labels, predictions)
        savings = 1 - cost_with / cost_without

        evaluation_matrices['savings'][index, i] = savings

    if evaluators['AEC']:
        expected_cost = labels * (probabilities * cost_matrix[:, 1, 1] + (1 - probabilities) * cost_matrix[:, 0, 1]) \
            + (1 - labels) * (probabilities * cost_matrix[:, 1, 0] + (1 - probabilities) * cost_matrix[:, 0, 0])

        aec = expected_cost.mean()

        evaluation_matrices['AEC'][index, i] = aec

    if evaluators['PR']:
        precision, recall, _ = metrics.precision_recall_curve(y_true=labels, probas_pred=probabilities)

        # AUC is not recommended here (see sklearn docs)
        # We will use Average Precision (AP)
        ap = metrics.average_precision_score(y_true=labels, y_score=probabilities)

        evaluation_matrices['PR'][index, i] = np.array([precision, recall, ap], dtype=object)

    if evaluators['brier']:

        brier = ((probabilities - labels)**2).mean()

        evaluation_matrices['brier'][index, i] = brier

    if evaluators['recall_overlap']:

        recalled = labels[labels == 1] * predictions[labels == 1]

        evaluation_matrices['recall_overlap'][index, i] = recalled

    if evaluators['recall_correlation']:

        pos_probas = probabilities[labels == 1]

        # Sort indices from high to low
        sorted_indices_probas = np.argsort(pos_probas)[::-1]
        prob_rankings = np.argsort(sorted_indices_probas)

        evaluation_matrices['recall_correlation'][index, i] = prob_rankings

    if evaluators['time']:

        evaluation_matrices['time'][index, i] = info['time']

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

        table_traditional = [['Method', 'Accuracy','Acc_sd', 'Recall','Rec_sd', 'Precision','Pr_sd', 'F1-score','F1_sd', 'AR', 'AR_sd']]

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
                deviation = evaluation_matrices['traditional'][index, :].std()
                table_traditional.append([item, averages[0], deviation[0], averages[1], deviation[1], averages[2], deviation[2],  averages[3], deviation[3],
                                          avg_rankings[index], sd_rankings[index]])

                index += 1

        print(tabulate(table_traditional, headers="firstrow", floatfmt=("", ".4f",".4f",".4f",".4f",".4f", ".4f", ".4f", ".4f", ".4f", ".4f")))
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

    if evaluators['AUC']:

        table_auc = [['Method', 'AUC', 'sd', 'AR', 'sd']]

        # Compute rankings (- as higher is better)
        ranked_args = (-evaluation_matrices['AUC']).argsort(axis=0)
        rankings = np.arange(len(ranked_args))[ranked_args.argsort(axis=0)]
        rankings = rankings + 1
        avg_rankings = rankings.mean(axis=1)
        sd_rankings = np.sqrt(rankings.var(axis=1))

        # Summarize per method
        index = 0
        for item, value in methodologies.items():
            if value:
                table_auc.append([item, evaluation_matrices['AUC'][index, :].mean(),
                                  np.sqrt(evaluation_matrices['AUC'][index, :].var()), avg_rankings[index],
                                  sd_rankings[index]])
                index += 1

        print(tabulate(table_auc, headers="firstrow", floatfmt=("", ".4f", ".4f", ".4f", ".4f")))
        table_evaluation.append(table_auc)

        # Do tests if enough measurements are available (at least 3)
        if evaluation_matrices['AUC'].shape[1] > 2:
            friedchisq = friedmanchisquare(*evaluation_matrices['AUC'].T)
            print('\nAUC - Friedman test')
            print('H0: Model performance follows the same distribution')
            print('\tChi-square:\t%.4f' % friedchisq[0])
            print('\tp-value:\t%.4f' % friedchisq[1])
            if friedchisq[1] < 0.05:  # If p-value is significant, do Nemenyi post hoc test
                nemenyi = posthoc_nemenyi_friedman(evaluation_matrices['AUC'].T.astype(dtype=np.float32))
                print('\nNemenyi post hoc test:')
                print(nemenyi)

        print('_________________________________________________________________________')









