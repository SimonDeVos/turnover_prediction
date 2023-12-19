
import os
import warnings

import numpy as np
#import wandb
from sklearn import metrics
from sklearn.metrics import precision_recall_curve, auc, accuracy_score, f1_score
from tabulate import tabulate
from scipy.stats import spearmanr, combine_pvalues, friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman
from hmeasure import h_score
import csv

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

def round_nested_list(nested_list): #Used for printing results to .txt file
    """
    :param nested_list:
    :return: list where floats have max 4 decimals
    """
    rounded_list = []
    for elem in nested_list:
        if isinstance(elem, float):
            rounded_list.append(round(elem, 4))
        elif isinstance(elem, list):
            rounded_list.append(round_nested_list(elem))
        else:
            rounded_list.append(elem)
    return rounded_list


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

        evaluation_matrices['traditional'][index, i] = np.array([accuracy, recall, precision, f1_score])

    if evaluators['Accuracy']:
        true_pos = (predictions * labels).sum()
        true_neg = ((1 - predictions) * (1 - labels)).sum()
        false_pos = (predictions * (1 - labels)).sum()
        false_neg = ((1 - predictions) * labels).sum()
        accuracy = (true_pos + true_neg) / len(labels)

        evaluation_matrices['Accuracy'][index, i] = accuracy

    if evaluators['Recall']:
        true_pos = (predictions * labels).sum()
        true_neg = ((1 - predictions) * (1 - labels)).sum()
        false_pos = (predictions * (1 - labels)).sum()
        false_neg = ((1 - predictions) * labels).sum()
        recall = true_pos / (true_pos + false_neg)
        evaluation_matrices['Recall'][index, i] = recall

    if evaluators['Precision']:
        true_pos = (predictions * labels).sum()
        true_neg = ((1 - predictions) * (1 - labels)).sum()
        false_pos = (predictions * (1 - labels)).sum()
        false_neg = ((1 - predictions) * labels).sum()

        # Make sure no division by 0!
        if (true_pos == 0) and (false_pos == 0):
            precision = 0
            print('\t\tWARNING: No positive predictions!')
        else:
            precision = true_pos / (true_pos + false_pos)
        evaluation_matrices['Precision'][index, i] = precision

    if evaluators['Specificity']:
        true_pos = (predictions * labels).sum()
        true_neg = ((1 - predictions) * (1 - labels)).sum()
        false_pos = (predictions * (1 - labels)).sum()
        false_neg = ((1 - predictions) * labels).sum()

        # Make sure no division by 0!
        if (true_pos == 0) and (false_pos == 0):
            specificity = 0
            print('\t\tWARNING: No positive predictions!')
        else:
            specificity = true_neg / (true_neg + false_pos)
        evaluation_matrices['Specificity'][index, i] = specificity


    if evaluators['F1']:
        true_pos = (predictions * labels).sum()
        true_neg = ((1 - predictions) * (1 - labels)).sum()
        false_pos = (predictions * (1 - labels)).sum()
        false_neg = ((1 - predictions) * labels).sum()

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

        evaluation_matrices['F1'][index, i] = f1_score


    if evaluators['AUC-PR']:
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        auc_pr = auc(recall, precision)

        evaluation_matrices['AUC-PR'][index, i] = auc_pr

#    if evaluators['ROC']:
#        fpr, tpr, roc_thresholds = metrics.roc_curve(y_true=labels, y_score=probabilities)
#        evaluation_matrices['ROC'][index, i] = np.array([fpr, tpr, roc_thresholds])

    if evaluators['AUC-ROC']:
        auc_roc = metrics.roc_auc_score(y_true=labels, y_score=probabilities)

        evaluation_matrices['AUC-ROC'][index, i] = auc_roc

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

    if evaluators['H_measure']:
        h_measure = h_score(labels, predictions)
        evaluation_matrices['H_measure'][index, i] = h_measure


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

    return evaluation_matrices

def evaluate_experiments(evaluators, methodologies, thresholding, evaluation_matrices, settings, dataset, directory, name):

    table_evaluation = []
    n_methodologies = sum(methodologies.values())

    names = []
    for key in methodologies.keys():
        if methodologies[key]:
            names.append(key)

    if evaluators['traditional']:

        table_traditional = [['Method', 'Accuracy','Acc_sd', 'Recall','Rec_sd', 'Precision','Pr_sd', 'F1-score','F1_sd', 'F1_AR', 'F1_AR_sd']]

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


        if evaluators['stat_hypothesis_testing']:
            if n_methodologies < 2:
                raise Warning('"stat_hypothesis_testing" is True, but not enough methods are compare. At least 2 are needed')

            # Do tests if enough measurements are available (at least 3)
            if np.array(all_f1s).shape[1] > 2:
                friedchisq = friedmanchisquare(*np.transpose(all_f1s))
                print('\nF1 - Friedman test')
                print('H0: There are no significant differences between the models')
                print('\tChi-square:\t%.4f' % friedchisq[0])
                print('\tp-value:\t%.4f' % friedchisq[1])
                if friedchisq[1] < 0.05:  # If p-value is significant, do Nemenyi post hoc test
                    print('\t\t'+str(round(friedchisq[1],4)) +" < 0.05: at least one model differs significantly from the others")
                    # Post-hoc Nemenyi Friedman: Rows are blocks, columns are groups
                    nemenyi = posthoc_nemenyi_friedman(np.array(all_f1s).T.astype(dtype=np.float32))
                    print('\nNemenyi post hoc test:')
                    print(nemenyi)
            elif np.array(all_f1s).shape[1] < 3:
                raise Warning('"stat_hypothesis_testing" is True, but not enough measurements. At least 3 are needed')
        print('_________________________________________________________________________')

    if evaluators['Accuracy']:
        table_acc = [['Method', 'Acc', 'Acc_sd', 'Acc_AR', 'Acc_AR_sd']]

        # Compute rankings (- as higher is better)
        ranked_args = (-evaluation_matrices['Accuracy']).argsort(axis=0)
        rankings = np.arange(len(ranked_args))[ranked_args.argsort(axis=0)]
        rankings = rankings + 1
        avg_rankings = rankings.mean(axis=1)
        sd_rankings = np.sqrt(rankings.var(axis=1))

        # Summarize per method
        index = 0
        for item, value in methodologies.items():
            if value:
                table_acc.append([item, evaluation_matrices['Accuracy'][index, :].mean(),
                                  np.sqrt(evaluation_matrices['Accuracy'][index, :].var()), avg_rankings[index],
                                  sd_rankings[index]])
                index += 1

        print(tabulate(table_acc, headers="firstrow", floatfmt=("", ".4f", ".4f", ".4f", ".4f")))
        table_evaluation.append(table_acc)
        print('_________________________________________________________________________')

    if evaluators['Recall']:
        table_rec = [['Method', 'Recall', 'Rec_sd', 'Rec_AR', 'Rec_AR_sd']]

        # Compute rankings (- as higher is better)
        ranked_args = (-evaluation_matrices['Recall']).argsort(axis=0)
        rankings = np.arange(len(ranked_args))[ranked_args.argsort(axis=0)]
        rankings = rankings + 1
        avg_rankings = rankings.mean(axis=1)
        sd_rankings = np.sqrt(rankings.var(axis=1))

        # Summarize per method
        index = 0
        for item, value in methodologies.items():
            if value:
                table_rec.append([item, evaluation_matrices['Recall'][index, :].mean(),
                                  np.sqrt(evaluation_matrices['Recall'][index, :].var()), avg_rankings[index],
                                  sd_rankings[index]])
                index += 1

        print(tabulate(table_rec, headers="firstrow", floatfmt=("", ".4f", ".4f", ".4f", ".4f")))
        table_evaluation.append(table_rec)
        print('_________________________________________________________________________')

    if evaluators['Precision']:
        table_prec = [['Method', 'Precision', 'Prec_sd', 'Prec_AR', 'Prec_AR_sd']]

        # Compute rankings (- as higher is better)
        ranked_args = (-evaluation_matrices['Precision']).argsort(axis=0)
        rankings = np.arange(len(ranked_args))[ranked_args.argsort(axis=0)]
        rankings = rankings + 1
        avg_rankings = rankings.mean(axis=1)
        sd_rankings = np.sqrt(rankings.var(axis=1))

        # Summarize per method
        index = 0
        for item, value in methodologies.items():
            if value:
                table_prec.append([item, evaluation_matrices['Precision'][index, :].mean(),
                                  np.sqrt(evaluation_matrices['Precision'][index, :].var()), avg_rankings[index],
                                  sd_rankings[index]])
                index += 1

        print(tabulate(table_prec, headers="firstrow", floatfmt=("", ".4f", ".4f", ".4f", ".4f")))
        table_evaluation.append(table_prec)
        print('_________________________________________________________________________')

    if evaluators['Specificity']:
        table_spec = [['Method', 'Specificity', 'Spec_sd', 'Spec_AR', 'Spec_AR_sd']]

        # Compute rankings (- as higher is better)
        ranked_args = (-evaluation_matrices['Specificity']).argsort(axis=0)
        rankings = np.arange(len(ranked_args))[ranked_args.argsort(axis=0)]
        rankings = rankings + 1
        avg_rankings = rankings.mean(axis=1)
        sd_rankings = np.sqrt(rankings.var(axis=1))

        # Summarize per method
        index = 0
        for item, value in methodologies.items():
            if value:
                table_spec.append([item, evaluation_matrices['Specificity'][index, :].mean(),
                                  np.sqrt(evaluation_matrices['Specificity'][index, :].var()), avg_rankings[index],
                                  sd_rankings[index]])
                index += 1

        print(tabulate(table_spec, headers="firstrow", floatfmt=("", ".4f", ".4f", ".4f", ".4f")))
        table_evaluation.append(table_spec)
        print('_________________________________________________________________________')

    if evaluators['F1']:
        table_F1 = [['Method', 'F1', 'F1_sd', 'F1_AR', 'F1_AR_sd']]

        # Compute rankings (- as higher is better)
        ranked_args = (-evaluation_matrices['F1']).argsort(axis=0)
        rankings = np.arange(len(ranked_args))[ranked_args.argsort(axis=0)]
        rankings = rankings + 1
        avg_rankings = rankings.mean(axis=1)
        sd_rankings = np.sqrt(rankings.var(axis=1))

        # Summarize per method
        index = 0
        for item, value in methodologies.items():
            if value:
                table_F1.append([item, evaluation_matrices['F1'][index, :].mean(),
                                  np.sqrt(evaluation_matrices['F1'][index, :].var()), avg_rankings[index],
                                  sd_rankings[index]])
                index += 1

        print(tabulate(table_F1, headers="firstrow", floatfmt=("", ".4f", ".4f", ".4f", ".4f")))
        table_evaluation.append(table_F1)


        print('_________________________________________________________________________')

    if evaluators['AUC-PR']:
        table_aucpr = [['Method', 'AUC-PR', 'AUC-PR_sd', 'AUC-PR_AR', 'AUC-PR_AR_sd']]

        # Compute rankings (- as higher is better)
        ranked_args = (-evaluation_matrices['AUC-PR']).argsort(axis=0)
        rankings = np.arange(len(ranked_args))[ranked_args.argsort(axis=0)]
        rankings = rankings + 1
        avg_rankings = rankings.mean(axis=1)
        sd_rankings = np.sqrt(rankings.var(axis=1))

        # Summarize per method
        index = 0
        for item, value in methodologies.items():
            if value:
                table_aucpr.append([item, evaluation_matrices['AUC-PR'][index, :].mean(),
                                  np.sqrt(evaluation_matrices['AUC-PR'][index, :].var()), avg_rankings[index],
                                  sd_rankings[index]])
                index += 1

        print(tabulate(table_aucpr, headers="firstrow", floatfmt=("", ".4f", ".4f", ".4f", ".4f")))
        table_evaluation.append(table_aucpr)

        print('_________________________________________________________________________')

    if evaluators['AUC-ROC']:

        table_aucroc = [['Method', 'AUC-ROC', 'AUC-ROC_sd', 'AUC-ROC_AR', 'AUC-ROC_AR_sd']]

        # Compute rankings (- as higher is better)
        ranked_args = (-evaluation_matrices['AUC-ROC']).argsort(axis=0)
        rankings = np.arange(len(ranked_args))[ranked_args.argsort(axis=0)]
        rankings = rankings + 1
        avg_rankings = rankings.mean(axis=1)
        sd_rankings = np.sqrt(rankings.var(axis=1))

        # Summarize per method
        index = 0
        for item, value in methodologies.items():
            if value:
                table_aucroc.append([item, evaluation_matrices['AUC-ROC'][index, :].mean(),
                                  np.sqrt(evaluation_matrices['AUC-ROC'][index, :].var()), avg_rankings[index],
                                  sd_rankings[index]])
                index += 1

        print(tabulate(table_aucroc, headers="firstrow", floatfmt=("", ".4f", ".4f", ".4f", ".4f")))
        table_evaluation.append(table_aucroc)

        if evaluators['stat_hypothesis_testing']:
            if n_methodologies < 2:
                raise Warning('"stat_hypothesis_testing" is True, but not enough methods are compare. At least 2 are needed')

            # Do tests if enough measurements are available (at least 3)
            if evaluation_matrices['AUC-ROC'].shape[1] > 2:
                friedchisq = friedmanchisquare(*evaluation_matrices['AUC-ROC'].T)
                print('\nAUC-ROC - Friedman test')
                print('H0: There are no significant differences between the models')
                print('\tChi-square:\t%.4f' % friedchisq[0])
                print('\tp-value:\t%.4f' % friedchisq[1])
                if friedchisq[1] < 0.05:  # If p-value is significant, do Nemenyi post hoc test
                    print('\t\t'+str(round(friedchisq[1],4)) +" < 0.05: at least one model differs significantly from the others")
                    nemenyi = posthoc_nemenyi_friedman(evaluation_matrices['AUC-ROC'].T.astype(dtype=np.float32))
                    print('\nNemenyi post hoc test:')
                    print(nemenyi)

            elif evaluation_matrices['AUC-ROC'].shape[1] < 3:
                raise Warning('"stat_hypothesis_testing" is True, but not enough measurements. At least 3 are needed')

        print('_________________________________________________________________________')

    if evaluators['savings']:

        table_savings = [['Method', 'Savings', 'Savings_sd', 'Savings_AR', 'Savings_AR_sd']]

        # Compute rankings (- as higher is better)
        ranked_args = (-evaluation_matrices['savings']).argsort(axis=0)
        rankings = np.arange(len(ranked_args))[ranked_args.argsort(axis=0)]
        rankings = rankings + 1
        avg_rankings = rankings.mean(axis=1)
        sd_rankings = np.sqrt(rankings.var(axis=1))

        # Summarize per method
        index = 0
        methods_used = []
        for item, value in methodologies.items():
            if value:
                methods_used.append(item)
                table_savings.append([item, evaluation_matrices['savings'][index, :].mean(),
                                      np.sqrt(evaluation_matrices['savings'][index, :].var()), avg_rankings[index],
                                      sd_rankings[index]])
                index += 1

        print(tabulate(table_savings, headers="firstrow", floatfmt=("", ".4f", ".4f", ".4f", ".4f")))
        table_evaluation.append(table_savings)

        #TODO: check from cost sens learning - remainder of Savings module
        #TODO: add  if evaluators['stat_hypothesis_testing']:

        print('_________________________________________________________________________')

    if evaluators['AEC']:

        table_aec = [['Method', 'AEC', 'AEC_sd', 'AEC_AR', 'AEC_AR_sd']]

        # Compute rankings (lower is better)
        ranked_args = (evaluation_matrices['AEC']).argsort(axis=0)
        rankings = np.arange(len(ranked_args))[ranked_args.argsort(axis=0)]
        rankings = rankings + 1
        avg_rankings = rankings.mean(axis=1)
        sd_rankings = np.sqrt(rankings.var(axis=1))

        # Summarize per method
        index = 0
        methods_used = []
        for item, value in methodologies.items():
            if value:
                methods_used.append(item)
                table_aec.append([item, evaluation_matrices['AEC'][index, :].mean(),
                                  np.sqrt(evaluation_matrices['AEC'][index, :].var()), avg_rankings[index],
                                  sd_rankings[index]])
                index += 1

        print(tabulate(table_aec, headers="firstrow", floatfmt=("", ".4f", ".4f", ".4f", ".4f")))
        table_evaluation.append(table_aec)

        # plt.xlabel('Methods')
        # plt.ylabel('AEC')
        # # plt.ylim(0, 1)
        # plt.boxplot(np.transpose(evaluation_matrices['AEC']))
        # plt.xticks(np.arange(n_methodologies) + 1, methods_used)
        # plt.xticks(rotation=40)
        # plt.savefig(str(directory + 'AEC_boxplot' + '.png'), bbox_inches='tight')
        # plt.show()

        if evaluators['stat_hypothesis_testing']:
            if n_methodologies < 2:
                raise Warning('"stat_hypothesis_testing" is True, but not enough methods are compare. At least 2 are needed')

            # Do tests if enough measurements are available (at least 3)
            if evaluation_matrices['AEC'].shape[1] > 2:
                friedchisq = friedmanchisquare(*evaluation_matrices['AEC'].T)
                print('\nSavings - Friedman test')
                print('H0: There are no significant differences between the models')
                print('\tChi-square:\t%.4f' % friedchisq[0])
                print('\tp-value:\t%.4f' % friedchisq[1])
                if friedchisq[1] < 0.05:  # If p-value is significant, do Nemenyi post hoc test
                    print('\t\t'+str(round(friedchisq[1],4)) +" < 0.05: at least one model differs significantly from the others")
                    nemenyi = posthoc_nemenyi_friedman(evaluation_matrices['AEC'].T.astype(dtype=np.float32))
                    print('\nNemenyi post hoc test:')
                    print(nemenyi)

            elif evaluation_matrices['AEC'].shape[1] < 3:
                raise Warning('"stat_hypothesis_testing" is True, but not enough measurements. At least 3 are needed')

        print('_________________________________________________________________________')

    if evaluators['PR']:
        table_ap = [['Method', 'Avg_Prec', 'Avg_Prec_sd', 'Avg_Prec_AR', 'Avg_Prec_AR_sd']]

        index = 0
        # fig2, ax2 = plt.subplots()
        # ax2.set_title('PR curve')
        # ax2.set_xlabel('Recall')
        # ax2.set_ylabel('Precision')

        all_aps = []
        for item, value in methodologies.items():
            if value:
                # See https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
                precisions = []
                mean_recall = np.linspace(0, 1, 100)
                aps = []

                for i in range(evaluation_matrices['PR'][index, :].shape[0]):
                    precision, recall, ap = list(evaluation_matrices['PR'][index, i])

                    interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
                    interp_precision[0] = 1
                    precisions.append(interp_precision)

                    aps.append(ap)

                mean_precision = np.mean(precisions, axis=0)
                mean_precision[-1] = 0

                # ax2.plot(mean_recall, mean_precision, label=item, lw=2, alpha=.8)
                # std_precision = np.std(precisions, axis=0)
                # precisions_upper = np.minimum(mean_precision + std_precision, 1)
                # precisions_lower = np.maximum(mean_precision - std_precision, 0)
                # ax2.fill_between(mean_recall, precisions_lower, precisions_upper, color='grey', alpha=.2)

                aps = np.array(aps)
                table_ap.append([item, aps.mean(), np.sqrt(aps.var())])

                all_aps.append(aps)

                index += 1

        # ax2.legend()
        # plt.savefig(str(directory + 'PR.png'), bbox_inches='tight')
        # plt.show()

        # Add rankings (higher is better)
        ranked_args = np.argsort(-np.array(all_aps), axis=0)
        rankings = np.arange(len(ranked_args))[ranked_args.argsort(axis=0)]
        rankings = rankings + 1
        avg_rankings = np.mean(rankings, axis=1)
        sd_rankings = np.sqrt(rankings.var(axis=1))
        for i in range(1, len(table_ap)):
            table_ap[i].append(avg_rankings[i - 1])
            table_ap[i].append(sd_rankings[i - 1])

        print(tabulate(table_ap, headers="firstrow", floatfmt=("", ".4f", ".4f", ".4f", ".4f")))
        table_evaluation.append(table_ap)

        if evaluators['stat_hypothesis_testing']:
            if n_methodologies < 2:
                raise Warning('"stat_hypothesis_testing" is True, but not enough methods are compare. At least 2 are needed')

            # Do tests if enough measurements are available (at least 3)
            if np.array(all_aps).shape[1] > 2:
                friedchisq = friedmanchisquare(*np.transpose(all_aps))
                print('\nAP - Friedman test')
                print('H0: There are no significant differences between the models')
                print('\tChi-square:\t%.4f' % friedchisq[0])
                print('\tp-value:\t%.4f' % friedchisq[1])
                if friedchisq[1] < 0.05:  # If p-value is significant, do Nemenyi post hoc test
                    print('\t\t'+str(round(friedchisq[1],4)) +" < 0.05: at least one model differs significantly from the others")
                    nemenyi = posthoc_nemenyi_friedman(np.array(all_aps).T.astype(dtype=np.float32))
                    print('\nNemenyi post hoc test:')
                    print(nemenyi)
            elif np.array(all_aps).shape[1]  < 3:
                raise Warning('"stat_hypothesis_testing" is True, but not enough measurements. At least 3 are needed')
        print('_________________________________________________________________________')

    if evaluators['H_measure']:

        table_H = [['Method', 'H_measure', 'H_measure_sd', 'H_measure_AR', 'H_measure_AR_sd']]
        # Compute rankings (- as higher is better)
        ranked_args = (-evaluation_matrices['H_measure']).argsort(axis=0)
        rankings = np.arange(len(ranked_args))[ranked_args.argsort(axis=0)]
        rankings = rankings + 1
        avg_rankings = rankings.mean(axis=1)
        sd_rankings = np.sqrt(rankings.var(axis=1))

        # Summarize per method
        index = 0
        for item, value in methodologies.items():
            if value:
                table_H.append([item, evaluation_matrices['H_measure'][index, :].mean(),
                                np.sqrt(evaluation_matrices['H_measure'][index, :].var()), avg_rankings[index],
                                sd_rankings[index]])
                index += 1

        print(tabulate(table_H, headers="firstrow", floatfmt=("", ".4f", ".4f", ".4f", ".4f")))
        table_evaluation.append(table_H)

        if evaluators['stat_hypothesis_testing']:
            if n_methodologies < 2:
                raise Warning('"stat_hypothesis_testing" is True, but not enough methods are compare. At least 2 are needed')

            # Do tests if enough measurements are available (at least 3)
            if evaluation_matrices['H_measure'].shape[1] > 2:
                friedchisq = friedmanchisquare(*evaluation_matrices['H_measure'].T)
                print('\nH_measure score - Friedman test')
                print('H0: There are no significant differences between the models')
                print('\tChi-square:\t%.4f' % friedchisq[0])
                print('\tp-value:\t%.4f' % friedchisq[1])
                if friedchisq[1] < 0.05:  # If p-value is significant, do Nemenyi post hoc test
                    print('\t\t'+str(round(friedchisq[1],4)) +" < 0.05: at least one model differs significantly from the others")
                    nemenyi = posthoc_nemenyi_friedman(evaluation_matrices['H_measure'].T.astype(dtype=np.float32))
                    print('\nNemenyi post hoc test:')
                    print(nemenyi)
            elif evaluation_matrices['H_measure'].shape[1] < 3:
                raise Warning('"stat_hypothesis_testing" is True, but not enough measurements. At least 3 are needed')

        print('_________________________________________________________________________')

    if evaluators['brier']:
        table_brier = [['Method', 'Brier_score', 'Brier_score_sd', 'Brier_score_AR', 'Brier_score_AR_sd']]

        # Compute rankings (lower is better)
        ranked_args = (evaluation_matrices['brier']).argsort(axis=0)
        rankings = np.arange(len(ranked_args))[ranked_args.argsort(axis=0)]
        rankings = rankings + 1
        avg_rankings = rankings.mean(axis=1)
        sd_rankings = np.sqrt(rankings.var(axis=1))

        # Summarize per method
        index = 0
        for item, value in methodologies.items():
            if value:
                table_brier.append([item, evaluation_matrices['brier'][index, :].mean(),
                                    np.sqrt(evaluation_matrices['brier'][index, :].var()), avg_rankings[index],
                                    sd_rankings[index]])
                index += 1

        print(tabulate(table_brier, headers="firstrow", floatfmt=("", ".6f", ".6f", ".4f", ".4f")))
        table_evaluation.append(table_brier)

        if evaluators['stat_hypothesis_testing']:
            if n_methodologies < 2:
                raise Warning('"stat_hypothesis_testing" is True, but not enough methods are compare. At least 2 are needed')

            # Do tests if enough measurements are available (at least 3)
            if evaluation_matrices['brier'].shape[1] > 2:
                friedchisq = friedmanchisquare(*evaluation_matrices['brier'].T)
                print('\nBrier score - Friedman test')
                print('H0: There are no significant differences between the models')
                print('\tChi-square:\t%.4f' % friedchisq[0])
                print('\tp-value:\t%.4f' % friedchisq[1])
                if friedchisq[1] < 0.05:  # If p-value is significant, do Nemenyi post hoc test
                    print('\t\t'+str(round(friedchisq[1],4)) +" < 0.05: at least one model differs significantly from the others")
                    nemenyi = posthoc_nemenyi_friedman(evaluation_matrices['brier'].T.astype(dtype=np.float32))
                    print('\nNemenyi post hoc test:')
                    print(nemenyi)
            elif evaluation_matrices['brier'].shape[1] < 3:
                raise Warning('"stat_hypothesis_testing" is True, but not enough measurements. At least 3 are needed')
        print('_________________________________________________________________________')

    if evaluators['time']:

        table_time = [['Method', 'Time', 'Time_sd', 'Time_AR', 'Time_AR_sd']]

        # Compute rankings (lower is better)
        ranked_args = (evaluation_matrices['time']).argsort(axis=0)
        rankings = np.arange(len(ranked_args))[ranked_args.argsort(axis=0)]
        rankings = rankings + 1
        avg_rankings = rankings.mean(axis=1)
        sd_rankings = np.sqrt(rankings.var(axis=1))

        # Summarize per method
        index = 0
        for item, value in methodologies.items():
            if value:
                table_time.append([item, evaluation_matrices['time'][index, :].mean(),
                                    np.sqrt(evaluation_matrices['time'][index, :].var()), avg_rankings[index],
                                    sd_rankings[index]])
                index += 1

        print(tabulate(table_time, headers="firstrow", floatfmt=("", ".6f", ".6f", ".4f", ".4f")))
        table_evaluation.append(table_time)

        if evaluators['stat_hypothesis_testing']:
            if n_methodologies < 2:
                raise Warning('"stat_hypothesis_testing" is True, but not enough methods are compare. At least 2 are needed')

            # Do tests if enough measurements are available (at least 3)
            if evaluation_matrices['time'].shape[1] > 2:
                friedchisq = friedmanchisquare(*evaluation_matrices['time'].T)
                print('\nTime - Friedman test')
                print('H0: There are no significant differences between the models')
                print('\tChi-square:\t%.4f' % friedchisq[0])
                print('\tp-value:\t%.4f' % friedchisq[1])
                if friedchisq[1] < 0.05:  # If p-value is significant, do Nemenyi post hoc test
                    print('\t\t'+str(round(friedchisq[1],4)) +" < 0.05: at least one model differs significantly from the others")
                    nemenyi = posthoc_nemenyi_friedman(evaluation_matrices['time'].T.astype(dtype=np.float32))
                    print('\nNemenyi post hoc test:')
                    print(nemenyi)
            elif evaluation_matrices['time'].shape[1] < 3:
                raise Warning('"stat_hypothesis_testing" is True, but not enough measurements. At least 3 are needed')
        print('_________________________________________________________________________')


    table_evaluation_rounded = round_nested_list(table_evaluation) # Floats have max 4 decimals for readability

    #Add ('a': append) results of evaluators to the summary file
    with open(directory, 'a') as file:
        for i in table_evaluation_rounded:
            file.write(tabulate(i, floatfmt=".4f") + '\n')

    # Function to flatten a nested dictionary
    def flatten_dict(d, parent_key='', sep='_'):
        flat_dict = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                flat_dict.update(flatten_dict(v, new_key, sep=sep))
            else:
                flat_dict[new_key] = v
        return flat_dict

    result_dict = settings
    result_dict.update(dataset)
    result_dict.update(thresholding)

    for header_row, *data_rows in table_evaluation_rounded:
        for row in data_rows:
            method = row[0]
            for i in range(1, len(header_row)):
                key = header_row[i]
                value = row[i]
                if method not in result_dict:
                    result_dict[method] = {}
                result_dict[method][key] = value

    # Flatten the result_dict
    flat_result_dict = flatten_dict(result_dict)

    print(flat_result_dict)

    # Insert WandB here:
#    wandb.init(project='projectname', entity='entityname', reinit=False, save_code=True)
#    wandb.log(flat_result_dict)
#    wandb.finish()

    print('Summary of experiment setup and results written to '+directory)

