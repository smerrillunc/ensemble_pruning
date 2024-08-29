import pandas as pd
import numpy as np

import os, sys
from datetime import date
import random

sys.path.insert(0, '/nas/longleaf/home/smerrill/evolution/')

import pickle

from sklearn.tree import DecisionTreeClassifier
from joblib import dump, load

from scipy.stats import wasserstein_distance

import numpy as np
from scipy.spatial.distance import cosine, euclidean
import argparse

from utils import *
from fitness_functions import *
from voting_classifier import voting_classifier


def get_oob_fitness_metrics(model_pool_clf, indices):
    summary_dict = {}
    #### OOB ACCs ######    
    ensemble_oob_preds, ensemble_oob_pred_probs, oob_pred_counts = compute_ensemble_oob_preds(model_pool_clf, indices, x_train, y_train, feature_shift='oob', logistic_regression_train=logistic_regression_train, logistic_regression_val = logistic_regression_val)
    oob_acc, oob_prec, oob_rec, oob_auc, oob_percent_positive_preds = compute_metrics(ensemble_oob_preds, ensemble_oob_pred_probs, y_train)

    summary_dict['ensemble_oob_acc'] = oob_acc
    summary_dict['ensemble_oob_precision'] = oob_prec
    summary_dict['ensemble_oob_recall'] = oob_rec
    summary_dict['ensemble_oob_auc'] = oob_auc
    summary_dict['ensemble_oob_positive_pred_percent'] = oob_percent_positive_preds
 
    ##### FEATURE SHIFT OOB ####
    ensemble_oob_preds, ensemble_oob_pred_probs, oob_pred_counts = compute_ensemble_oob_preds(model_pool_clf, indices, x_train, y_train, feature_shift='feature_shift', logistic_regression_train=logistic_regression_train, logistic_regression_val = logistic_regression_val)
    oob_acc, oob_prec, oob_rec, oob_auc, oob_percent_positive_preds = compute_metrics(ensemble_oob_preds, ensemble_oob_pred_probs, y_train)

    summary_dict['ensemble_feature_shift_acc'] = oob_acc
    summary_dict['ensemble_feature_shift_precision'] = oob_prec
    summary_dict['ensemble_feature_shift_recall'] = oob_rec
    summary_dict['ensemble_feature_shift_auc'] = oob_auc
    summary_dict['ensemble_feature_shift_positive_pred_percent'] = oob_percent_positive_preds


    ##### Both OOB ####
    ensemble_oob_preds, ensemble_oob_pred_probs, oob_pred_counts = compute_ensemble_oob_preds(model_pool_clf, indices, x_train, y_train, feature_shift='both', logistic_regression_train=logistic_regression_train, logistic_regression_val = logistic_regression_val)
    oob_acc, oob_prec, oob_rec, oob_auc, oob_percent_positive_preds = compute_metrics(ensemble_oob_preds, ensemble_oob_pred_probs, y_train)

    summary_dict['ensemble_both_acc'] = oob_acc
    summary_dict['ensemble_both_precision'] = oob_prec
    summary_dict['ensemble_both_recall'] = oob_rec
    summary_dict['ensemble_both_auc'] = oob_auc
    summary_dict['ensemble_both_positive_pred_percent'] = oob_percent_positive_preds

    ### Logistic regression train
    ensemble_oob_preds, ensemble_oob_pred_probs, oob_pred_counts = compute_ensemble_oob_preds(model_pool_clf, indices, x_train, y_train, feature_shift='regression_train', logistic_regression_train=logistic_regression_train, logistic_regression_val = logistic_regression_val)
    oob_acc, oob_prec, oob_rec, oob_auc, oob_percent_positive_preds = compute_metrics(ensemble_oob_preds, ensemble_oob_pred_probs, y_train)

    summary_dict['ensemble_logisticR_train_acc'] = oob_acc
    summary_dict['ensemble_logisticR_train_precision'] = oob_prec
    summary_dict['ensemble_logisticR_train_recall'] = oob_rec
    summary_dict['ensemble_logisticR_train_auc'] = oob_auc
    summary_dict['ensemble_logisticR_train_positive_pred_percent'] = oob_percent_positive_preds

    ### Logistic regression Val
    ensemble_oob_preds, ensemble_oob_pred_probs, oob_pred_counts = compute_ensemble_oob_preds(model_pool_clf, indices, x_train, y_val_id, feature_shift='regression_val', logistic_regression_train=logistic_regression_train, logistic_regression_val = logistic_regression_val)
    oob_acc, oob_prec, oob_rec, oob_auc, oob_percent_positive_preds = compute_metrics(ensemble_oob_preds, ensemble_oob_pred_probs, y_val_id)

    summary_dict['ensemble_logisticR_val_acc'] = oob_acc
    summary_dict['ensemble_logisticR_val_precision'] = oob_prec
    summary_dict['ensemble_logisticR_val_recall'] = oob_rec
    summary_dict['ensemble_logisticR_val_auc'] = oob_auc
    summary_dict['ensemble_logisticR_val_positive_pred_percent'] = oob_percent_positive_preds
    return summary_dict


def get_ensemble_fitness_metrics(model_preds, model_pred_probs, ensemble_preds, ensemble_pred_probs, Y):
    summary_dict = {}
    # Ensemble confidences
    ensemble_confidences = ensemble_pred_probs.max(axis=1)
    model_confidences = model_pred_probs.max(axis=2)

    
    # Ensemble Accuracy
    acc, prec, rec, auc, percent_positive_preds = compute_metrics(ensemble_preds, ensemble_pred_probs, Y)

    summary_dict['ensemble_acc'] = acc
    summary_dict['ensemble_precision'] = prec
    summary_dict['ensemble_recall'] = rec
    summary_dict['ensemble_auc'] = auc
    summary_dict['ensemble_positive_pred_percent'] = percent_positive_preds
        
    summary_dict['ensemble_mean_confidence'] = ensemble_confidences.mean()
    summary_dict['ensemble_std_confidence'] = ensemble_confidences.std()
    summary_dict['ensemble_max_confidence'] = ensemble_confidences.max()
    summary_dict['ensemble_min_confidence'] = ensemble_confidences.min()

    # std adjusted
    summary_dict['ensemble_mean_std_confidence'] = ensemble_confidences.mean()/ensemble_confidences.std()
    summary_dict['ensemble_max_std_confidence'] = ensemble_confidences.max()/ensemble_confidences.std()
    summary_dict['ensemble_min_std_confidence'] = ensemble_confidences.min()/ensemble_confidences.std()

    
    ## Model-ensemble Level fitness functions
    acc_to_disagreement = acc_to_disagreement_fitness(model_preds, ensemble_preds, Y)
    correct_disagreement, incorrect_disagreement = disagreement_fitness(model_preds, ensemble_preds, Y)
    entropy = entropy_fitnes(model_preds, Y)
    kw = kw_variance_fitness(model_preds, Y)
    
    summary_dict['acc_to_disagreement'] = acc_to_disagreement
    summary_dict['correct_disagreement'] = correct_disagreement
    summary_dict['incorrect_disagreement'] = incorrect_disagreement
    summary_dict['entropy'] = entropy
    summary_dict['kw'] = kw

    # Model level fitness
    model_pred_metrics = np.array([compute_metrics(model_preds[x], model_pred_probs[x], Y) for x in range(len(model_preds))])
    means = model_pred_metrics.mean(axis=0)
    stds = model_pred_metrics.std(axis=0)
    mins = model_pred_metrics.min(axis=0)
    maxs = model_pred_metrics.max(axis=0)

    summary_dict['mean_model_acc'] = means[0]
    summary_dict['mean_model_precision'] = means[1]
    summary_dict['mean_model_recall'] = means[2]
    summary_dict['mean_model_auc'] = means[3]
    summary_dict['mean_model_percent_positive'] = means[4]
    
    summary_dict['min_model_acc'] = mins[0]
    summary_dict['min_model_precision'] = mins[1]
    summary_dict['min_model_recall'] = mins[2]
    summary_dict['min_model_auc'] = mins[3]
    summary_dict['min_model_percent_positive'] =mins[4]
    
    summary_dict['max_model_acc'] = maxs[0]
    summary_dict['max_model_precision'] = maxs[1]
    summary_dict['max_model_recall'] = maxs[2]
    summary_dict['max_model_auc'] = maxs[3]
    summary_dict['max_model_percent_positive'] =maxs[4]
    
    summary_dict['mean_model_acc_std'] = stds[0]
    summary_dict['mean_model_precision_std'] = stds[1]
    summary_dict['mean_model_recall_std'] = stds[2]
    summary_dict['mean_model_auc_std'] = stds[3]
    summary_dict['mean_model_percent_positive_std'] = stds[4]


    # std adjusted
    summary_dict['mean_std_model_acc'] = means[0]/stds[0]
    summary_dict['mean_std_model_precision'] = means[1]/stds[1]
    summary_dict['mean_std_model_recall'] = means[2]/stds[2]
    summary_dict['mean_std_model_auc'] = means[3]/stds[3]
    summary_dict['mean_std_model_percent_positive'] = means[4]/stds[4]
    
    summary_dict['min_std_model_acc'] = mins[0]/stds[0]
    summary_dict['min_std_model_precision'] = mins[1]/stds[1]
    summary_dict['min_std_model_recall'] = mins[2]/stds[2]
    summary_dict['min_std_model_auc'] = mins[3]/stds[3]
    summary_dict['min_std_model_percent_positive'] =mins[4]/stds[4]
    
    summary_dict['max_std_model_acc'] = maxs[0]/stds[0]
    summary_dict['max_std_model_precision'] = maxs[1]/stds[1]
    summary_dict['max_std_model_recall'] = maxs[2]/stds[2]
    summary_dict['max_std_model_auc'] = maxs[3]/stds[3]
    summary_dict['max_std_model_percent_positive'] =maxs[4]/stds[4]

    
    mean_confidence = model_confidences.mean(axis=1).mean()
    min_confidence = model_confidences.mean(axis=1).min()
    max_confidence = model_confidences.mean(axis=1).max()
    std_confidence = model_confidences.mean(axis=1).std()
    
    summary_dict['mean_model_confidence'] = mean_confidence
    summary_dict['min_model_confidence'] = min_confidence
    summary_dict['max_model_confidence'] = max_confidence
    summary_dict['model_confidence_std'] = std_confidence

    # std adjusted
    summary_dict['mean_std_model_confidence'] = mean_confidence/std_confidence

    # model level agreement measures
    agreement_avg, agreement_std, conf_agreement_avg, conf_agreement_std, \
    min_agreement, min_agreement_std, min_conf_agreement, min_conf_agreement_std = \
    get_agreement_measures(model_preds, model_pred_probs)
    
    
    summary_dict['mean_model_agreement'] = agreement_avg
    summary_dict['model_agreement_std'] = agreement_std

    summary_dict['mean_model_conf_agreement'] = conf_agreement_avg
    summary_dict['model_conf_agreement_std'] = conf_agreement_std
    
    summary_dict['min_model_agreement'] = min_agreement
    summary_dict['min_model_agreement_std'] = min_agreement_std
    summary_dict['min_model_conf_agreement'] = min_conf_agreement
    summary_dict['min_model_conf_agreement_std'] = min_conf_agreement_std

    # std adjusted
    summary_dict['mean_std_model_agreement'] = agreement_avg/agreement_std
    summary_dict['mean_std_model_conf_agreement'] = conf_agreement_avg/conf_agreement_std
    summary_dict['min_std_model_agreement'] = min_agreement/min_agreement_std
    summary_dict['min_std_model_conf_agreement'] = min_conf_agreement/min_conf_agreement_std

    # https://openreview.net/pdf/e5517ab786357cdf75bdd49c92ed293da740446e.pdf############################
    G = get_model_misclass_mat(model_preds, Y)
    mutual_misclassification = G.mean()
    mutual_misclassification_min = G.mean(axis=1).min()
    mutual_misclassification_max = G.mean(axis=1).max()
    mutual_misclassification_std = G.mean(axis=1).std()

    summary_dict['mutual_misclassification_mean'] = mutual_misclassification
    summary_dict['mutual_misclassification_min'] = mutual_misclassification_min
    summary_dict['mutual_misclassification_max'] = mutual_misclassification_max
    summary_dict['mutual_misclassification_std'] = mutual_misclassification_std

    # std adjusted
    summary_dict['mutual_misclassification_mean_std'] = mutual_misclassification/mutual_misclassification_std
    summary_dict['mutual_misclassification_min_std'] = mutual_misclassification_min/mutual_misclassification_std
    summary_dict['mutual_misclassification_max_std'] = mutual_misclassification_max/mutual_misclassification_std


    ###############################################################################################################
    
    ############################ DISTANCE METRICS ON CONFIDENCES####################################################
    
    emds = []
    mmds = []
    fids=[]
    if model_confidences.shape[1] > 100:
        idxs = np.random.choice(range(model_confidences.shape[1]), size=100)
    else:
        idxs = [x for x in range(model_confidences.shape[1])]

    model_confidences = model_confidences[:,idxs]
    model_preds = model_preds[:,idxs]

    for i in range(0, model_confidences.shape[0]):
        for j in range(i, model_confidences.shape[0]):
            # Earths mover distance
            emd = wasserstein_distance(model_confidences[i], model_confidences[j])
            emds.append(emd)

            # Maximum mean discrepancy (MMD) 
            mmd = compute_mmd(model_confidences[i].reshape(-1, 1), model_confidences[j].reshape(-1, 1))
            mmds.append(mmd)

            fid = fisher_information_distance(model_confidences[i], np.eye(model_confidences[i].shape[0]), model_confidences[j], np.eye(model_confidences[i].shape[0]))
            fids.append(fid)

    mean_emd = np.mean(emds)
    min_emd = np.min(emds)
    max_emd = np.max(emds)
    std_emd = np.std(emds)

    mean_mmd = np.mean(mmds)
    min_mmd = np.min(mmds)
    max_mmd = np.max(mmds)
    std_mmd = np.std(mmds)

    mean_fid = np.mean(fids)
    min_fid = np.min(fids)
    max_fid= np.max(fids)
    std_fid = np.std(fids)

    summary_dict['mean_emd'] = mean_emd
    summary_dict['min_emd'] = min_emd
    summary_dict['max_emd'] = max_emd
    summary_dict['std_emd'] = std_emd

    # std adjuste
    summary_dict['mean_std_emd'] = mean_emd/std_emd
    summary_dict['min_std_emd'] = min_emd/std_emd
    summary_dict['max_std_emd'] = max_emd/std_emd


    summary_dict['mean_mmd'] = mean_mmd
    summary_dict['min_mmd'] = min_mmd
    summary_dict['max_mmd'] = max_mmd
    summary_dict['std_mmd'] = std_mmd

    # std adjust
    summary_dict['mean_std_mmd'] = mean_mmd/std_mmd
    summary_dict['min_std_mmd'] = min_mmd/std_mmd
    summary_dict['max_std_mmd'] = max_mmd/std_mmd


    summary_dict['mean_fid'] = mean_fid
    summary_dict['min_fid'] = min_fid
    summary_dict['max_fid'] = max_fid
    summary_dict['std_fid'] = std_fid

    # std adjust
    summary_dict['mean_std_fid'] = mean_fid/std_fid
    summary_dict['min_std_fid'] = min_fid/std_fid
    summary_dict['max_std_fid'] = max_fid/std_fid


    ############################ DISTANCE METRICS ON BINARY PREDICTIONS####################################################
    cosines = []
    hams = []
    eucs =[]
    for i in range(0, model_preds.shape[1]):
        for j in range(i, model_preds.shape[0]):
            cos = cosine(model_preds[i], model_preds[j])
            ham = hamming_distance(model_preds[i], model_preds[j])
            euc = euclidean(model_preds[i], model_preds[j])

            cosines.append(cos)
            hams.append(ham)
            eucs.append(euc)

    mean_cos = np.mean(cosines)
    min_cos = np.min(cosines)
    max_cos = np.max(cosines)
    std_cos = np.std(cosines)

    mean_ham = np.mean(hams)
    min_ham = np.min(hams)
    max_ham = np.max(hams)
    std_ham = np.std(hams)

    mean_euc = np.mean(eucs)
    min_euc = np.min(eucs)
    max_euc= np.max(eucs)
    std_euc = np.std(eucs)

    summary_dict['mean_cos'] = mean_cos
    summary_dict['min_cos'] = min_cos
    summary_dict['max_cos'] = max_cos
    summary_dict['std_cos'] = std_cos

    # std adjust
    summary_dict['mean_std_cos'] = mean_cos/std_cos
    summary_dict['min_std_cos'] = min_cos/std_cos
    summary_dict['max_std_cos'] = max_cos/std_cos


    summary_dict['mean_ham'] = mean_ham
    summary_dict['min_ham'] = min_ham
    summary_dict['max_ham'] = max_ham
    summary_dict['std_ham'] = std_ham

    # std adjust
    summary_dict['mean_std_ham'] = mean_ham/std_ham
    summary_dict['min_std_ham'] = min_ham/std_ham
    summary_dict['max_std_ham'] = max_ham/std_ham

    summary_dict['mean_euc'] = mean_euc
    summary_dict['min_euc'] = min_euc
    summary_dict['max_euc'] = max_euc
    summary_dict['std_euc'] = std_euc

    # std adjust
    summary_dict['mean_std_euc'] = mean_euc/std_euc
    summary_dict['min_std_euc'] = min_euc/std_euc
    summary_dict['max_std_euc'] = max_euc/std_euc

    
    return summary_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Optional parameters for your script")

    parser.add_argument("--dataset_path", type=str, default="/Users/scottmerrill/Documents/UNC/Research/OOD-Ensembles/datasets", help="path to dataset")
    parser.add_argument("--dataset", type=str, default="heloc_tf", help="Dataset Name")
    parser.add_argument("--ntrls", type=int, default=100000, help ="Number of random search trials")
    parser.add_argument("--max_ensemble_size", type=int, default=250, help ="Number of random search trials")
    parser.add_argument("--model_pool_size", type=int, default=10000, help ="Model Pool Size")

    parser.add_argument("--model_pool_path", type=str, default=None, help ="Optional model pool to consider")

    # admin params    
    parser.add_argument("--save_name", type=str, default=None, help="Save Name")
    parser.add_argument("--save_path", type=str, default="/Users/scottmerrill/Desktop", help="Save Path")

    args = vars(parser.parse_args())

    # save paths
    if args['save_name'] == None:
        args['save_name'] = date.today().strftime('%Y%m%d')

    save_path = create_directory_if_not_exists(args['save_path'] + '/exps/{}/{}'.format(args['save_name'], args['dataset'])) 
    save_dict_to_file(args, save_path + '/experiment_args.txt')  
    
    # datasets
    x_train, y_train, x_val_id, y_val_id, x_val_ood, y_val_ood = get_dataset(args['dataset_path'], args['dataset'])

    # RANDOM SEARCH ENSEMBLES   
    num_features = x_train.shape[1]

    model = DecisionTreeClassifier
    training_fracs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    num_models = args['model_pool_size']

    if args['model_pool_path'] == None:
        model_pool = []
        os.makedirs(save_path+'/models')
        for i in range(int(num_models//(len(training_fracs)*(num_features-2)))):
            for training_frac in training_fracs:
                for feature_ones in range(2, num_features):
                    print(i, training_frac)
                    model_params = {'max_depth':np.random.randint(1, 10),
                       'class_weight':'balanced',
                        'min_samples_leaf':np.random.randint(3, 10),
                        'splitter':np.random.choice(['best', 'random'])}


                    seed = np.random.randint(0, 99999999)
                    estimator = model(**model_params, random_state=seed)

                    estimator.feature_ones = feature_ones
                    estimator.training_frac = training_frac
                    estimator, file_name = train_model(estimator, x_train, y_train, training_frac, save_path)

                    model_pool.append(estimator)
    else:
        print("Loding Modle Pool")
        all_models = os.listdir(args['model_pool_path'])
        model_pool = [load_model(args['model_pool_path'] + f'/{x}') for x in all_models]


    x_train_noise = random_noise(x_train, 0.25)
    x_val_id_noise = random_noise(x_val_id, 0.25)
    x_val_ood_noise = random_noise(x_val_ood, 0.25)

    x_train_corrupt = scarf_corruptions(x_train, 0.25)
    x_val_id_corrupt = scarf_corruptions(x_val_id, 0.25)
    x_val_ood_corrupt = scarf_corruptions(x_val_ood, 0.25)


    # store these in model pool as attributes
    model_pool_clf = voting_classifier(model_pool, num_features)

    # precompute all of these predictions and save in memory
    model_pool_train_pred_probs = np.array(model_pool_clf.get_model_pred_probs(x_train))
    model_pool_val_pred_probs = np.array(model_pool_clf.get_model_pred_probs(x_val_id))

    model_pool_train_noise_pred_probs = np.array(model_pool_clf.get_model_pred_probs(x_train_noise))
    model_pool_val_noise_pred_probs = np.array(model_pool_clf.get_model_pred_probs(x_val_id_noise))

    model_pool_train_corrupt_pred_probs = np.array(model_pool_clf.get_model_pred_probs(x_train_corrupt))
    model_pool_val_corrupt_pred_probs = np.array(model_pool_clf.get_model_pred_probs(x_val_id_corrupt))

    model_pool_ood_pred_probs = np.array(model_pool_clf.get_model_pred_probs(x_val_ood))

    # save them as attributes
    model_pool_clf.train_pred_probs = model_pool_train_pred_probs
    model_pool_clf.val_pred_probs = model_pool_val_pred_probs

    model_pool_clf.train_noise_pred_probs = model_pool_train_noise_pred_probs
    model_pool_clf.val_noise_pred_probs = model_pool_val_noise_pred_probs

    model_pool_clf.train_corrupt_pred_probs = model_pool_train_corrupt_pred_probs
    model_pool_clf.val_corrupt_pred_probs = model_pool_val_corrupt_pred_probs

    model_pool_clf.val_ood_pred_probs = model_pool_ood_pred_probs

    # OOD Accuracy computations
    AUCTHRESHS = [0.1, 0.2, 0.3, 0.4, 1. ]

    model_pool_preds = model_pool_clf.predict(x_val_ood)
    model_pool_pred_probs = model_pool_clf.predict_proba(x_val_ood)
    precision, recall, auc = get_precision_recall_auc(model_pool_pred_probs, y_val_ood, AUCTHRESHS)
    
    with open(save_path + f'/model_pool_precision.pkl', 'wb') as file:
        pickle.dump(precision, file)

    with open(save_path + f'/model_pool_recall.pkl', 'wb') as file:
        pickle.dump(recall, file)

    with open(save_path + f'/model_pool_auc.pkl', 'wb') as file:
        pickle.dump(auc, file)


    results_df = pd.DataFrame()
    precisions_df = pd.DataFrame()
    recalls_df = pd.DataFrame()
    aucs_df = pd.DataFrame()

    # logistic regression oob samples
    logistic_regression_train = get_oob_index_LR(x_train, y_train, x_train, percentile=25)
    logistic_regression_val = get_oob_index_LR(x_train, y_train, x_val_id, percentile=25)

    best_error = float('inf')
    for i in range(args['ntrls']):
        print(i)
        ensemble_size = np.random.randint(10, args['max_ensemble_size'])
        print(model_pool_clf.ensemble.shape[0], ensemble_size)
        indices = np.random.choice(model_pool_clf.ensemble.shape[0], size=ensemble_size, replace=True)

        # compute OOD metrics
        pruned_ensemble_ood_preds, pruned_ensemble_ood_pred_probs = get_ensemble_preds_from_models(model_pool_clf.val_ood_pred_probs[indices])
        error = -np.mean(auprc_threshs(pruned_ensemble_ood_pred_probs.max(axis=1), y_val_ood, AUCTHRESHS))
        precision, recall, auc = get_precision_recall_auc(pruned_ensemble_ood_pred_probs, y_val_ood, AUCTHRESHS)

        # Get summary stats
        description_dict = get_ensemble_description(model_pool_clf, indices)


        model_pool_clf.oob_pred_probs = model_pool_train_pred_probs
        oob_metrics_train_dict = get_oob_fitness_metrics(model_pool_clf, indices)
        oob_metrics_train_dict = {key + '_train' : value for key, value in oob_metrics_train_dict.items()}

        model_pool_clf.oob_pred_probs = model_pool_train_noise_pred_probs
        oob_metrics_train_noise_dict = get_oob_fitness_metrics(model_pool_clf, indices)
        oob_metrics_train_noise_dict = {key + '_train_noise' : value for key, value in oob_metrics_train_noise_dict.items()}

        model_pool_clf.oob_pred_probs = model_pool_train_corrupt_pred_probs
        oob_metrics_train_corrupt_dict = get_oob_fitness_metrics(model_pool_clf, indices)
        oob_metrics_train_corrupt_dict = {key + '_train_corrupt' : value for key, value in oob_metrics_train_corrupt_dict.items()}

        # train fitness_dict #####################################################################
        pruned_model_pred_probs = model_pool_clf.train_pred_probs[indices]

        pruned_model_preds = pruned_model_pred_probs.argmax(axis=2)
        pruned_ensemble_preds, pruned_ensemble_pred_probs = get_ensemble_preds_from_models(pruned_model_pred_probs)
        train_dict = get_ensemble_fitness_metrics(pruned_model_preds, 
                                                  pruned_model_pred_probs, 
                                                  pruned_ensemble_preds, 
                                                  pruned_ensemble_pred_probs, 
                                                  y_train)
        train_dict = {key + '_train' : value for key, value in train_dict.items()}

        ###########################################################################################


        # train noise fitness_dict ##################################################################
        pruned_model_pred_probs = model_pool_clf.train_noise_pred_probs[indices]
        pruned_model_preds = pruned_model_pred_probs.argmax(axis=2)
        pruned_ensemble_preds, pruned_ensemble_pred_probs = get_ensemble_preds_from_models(pruned_model_pred_probs)
        train_noise_dict = get_ensemble_fitness_metrics(pruned_model_preds, 
                                                  pruned_model_pred_probs, 
                                                  pruned_ensemble_preds, 
                                                  pruned_ensemble_pred_probs, 
                                                  y_train)
        train_noise_dict = {key + '_train_noise' : value for key, value in train_noise_dict.items()}
        ###########################################################################################

        # train corrupt fitness_dict ##################################################################
        pruned_model_pred_probs = model_pool_clf.train_corrupt_pred_probs[indices]
        pruned_model_preds = pruned_model_pred_probs.argmax(axis=2)
        pruned_ensemble_preds, pruned_ensemble_pred_probs = get_ensemble_preds_from_models(pruned_model_pred_probs)
        train_corrupt_dict = get_ensemble_fitness_metrics(pruned_model_preds, 
                                                  pruned_model_pred_probs, 
                                                  pruned_ensemble_preds, 
                                                  pruned_ensemble_pred_probs, 
                                                  y_train)
        train_corrupt_dict = {key + '_train_corrupt' : value for key, value in train_corrupt_dict.items()}

        ###########################################################################################

        # val fitness_dict #####################################################################
        pruned_model_pred_probs = model_pool_clf.val_pred_probs[indices]
        pruned_model_preds = pruned_model_pred_probs.argmax(axis=2)
        pruned_ensemble_preds, pruned_ensemble_pred_probs = get_ensemble_preds_from_models(pruned_model_pred_probs)
        val_dict = get_ensemble_fitness_metrics(pruned_model_preds, 
                                                  pruned_model_pred_probs, 
                                                  pruned_ensemble_preds, 
                                                  pruned_ensemble_pred_probs, 
                                                  y_val_id)
        val_dict = {key + '_val_id_noise' : value for key, value in val_dict.items()}
        ###########################################################################################

        # val noise dict #####################################################################
        pruned_model_pred_probs = model_pool_clf.val_noise_pred_probs[indices]
        pruned_model_preds = pruned_model_pred_probs.argmax(axis=2)
        pruned_ensemble_preds, pruned_ensemble_pred_probs = get_ensemble_preds_from_models(pruned_model_pred_probs)
        val_noise_dict = get_ensemble_fitness_metrics(pruned_model_preds, 
                                                  pruned_model_pred_probs, 
                                                  pruned_ensemble_preds, 
                                                  pruned_ensemble_pred_probs, 
                                                  y_val_id)
        val_noise_dict = {key + '_val_id_noise' : value for key, value in val_noise_dict.items()}

        ###########################################################################################

        # val corrupt dict #####################################################################
        pruned_model_pred_probs = model_pool_clf.val_corrupt_pred_probs[indices]
        pruned_model_preds = pruned_model_pred_probs.argmax(axis=2)
        pruned_ensemble_preds, pruned_ensemble_pred_probs = get_ensemble_preds_from_models(pruned_model_pred_probs)
        val_corrupt_dict = get_ensemble_fitness_metrics(pruned_model_preds, 
                                                  pruned_model_pred_probs, 
                                                  pruned_ensemble_preds, 
                                                  pruned_ensemble_pred_probs, 
                                                  y_val_id)
        val_corrupt_dict = {key + '_val_id_corrupt' : value for key, value in val_corrupt_dict.items()}
        ###########################################################################################

        description_dict.update(oob_metrics_train_dict)
        description_dict.update(oob_metrics_train_noise_dict)
        description_dict.update(oob_metrics_train_corrupt_dict)
        description_dict.update(train_dict)
        description_dict.update(train_noise_dict)
        description_dict.update(train_corrupt_dict)

        description_dict.update(val_dict)
        description_dict.update(val_noise_dict)
        description_dict.update(val_corrupt_dict)

        description_dict['error'] = error
        description_dict['ensemble_size'] = ensemble_size
        
        results_df = pd.concat([results_df, pd.DataFrame([description_dict])])

        recalls_df = pd.concat([recalls_df, pd.DataFrame(recall)], axis=1)
        precisions_df = pd.concat([precisions_df, pd.DataFrame(precision)], axis=1)
        aucs_df = pd.concat([aucs_df, pd.DataFrame(auc)], axis=1)


        recalls_df.to_csv(save_path + '/recalls_df.csv')
        results_df.to_csv(save_path + '/results_df.csv')
        precisions_df.to_csv(save_path + '/precisions_df.csv')
        aucs_df.to_csv(save_path + '/aucs_df.csv')

