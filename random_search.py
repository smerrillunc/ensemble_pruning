import pandas as pd
import numpy as np

import os, sys
from datetime import date
import random

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Optional parameters for your script")

    parser.add_argument("--dataset_path", type=str, default="/Users/scottmerrill/Documents/UNC/Research/OOD-Ensembles/datasets", help="path to dataset")
    parser.add_argument("--dataset", type=str, default="heloc_tf", help="Dataset Name")
    
    parser.add_argument("--model_pool_size", type=int, default=200, help ="Model Pool Size")
    parser.add_argument("--ensemble_size", type=int, default=25,help ="Ensemble Size")
    parser.add_argument("--ntrls", type=int, default=1000,help ="number of trials of random search")
    
    # admin params    
    parser.add_argument("--model_pool_path", type=str, default=None, help ="Optional model pool to consider")
    parser.add_argument("--seed", type=int, default=1,help ="Seed")
    parser.add_argument("--save_name", type=str, default=None, help="Save Name")
    parser.add_argument("--save_path", type=str, default="/Users/scottmerrill/Desktop", help="Save Path")

    args = vars(parser.parse_args())

    # seeds
    np.random.seed(args['seed'])
    random.seed(args['seed'])
    rng = np.random.RandomState(args['seed'])
    # save paths

    if args['save_name'] == None:
        args['save_name'] = date.today().strftime('%Y%m%d')

    save_path = create_directory_if_not_exists(args['save_path'] + '/exps/{}/{}'.format(args['save_name'], args['dataset'])) 
    save_dict_to_file(args, save_path + '/experiment_args.txt')  

    # datasets
    x_train, y_train, x_val_id, y_val_id, x_val_ood, y_val_ood = get_dataset(args['dataset_path'], args['dataset'])
    num_features = x_train.shape[1]

    if args['model_pool_path'] == None:
        model = DecisionTreeClassifier
        training_fracs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        num_models = args['model_pool_size']

        model_pool = []
        all_models = []

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
                    estimator, filename = train_model(estimator, x_train, y_train, training_frac, save_path)

                    model_pool.append(estimator)
                    all_models.append(filename)
    else:
        print("Loding Model Pool")
        all_models = os.listdir(args['model_pool_path'])
        model_pool = [load_model(args['model_pool_path'] + f'/{x}') for x in all_models]

    # save this file mapping
    model_file_map = {x:all_models[x] for x in range(len(all_models))}
    with open(save_path + f'/model_file_map.pkl', 'wb') as file:
        pickle.dump(model_file_map, file)

    # store these in model pool as attributes
    model_pool_clf = voting_classifier(model_pool, num_features)
    model_pool_ood_pred_probs = np.array(model_pool_clf.get_model_pred_probs(x_val_ood))

    # save them as attributes
    model_pool_clf.val_ood_pred_probs = model_pool_ood_pred_probs

    AUCTHRESHS = np.array([0.1, 0.2, 0.3, 0.4, 1. ])
    model_pool_preds = model_pool_clf.predict(x_val_ood)
    model_pool_pred_probs = model_pool_clf.predict_proba(x_val_ood)
    precision, recall, auc = get_precision_recall_auc(model_pool_pred_probs, y_val_ood, AUCTHRESHS)
    
    with open(save_path + f'/model_pool_precision.pkl', 'wb') as file:
        pickle.dump(precision, file)

    with open(save_path + f'/model_pool_recall.pkl', 'wb') as file:
        pickle.dump(recall, file)

    with open(save_path + f'/model_pool_auc.pkl', 'wb') as file:
        pickle.dump(auc, file)

    ####################################################
    #################### Starting Script ##################
    ####################################################
    print("Starting Script")
    
    precisions_df = pd.DataFrame()
    recalls_df = pd.DataFrame()
    aucs_df = pd.DataFrame()

    for trial in range(args['ntrls']):
        print(f"Starting Trial {trial}")

        indices = np.random.choice(model_pool_clf.ensemble.shape[0], size=args['ensemble_size'], replace=True)

        tmp = {'generation':trial,
               'ensemble_files':','.join(str(x) for x in indices)}

        ood_preds, ood_pred_probs = get_ensemble_preds_from_models(model_pool_clf.val_ood_pred_probs[indices])
        
        # save dfs
        precision, recall, auc = get_precision_recall_auc(ood_pred_probs, y_val_ood, AUCTHRESHS)
        print(recall)
        recalls_df = pd.concat([recalls_df, pd.DataFrame(recall)], axis=1)
        precisions_df = pd.concat([precisions_df, pd.DataFrame(precision)], axis=1)
        aucs_df = pd.concat([aucs_df, pd.DataFrame(auc)], axis=1)
    
        precisions_df.to_csv(save_path+'/precisions_df.csv', index=False)
        recalls_df.to_csv(save_path+'/recalls_df.csv', index=False)
        aucs_df.to_csv(save_path+'/aucs_df.csv', index=False)
