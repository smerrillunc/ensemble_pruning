import pandas as pd
import numpy as np

import sys, os
import random
import numbers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.integrate import trapezoid
from joblib import load, dump
from time import sleep


def get_dataset(dataset_path, dataset_name):
    """
    Description: Retrieve a chosen dataset
    """

    table_shift = ["diabetes_readmission",
                    "anes",
                    "assistments",
                    "nhanes_lead",
                    "college_scorecard",
                    "brfss_diabetes",
                    "acsfoodstamps",
                    "heloc",
                    "brfss_blood_pressure",
                    "mimic_extract_los_3",
                    "mimic_extract_mort_hosp",
                    "acsincome",
                    "acspubcov",
                    "physionet",
                    "acsunemployment",
                    'compas',
                    'german']

    if dataset_name == "CHEM":
        with open(f'{dataset_path}/CHEMOOD/train.csv', 'r') as f:
            X = np.float32(np.array([line.strip().split(',')[2:] for line in f])[1:])
        with open(f'{dataset_path}/CHEMOOD/train.csv', 'r') as f:
            Y = np.float32(np.array([line.strip().split(',')[1] for line in f])[1:])

        with open(f'{dataset_path}/CHEMOOD/val_id.csv', 'r') as f:
            X_val = np.float32(np.array([line.strip().split(',')[2:] for line in f])[1:])

        with open(f'{dataset_path}/CHEMOOD/val_id.csv', 'r') as f:
            Y_val = np.float32(np.array([line.strip().split(',')[1] for line in f])[1:])

        with open(f'{dataset_path}/CHEMOOD/val_ood.csv', 'r') as f:
            X_val_ood = np.float32(np.array([line.strip().split(',')[2:] for line in f])[1:])

        with open(f'{dataset_path}/CHEMOOD/val_ood.csv', 'r') as f:
            Y_val_ood = np.float32(np.array([line.strip().split(',')[1] for line in f])[1:])

        # with open(f'{dataset_path}/CHEMOOD/test_id.csv', 'r') as f:
        #    X_test = torch.from_numpy(np.float32(np.array([line.strip().split(',')[2:] for line in f])[1:]))

        # with open(f'{dataset_path}/CHEMOOD/test_id.csv', 'r') as f:
        #    Y_test = torch.from_numpy(np.float32(np.array([line.strip().split(',')[1] for line in f])[1:]))

        #with open(f'{dataset_path}/CHEMOOD/test_ood.csv', 'r') as f:
        #    X_test_ood = torch.from_numpy(np.float32(np.array([line.strip().split(',')[2:] for line in f])[1:]))

        #with open(f'{dataset_path}/CHEMOOD/test_ood.csv', 'r') as f:
        #    Y_test_ood = torch.from_numpy(np.float32(np.array([line.strip().split(',')[1] for line in f])[1:]))


    elif dataset_name in ['adult_tf', 'heloc_tf', 'yeast_tf', 'synthetic', 'hosptial_tf']:
        X = np.load(f'{dataset_path}/{dataset_name}/xs_train.npy')
        Y = np.load(f'{dataset_path}/{dataset_name}/ys_train.npy')

        X_val = np.load(f'{dataset_path}/{dataset_name}/xs_val_id.npy')
        Y_val = np.load(f'{dataset_path}/{dataset_name}/ys_val_id.npy')

        X_val_ood = np.load(f'{dataset_path}/{dataset_name}/xs_val_ood.npy')
        Y_val_ood = np.load(f'{dataset_path}/{dataset_name}/ys_val_ood.npy')

    elif dataset_name in table_shift:
        X = np.loadtxt(f'{dataset_path}/{dataset_name}/xs_train.csv', delimiter=',', skiprows=1)
        Y = np.loadtxt(f'{dataset_path}/{dataset_name}/ys_train.csv', delimiter=',', skiprows=1)

        X_val = np.loadtxt(f'{dataset_path}/{dataset_name}/xs_val_id.csv', delimiter=',', skiprows=1)
        Y_val = np.loadtxt(f'{dataset_path}/{dataset_name}/ys_val_id.csv', delimiter=',', skiprows=1)

        X_val_ood = np.loadtxt(f'{dataset_path}/{dataset_name}/xs_val_ood.csv', delimiter=',', skiprows=1)
        Y_val_ood = np.loadtxt(f'{dataset_path}/{dataset_name}/ys_val_ood.csv', delimiter=',', skiprows=1)

    else:
        print("Please specify a valid dataset and re-run the script")
        return 0

    return X, Y.ravel(), X_val, Y_val.ravel(), X_val_ood, Y_val_ood.ravel()


def create_directory_if_not_exists(directory):
    """
    Description: Create a new file if one doesn't exist
    """

    i = 0
    while True:
        try:
            if not os.path.exists(directory + f'/{i}'):
                os.makedirs(directory + f'/{i}')
                print(f"Directory '{directory + f'/{i}'}' created.")
                return directory + f'/{i}'
            else:
                print(f"Directory '{directory + f'/{i}'}' already exists.")
                i += 1

        except Exception as e:
            sleep(3)

def save_dict_to_file(dictionary, filename):
    """
    Description: Save a dictionary to a file
    """

    with open(filename, 'w') as file:
        for key, value in dictionary.items():
            file.write(f"{key}: {value}\n")


def parse_config_file(file_path):
    """
    Parses a configuration file and returns a dictionary of key-value pairs.
    
    Args:
        file_path (str): The path to the configuration file.
    
    Returns:
        dict: A dictionary containing the configuration parameters.
    """
    config_dict = {}
    
    with open(file_path, 'r') as file:
        for line in file:
            # Split each line into key and value
            key, value = line.strip().split(': ')
            # Add to the dictionary
            config_dict[key] = value
    return config_dict


def load_model(filename):
    loaded_clf = load(filename)
    return loaded_clf



def train_model(estimator, x_train, y_train, training_frac, save_path):
    num_features = x_train.shape[1]
    filename = f"{save_path}/models/{estimator.random_state}_{estimator.feature_ones}_{str(training_frac)}.json"

    features_encoding = generate_binary_vector(num_features, estimator.feature_ones, seed=estimator.random_state)

    sampled_indices, unsampled_indices = generate_sample_indices(estimator.random_state, 
                                                                 x_train.shape[0],
                                                                 training_frac)
    
    estimator.fit(x_train[sampled_indices][:,np.where(features_encoding==1)[0]], y_train[sampled_indices])
    
    # save estimator
    dump(estimator, filename, compress=9)

    return estimator, filename

def generate_sample_indices(random_state, n_samples, training_frac=0.1):
    """
    Private function used to _parallel_build_trees function."""

    random_instance = check_random_state(random_state)
    sample_indices = random_instance.randint(0, n_samples, int(training_frac*n_samples), dtype=np.int32)
    
    sample_counts = np.bincount(sample_indices, minlength=n_samples)
    unsampled_mask = sample_counts == 0
    indices_range = np.arange(n_samples)
    unsampled_indices = indices_range[unsampled_mask]

    return sample_indices, unsampled_indices

    
def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    Returns
    -------
    :class:`numpy:numpy.random.RandomState`
        The random state object based on `seed` parameter.

    Examples
    --------
    >>> from sklearn.utils.validation import check_random_state
    >>> check_random_state(42)
    RandomState(MT19937) at 0x...
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState instance" % seed
    )
    
def generate_binary_vector(length, num_ones, seed):
    if num_ones > length:
        raise ValueError("Number of zeros cannot exceed vector length.")
    
    # Initialize random seed if provided
    if seed is not None:
        random.seed(seed)
    
    # Create a list of 1s and 0s
    vector = [1] * num_ones + [0] * (length-num_ones)
    
    # Shuffle the vector to randomize positions of 1s and 0s
    random.shuffle(vector)
    
    return np.array(vector)

def random_noise(x, std_mult=0.25):
    noise = np.zeros_like(x)

    # Loop through each column to calculate the standard deviation and generate noise
    for i in range(x.shape[1]):
        column_std = np.std(x[:, i])  # Calculate the standard deviation of the column
        noise[:, i] = np.random.normal(0, std_mult*column_std, x.shape[0])  # Generate noise for the column
    perturbed_data = x + noise
    
    return perturbed_data

def scarf_corruptions(x, corruption_rate):
    # 1: create a mask of for each sample we set the jth column to True at random,
    # such that corruption_len / m = corruption_rate
    corruption_mask = np.random.rand(*x.shape) > corruption_rate

    # 2: create a random tensor of size x drawn from the uniform distribution defined 
    # by the min, max values of the training set
    features_low = x.min(axis=0)
    features_high = x.max(axis=0)
    x_random = np.random.uniform(features_low, features_high, x.shape)

    # 3: replace x_corrupted_ij by x_random_ij where mask_ij is true
    x_corrupted = np.where(corruption_mask, x_random, x)
    
    return x_corrupted


def get_ensemble_description(model_pool_clf, indices):
    
    feature_pcts = model_pool_clf.feature_encodings[indices].mean(axis=0)
    average_features_used = model_pool_clf.feature_encodings[indices].mean()
    std_features_used = model_pool_clf.feature_encodings[indices].std()

    summary_dict = {f'Feature_{x}': feature_pcts[x] for x in range(len(feature_pcts))}
    summary_dict['average_features'] = average_features_used
    summary_dict['std_features_used'] = std_features_used
    
    avg_training_data = np.mean(model_pool_clf.train_info[indices])
    min_training_data = np.min(model_pool_clf.train_info[indices])
    max_training_data = np.max(model_pool_clf.train_info[indices])
    std_training_data = np.std(model_pool_clf.train_info[indices])
    median_training_data = np.median(model_pool_clf.train_info[indices])

    avg_depth = np.mean(model_pool_clf.max_depths[indices])
    min_depth = np.min(model_pool_clf.max_depths[indices])
    max_depth = np.max(model_pool_clf.max_depths[indices])
    std_depth = np.std(model_pool_clf.max_depths[indices])
    median_depth = np.median(model_pool_clf.max_depths[indices])

    avg_leaf_samples = np.mean(model_pool_clf.min_samples_leafs[indices])
    min_leaf_samples = np.min(model_pool_clf.min_samples_leafs[indices])
    max_leaf_samples = np.max(model_pool_clf.min_samples_leafs[indices])
    std_leaf_samples = np.std(model_pool_clf.min_samples_leafs[indices])
    median_leaf_samples = np.median(model_pool_clf.min_samples_leafs[indices])

    random_splitters_frac = np.mean(model_pool_clf.splitters[indices] == 'random')
    best_splitters_frac = np.mean(model_pool_clf.splitters[indices] == 'best')

    # train data
    summary_dict['mean_training_data'] = average_features_used
    summary_dict['min_training_data'] = min_training_data
    summary_dict['max_training_data'] = max_training_data
    summary_dict['std_training_data'] = std_training_data
    summary_dict['median_training_data'] = median_training_data

    # std adjusted
    summary_dict['mean_std_training_data'] = average_features_used/std_training_data
    summary_dict['min_std_training_data'] = min_training_data/std_training_data
    summary_dict['max_std_training_data'] = max_training_data/std_training_data
    summary_dict['median_std_training_data'] = median_training_data/std_training_data


    # depth
    summary_dict['mean_depth'] = average_features_used
    summary_dict['min_depth'] = min_training_data
    summary_dict['max_depth'] = max_training_data
    summary_dict['std_depth'] = std_training_data
    summary_dict['median_depth'] = median_training_data

    # std adjusted
    summary_dict['mean_std_depth'] = average_features_used/std_training_data
    summary_dict['min_std_depth'] = min_training_data/std_training_data
    summary_dict['max_std_depth'] = max_training_data/std_training_data
    summary_dict['median_std_depth'] = median_training_data/std_training_data
    
    # depth
    summary_dict['mean_leaf_samples'] = avg_leaf_samples
    summary_dict['min_leaf_samples'] = min_leaf_samples
    summary_dict['max_leaf_samples'] = max_leaf_samples
    summary_dict['std_leaf_samples'] = std_leaf_samples
    summary_dict['median_leaf_samples'] = median_leaf_samples

    # std adjusted
    summary_dict['mean_std_leaf_samples'] = avg_leaf_samples/std_leaf_samples
    summary_dict['min_std_leaf_samples'] = min_leaf_samples/std_leaf_samples
    summary_dict['max_std_leaf_samples'] = max_leaf_samples/std_leaf_samples
    summary_dict['median_std_leaf_samples'] = median_leaf_samples/std_leaf_samples


    num_dt = 0
    num_xgb = 0
    for clf in model_pool_clf.ensemble[indices]:
        if isinstance(clf, DecisionTreeClassifier):
            num_dt += 1
        else:
            num_xgb += 1
            
    summary_dict['ensemble_size'] = len(indices)
    summary_dict['num_dt'] = num_dt
    summary_dict['num_xgb'] = num_xgb
    return summary_dict

def get_feature_shift_oob(x_train, y_train, feature_encoding, bottom_percent=10, random=False):
   
    # select a random feature
    if random:
        shift_feature = np.random.choice(np.where(feature_encoding==0)[0])
    else:
        shift_features = np.where(feature_encoding==0)[0]
        shift_feature = shift_features[0]
        shift_corr = abs(np.corrcoef(x_train[:,shift_feature], y_train)[0, 1])

        for i in range(1, len(shift_features)):
            corr = abs(np.corrcoef(x_train[:,shift_features[i]], y_train)[0, 1])
            if corr > shift_corr:
                shift_corr = corr
                shift_feature = shift_features[i]
        
    # determine unique values
    unique_values = np.unique(x_train[:,shift_feature])
    num_unique_values = len(unique_values)
    total_values = len(x_train[:,shift_feature])

    # determin if it's categorical or numeric and get oob instances
    if num_unique_values / total_values < 0.1:
        #print("Categorical Feature")
        #counts = Counter(x_train[:,shift_feature])
        #most_frequent_value = counts.most_common(1)[0][0]

        # Not doing random anymore so fitness is monotonic
        # Holding out most frequent category
        selected_category = np.random.choice(unique_values)
        oob_indicies = np.where(x_train[:,shift_feature] != selected_category)[0]

    else:
        #print("Numeric Feature")
        top_percent = 100 - bottom_percent

        bottom_percentile = np.percentile(x_train[:,shift_feature], bottom_percent)
        top_percentile = np.percentile(x_train[:,shift_feature], top_percent)

        # Get the indices of the bottom 10% values
        bottom_indices = np.where(x_train[:,shift_feature] <= bottom_percentile)[0]

        # Get the indices of the top 10% values
        top_indices = np.where(x_train[:,shift_feature] >= top_percentile)[0]
        oob_indicies = np.concatenate([bottom_indices, top_indices]) 

    return oob_indicies

def compute_ensemble_oob_preds(model_pool_clf,
                               indices,
                               x_train, 
                               y_train, 
                               feature_shift, 
                               logistic_regression_train,
                               logistic_regression_val):
    
    oob_pred_shape = (x_train.shape[0], len(np.unique(y_train)))
    oob_pred = np.zeros(shape=oob_pred_shape, dtype=np.float32)
    oob_pred_counts = np.zeros(shape=x_train.shape[0], dtype=np.float32)

    for idx in indices:
        features_encoding = model_pool_clf.feature_encodings[idx]

        if feature_shift == 'feature_shift':
            unsampled_indices = get_feature_shift_oob(x_train,
                                                      y_train,
                                                      features_encoding,
                                                      bottom_percent=10,
                                                      random=False)
            
        elif feature_shift == 'regression_train':
            unsampled_indices = logistic_regression_train
            pass
        elif feature_shift == 'regression_val':
            unsampled_indices = logistic_regression_val

        else:
            _, unsampled_indices = generate_sample_indices(model_pool_clf.ensemble[idx].random_state,
                                                           x_train.shape[0],
                                                           model_pool_clf.ensemble[idx].training_frac)

            # also perform feature shift
            if feature_shift == 'both':
                unsampled_indices = get_feature_shift_oob(x_train[unsampled_indices],
                                                          y_train[unsampled_indices],
                                                          features_encoding,
                                                          bottom_percent=20,
                                                          random=False)




        # OOB Preds
        y_oob_pred_probs = model_pool_clf.oob_pred_probs[idx][unsampled_indices]
        y_oob_preds = y_oob_pred_probs.round()
        oob_pred[unsampled_indices, ...] += y_oob_preds
        oob_pred_counts[unsampled_indices] += 1
    ensemble_oob_preds = oob_pred.argmax(axis=1)

    # catch divide by zero error.  We filter out these predictions later
    ensemble_oob_pred_probs = oob_pred/oob_pred.sum(axis=1)[:,np.newaxis]

    return ensemble_oob_preds, ensemble_oob_pred_probs, oob_pred_counts

def get_oob_index_LR(x_train, y_train, x_test, percentile=10):
    clf = LogisticRegression(class_weight='balanced')
    clf.fit(x_train, y_train)

    distances_to_hyperplane = clf.decision_function(x_test)
    distances_to_hyperplane = np.abs(distances_to_hyperplane)
    
    percentile_value = np.percentile(distances_to_hyperplane, percentile)
    indexes = np.where(distances_to_hyperplane <= percentile_value)[0]
    
    return indexes      

def auprc_threshs(confidences, Y, threshs, as_percentages=True):
    areas = []
    sorti = np.argsort(-confidences)
    sortY = Y[sorti]
    precisions = np.cumsum(sortY)/np.arange(1, len(sortY)+1)
    recalls = np.cumsum(sortY)/np.sum(sortY)
    for t in threshs:
        interp_prec_t = np.interp(t, recalls, precisions)
        recalls_thresh = np.concatenate(([0.0], recalls[recalls<t], [t]))
        precisions_thresh = np.concatenate(([1.0], precisions[recalls<t], [interp_prec_t]))
        area_t = trapezoid(precisions_thresh, recalls_thresh)
        # area_t = metrics.auc(recalls_thresh, precisions_thresh)
        if as_percentages:
            area_t = area_t/t
        areas.append(area_t)
        
    return np.array(areas)

def get_precision_recall_auc(ensemble_pred_probs, y_val_ood, AUCTHRESHS):
    ensemble_preds = ensemble_pred_probs.argmax(axis=1)
    ensemble_preds_mean = ensemble_pred_probs[:,1]
    ensemble_preds_std = ensemble_pred_probs[:,0]*ensemble_pred_probs[:,1]

    std_threshs = np.linspace(np.min(ensemble_preds_std), np.max(ensemble_preds_std), 100)
    reject_rate = [1 - np.mean((ensemble_preds_std<=s)) for s in std_threshs]

    accus = [np.mean((ensemble_preds==y_val_ood)[(ensemble_preds_std<=s)]) for s in std_threshs]
    tps = [np.sum(((y_val_ood)*(ensemble_preds==y_val_ood))[(ensemble_preds_std<=s)]) for s in std_threshs]  # correct and positive
    fps = [np.sum(((ensemble_preds)*(ensemble_preds!=y_val_ood))[(ensemble_preds_std<=s)]) for s in std_threshs]  # incorrect and predicted positive
    AUC = auprc_threshs(ensemble_pred_probs.max(axis=1), y_val_ood, AUCTHRESHS)

    pos = np.sum(y_val_ood)
    recall = [tp/pos for tp in tps]
    precision = [tp/(tp+fp) for tp, fp in zip(tps, fps)]
    
    return precision, recall, AUC


def get_ensemble_preds_from_models(model_pred_probs):
    c1_probs = model_pred_probs.argmax(axis=2).sum(axis=0)/model_pred_probs.shape[0]
    c0_probs = 1 - c1_probs
    pred_probs = np.vstack([c0_probs, c1_probs]).T
    preds = pred_probs.argmax(axis=1)
    return preds, pred_probs


