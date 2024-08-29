import numpy as np
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from scipy.spatial.distance import cdist

def get_non_nan_indexes(array):
    """
    Returns the indexes of all non-NaN values in the given array.
    
    Parameters:
    array (np.ndarray): Input array containing possible NaN values.
    
    Returns:
    np.ndarray: Indexes of all non-NaN values.
    """
    non_nan_indexes = np.where(~np.isnan(array))[0]
    return non_nan_indexes


def compute_metrics(preds, pred_probs, datay):
    # will be nan due to oob divide by zero
    indexes = get_non_nan_indexes(pred_probs[:, 1])


    if len(indexes) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    percent_positive_preds = np.sum(preds[indexes])/len(preds[indexes])
    acc = accuracy_score(datay[indexes], preds[indexes].astype(int))
    
    pos_preds_count = sum(preds[indexes])
    if (pos_preds_count == len(indexes)) | (pos_preds_count==0):
        prec = np.nan
    else:
        prec = precision_score(datay[indexes], preds[indexes].astype(int))

    rec = recall_score(datay[indexes], preds[indexes].astype(int))
    
    try:
        auc = roc_auc_score(datay[indexes], pred_probs[indexes, 1])
    except Exception as e:
        auc = np.nan
        print('AUC ERROR: ', e)

    return acc, prec, rec, auc, percent_positive_preds


def acc_to_disagreement_fitness(model_preds, ensemble_preds, y_test):
    """
    Show that ensembling improves performance significantly whenever the disagreement
    rate is large relative to the average error rate
    """
    
    disagreement = np.mean((model_preds != ensemble_preds))
    accuracy = (ensemble_preds == y_test).mean()
    
    return accuracy/disagreement

def entropy_fitnes(model_preds, y_test):
    ensemble_size = model_preds.shape[0] 
    correct_pred = (model_preds == y_test).sum(axis=0)
    incorrect_pred = model_preds.shape[0] - correct_pred
    entropy = np.mean(1/(ensemble_size - .5*ensemble_size)*np.min([correct_pred, incorrect_pred], axis=0))
    return entropy

def kw_variance_fitness(model_preds, y_test):
    ensemble_size = model_preds.shape[0] 
    correct_pred = (model_preds == y_test).sum(axis=0)
    incorrect_pred = model_preds.shape[0] - correct_pred
    kw = np.mean(correct_pred*incorrect_pred)/(ensemble_size**2)
    return kw

def disagreement_fitness(model_preds, ensemble_preds, y_test):
    """
    We want to maximize disagreement between individual model predictions
    and ensemble predictions when the voting classifier is correct. We want to
    minimize disagreement when the entire voting classifier is incorrect
    """
    model_ensemble_disagreement = (model_preds != ensemble_preds).mean(axis=0)
    
    correct_preds = (ensemble_preds == y_test)
    incorrect_preds = (ensemble_preds != y_test)
    
    correct_disagreement = np.mean(model_ensemble_disagreement[correct_preds])
    incorrect_disagreement = np.mean(model_ensemble_disagreement[incorrect_preds])
    
    return correct_disagreement, incorrect_disagreement

def get_agreement_measures(model_preds, model_pred_probs):
    agreement_mat = np.ones((model_preds.shape[0], model_preds.shape[0]))
    agreement_std_mat = np.ones((model_preds.shape[0], model_preds.shape[0]))

    conf_agreement_mat = np.ones((model_preds.shape[0], model_preds.shape[0]))
    conf_agreement_std_mat = np.ones((model_preds.shape[0], model_preds.shape[0]))

    for i in range(0, model_preds.shape[0]):
        for j in range(i, model_preds.shape[0]):
            same = (model_preds[i] == model_preds[j])
            agreement = np.mean(same)
            agreement_std = np.std(same)

            conf_overlap = (model_pred_probs[i][:,0] * model_pred_probs[j][:,0])
            conf_agreement = np.mean(conf_overlap)
            conf_agreement_std = np.std(conf_overlap)

            agreement_mat[i, j] = agreement
            agreement_mat[j, i] = agreement

            agreement_std_mat[i, j] = agreement_std
            agreement_std_mat[j, i ] = agreement_std

            conf_agreement_mat[i, j] = conf_agreement
            conf_agreement_mat[j, i] = conf_agreement

            conf_agreement_std_mat[i, j] = conf_agreement_std
            conf_agreement_std_mat[j, i] = conf_agreement_std


    agreement_avg = agreement_mat.mean()
    agreement_std = agreement_std_mat.mean()

    conf_agreement_avg = conf_agreement_mat.mean()
    conf_agreement_std = conf_agreement_mat.mean()

    min_agreement = agreement_mat.min(axis=0).mean()
    min_conf_agreement = conf_agreement_mat.min(axis=0).mean()

    min_agreement_std = agreement_mat.min(axis=0).mean()
    min_conf_agreement_std = conf_agreement_mat.min(axis=0).mean()
    
    return agreement_avg, agreement_std, conf_agreement_avg, conf_agreement_std, min_agreement, min_agreement_std, min_conf_agreement, min_conf_agreement_std


def get_model_misclass_mat(model_preds, Y):
    model_misclass_mat = np.zeros((model_preds.shape[0], model_preds.shape[1]))
    M = model_preds.shape[1]

    for i in range(0, model_preds.shape[0]):
        # 0 for correct predictions, 1 for incorrect
        model_misclass_mat[i] = ~(model_preds[i] == Y)

    G = np.matmul(model_misclass_mat, model_misclass_mat.T)

    # Step 2: Update the off-diagonal elements
    i_indices, j_indices = np.indices(G.shape)
    diagonal_mask = (i_indices == j_indices)

    # Avoid modifying the diagonal elements
    G[~diagonal_mask] = 0.5 * (G[~diagonal_mask] / G[i_indices[~diagonal_mask], i_indices[~diagonal_mask]] + G[~diagonal_mask] / G[j_indices[~diagonal_mask], j_indices[~diagonal_mask]])
    np.fill_diagonal(G, np.diagonal(G) / M)
    G = np.nan_to_num(G, nan=1)

    return G

def gaussian_kernel(x, y, sigma=1.0):
    dists = cdist(x, y, 'euclidean')
    return np.exp(-dists ** 2 / (2 * sigma ** 2))

def compute_mmd(x, y, sigma=1.0):
    m = x.shape[0]
    n = y.shape[0]

    Kxx = gaussian_kernel(x, x, sigma)
    Kyy = gaussian_kernel(y, y, sigma)
    Kxy = gaussian_kernel(x, y, sigma)

    mmd = Kxx.sum() / (m * m) + Kyy.sum() / (n * n) - 2 * Kxy.sum() / (m * n)
    return mmd

def fisher_information_distance(mu1, sigma1, mu2, sigma2):
    inv_sigma1 = np.linalg.inv(sigma1)
    inv_sigma2 = np.linalg.inv(sigma2)
    
    mean_diff = mu1 - mu2
    mean_diff_term = np.dot(mean_diff.T, np.dot(inv_sigma1 + inv_sigma2, mean_diff))
    
    trace_term = np.trace(np.dot(inv_sigma1, sigma2) + np.dot(inv_sigma2, sigma1) - 2 * np.identity(len(mu1)))
    
    fid = 0.5 * (mean_diff_term + trace_term)
    return fid

def hamming_distance(x, y):
    return np.sum(np.abs(x - y))
    

def get_ensemble_preds_from_models(model_pred_probs):
    c1_probs = model_pred_probs.argmax(axis=2).sum(axis=0)/model_pred_probs.shape[0]
    c0_probs = 1 - c1_probs
    pred_probs = np.vstack([c0_probs, c1_probs]).T
    preds = pred_probs.argmax(axis=1)
    return preds, pred_probs