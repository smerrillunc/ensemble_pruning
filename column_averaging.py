
import pandas as pd
import numpy as np
import os,sys
from datetime import datetime
import random
import numbers
import warnings
import argparse
from datetime import date
from SaveUtil import SaveUtil
from deap import base, creator, tools, algorithms
import copy

def precision_at_k(column1, column2, k=10, largest1=True, largest2=False):
    """
    Compute Precision at k between two columns.
    
    Parameters:
    column1 (pd.Series): First column of data.
    column2 (pd.Series): Second column of data.
    k (int): Number of top elements to consider.
    
    Returns:
    float: Precision at k score.
    """
    # Get the top k indices for both columns
    if largest1:
        top_k_col1 = column1.nlargest(k).index
    else:
        top_k_col1 = column1.nsmallest(k).index
    
    if largest2:
        top_k_col2 = column2.nlargest(k).index
    else:
        top_k_col2 = column2.nsmallest(k).index
    
    # Compute the number of common indices
    common_indices = len(set(top_k_col1).intersection(set(top_k_col2)))
    
    # Precision at k
    precision = common_indices / k
    
    return precision

def add_difference_cols(results_df2, cols_to_diff):
    results_df = copy.deepcopy(results_df2)
    start_cols = results_df.columns
    for col in cols_to_diff:
        new_cols = [x for x in start_cols if col in x]
        replace_string =  '_' + col.split('_')[-1]
        orig_cols = [x.replace(replace_string, '') for x in copy.deepcopy(new_cols)]
        
        for i in range(len(orig_cols)-1, -1, -1):
            if orig_cols[i] not in results_df.columns:
                print(orig_cols[i])
                del new_cols[i]
                del orig_cols[i]
        
        diff_cols = [x + '_diff' for x in copy.deepcopy(new_cols)]    
        #print(results_df[new_cols])
        results_df[diff_cols] = (results_df[orig_cols].values - results_df[new_cols].values)
        results_df[diff_cols] = results_df[diff_cols].values/results_df[orig_cols].values
        results_df[diff_cols] = results_df[diff_cols].fillna(0)
    return results_df

def get_results_df(save_path):
    exps = os.listdir(save_path)
    exps.sort()
    
    # concatonte all resutls_df
    results_df = pd.DataFrame()
    for exp in exps:
        try:
            tmp = pd.read_csv(save_path + f'/{exp}/results_df.csv')
            if tmp.empty:
                continue
            results_df = pd.concat([results_df, tmp])
        except Exception as e:
            print(e)
    print(results_df.shape)
    results_df = results_df.fillna(0)
    results_df = results_df.loc[:, (results_df.sum(axis=0) != 0)]
    
    for col in results_df.columns:
        try:
            results_df[col] = results_df[col].astype(float)
        except ValueError:
            results_df.drop(columns=[col], inplace=True)

    cols_to_diff = ['noise', 
                'corrupt',
                'val_id_noise', 
               'val_id_corrupt', ]

    results_df = add_difference_cols(results_df, cols_to_diff)
    return results_df

# Attribute generator: generate indices for columns
def generate_indices():
    return random.sample(range(len(opt_cols)), num_features_to_use)

# Mutation operator ensuring unique indices
def mutUniformIntCustom(individual, low, up, indpb):
    size = len(individual)
    for i in range(size):
        if random.random() < indpb:
            individual[i] = random.randint(low, up)
            while individual.count(individual[i]) > 1:
                individual[i] = random.randint(low, up)
    return individual,

# Evaluation function
def evaluate(individual):
    smallests = []
    largests = []
    for dataset in datasets:
        columns = dataset_results[dataset].iloc[:, individual]
        mean_val = columns.mean(axis=1)

        prec_k_largest = precision_at_k(mean_val, dataset_targets[dataset], k=k, largest1=True, largest2=largest2)
        prec_k_smallest = precision_at_k(mean_val, dataset_targets[dataset], k=k, largest1=False, largest2=largest2)
        smallests.append(prec_k_smallest)
        largests.append(prec_k_largest)
    return max(np.mean(smallests), np.mean(largests)),


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Optional parameters for your script")

    
    # for clustering setting
    parser.add_argument("--num_features_to_use", type=int, default=3, help ="Number of features to average")
    parser.add_argument("--pop_size", type=int, default=100, help ="Population Size")
    parser.add_argument("--ngen", type=int, default=100000, help="Number of generations")
    parser.add_argument("--prec_k", type=int, default=50, help="Number of generations")
    parser.add_argument("--cxpb", type=float, default=0.7, help="Number of generations")
    parser.add_argument("--mutpb", type=float, default=0.2, help="Number of generations")

    parser.add_argument("--load_path", type=str, default="/proj/mcavoy_lab/data/evo_models/exps/cheating_fitness_0802_EOD/", help="Load Path")
    parser.add_argument("--target", type=str, default='ensemble_auc_val_ood', help="Target Col")

    # admin params    
    parser.add_argument("--seed", type=int, default=1,help ="Seed")
    parser.add_argument("--save_name", type=str, default=None, help="Save Name")
    parser.add_argument("--save_path", type=str, default="/Users/scottmerrill/Desktop", help="Save Path")

    args = vars(parser.parse_args())

    num_features_to_use = args['num_features_to_use']
    ngen = args['ngen']
    cxpb = args['cxpb']
    mutpb = args['mutpb']
    largest2 = True
    k = args['prec_k']

    # seeds
    np.random.seed(args['seed'])
    random.seed(args['seed'])
    rng = np.random.RandomState(args['seed'])
    # save paths

    if args['save_name'] == None:
        args['save_name'] = date.today().strftime('%Y%m%d')

    save_path = SaveUtil.create_directory_if_not_exists(args['save_path'] + '/exps/{}/'.format(args['save_name'])) 
    SaveUtil.save_dict_to_file(args, save_path + '/experiment_args.txt')  

    dataset_results = {}
    dataset_targets = {}
    target = args['target']

    datasets = ['adult_tf', 'heloc_tf', 'compas', 'german']
    for i, dataset in enumerate(datasets):
        if dataset not in dataset_results.keys():
            results_df = get_results_df(args['load_path'] + f'{dataset}')
            dataset_results[dataset] = results_df
            dataset_targets[dataset] = results_df[target]

    # ugly way to common columns that we care about
    opt_cols = dataset_results[datasets[0]].columns.intersection(dataset_results[datasets[1]].columns).intersection(dataset_results[datasets[2]].columns).intersection(dataset_results[datasets[3]].columns)
    opt_cols = [x for x in opt_cols if "ood" not in x]
    opt_cols = [x for x in opt_cols if "eature" not in x]
    opt_cols.remove('error')

    for i, dataset in enumerate(datasets):
        dataset_results[dataset] = dataset_results[dataset][opt_cols]


    # Evolutionary algorithm setup
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register("indices", generate_indices)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mutate", mutUniformIntCustom, low=0, up=len(opt_cols)-1, indpb=0.2)


    output_df = pd.DataFrame()
    population = toolbox.population(n=args['pop_size'])
    for gen in range(ngen):
        print(f'STARTING GEN: {gen}')
        offspring = algorithms.varAnd(population, toolbox, cxpb, mutpb)
        fits = map(toolbox.evaluate, offspring)

        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit

        population = toolbox.select(offspring, k=len(population))

        best_ind = tools.selBest(population, 1)[0]
        tmp = {'gen':gen,
              'best_ind':best_ind,
              'score':evaluate(best_ind)[0],
              'best_names':list(dataset_results[dataset].columns[best_ind])}
        print(tmp)
        output_df = pd.concat([output_df, pd.DataFrame([tmp])])

        print(f"Generation {gen}: Best individual = {best_ind}, Score = {evaluate(best_ind)[0]}")
        output_df.to_csv(save_path + '/output_df.csv')

    best_ind = tools.selBest(population, 1)[0]
    print("Final best individual is: ", best_ind)
    print("With a score of: ", evaluate(best_ind)[0])


