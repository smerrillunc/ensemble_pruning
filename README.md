# Ensemble Pruning
- This repo explores ensemble pruning.  Below is a description of the files.
1.  **random_search.py**: A simple ranodm search of ensembles that may outperform model pool.
2.  **cheating_fitness.py**: A script that computes all of the fitness as well as precisions, recalls, etc.  With this massive catelog we can try to find those fitness functions that correlate the most with OOD AUC.
3.  **column_averaging.py**: A super quick attempt at an evolutionary algorithm that attempts to identify multiple fitness functions to average which maximizes OOD AUC.
4.  **fitness_functions.py**: A giant mess of a bunch of different fitness functions we've explored
5.  **voting_classifier.py**: A class that makes a voting classifier and has some helpful methods
6.  **utils.py**: Miscelaneous utilities utilized in 1, 2 and some of the notebooks
7.  **fitness_funcs_list.xlsx**: comprehensive list of fitness functions and Descriptions

Notebooks:
- 3 notebooks that may be helpful to analyze the different fitness functions
