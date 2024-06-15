import numpy as np
from importlib import reload
from sklearn.datasets import load_svmlight_file

from files import utils
reload(utils)

def init_experiment(func_name, dataset="mushrooms", d=112, seed=18, c=False, alpha=0.1, **init_args):
    args = {}
    args['func_name'] = func_name
    if func_name == "quadratic":
        np.random.seed(seed)
        L = init_args['L']
        mu = init_args['mu']
        args['A'] = utils.generate_matrix(d, mu, L)
        args['b'] = np.random.random(d)
        if c:
            args['c'] = np.random.random(1)
        else:
            args['c'] = np.array([0])
        args['dataset'] = "Synthetic"
            
    elif func_name == "logreg":
        if dataset == "mushrooms":
            dataset = "../data/mushrooms.txt"
            args['dataset'] = "Mushrooms"
            k = 3
        else:
            dataset = "../data/MNIST.txt"
            args['dataset'] = "MNIST"
            k = 1
        data = load_svmlight_file(dataset)
        X, y = data[0].toarray(), data[1]
        y = y * 2 - k
        args['X'] = X
        args['y'] = y
        args['alpha'] = alpha
        
    elif func_name == "SVM":
        if dataset == "mushrooms":
            dataset = "../data/mushrooms.txt"
            args['dataset'] = "Mushrooms"
            k = 3
        else:
            dataset = "../data/MNIST.txt"
            args['dataset'] = "MNIST"
            k = 1
        data = load_svmlight_file(dataset)
        X, y = data[0].toarray(), data[1]
        y = y * 2 - k
        args['X'] = X
        args['y'] = y
        args['alpha'] = alpha
    
    return args
