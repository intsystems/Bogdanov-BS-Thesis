import numpy as np
from importlib import reload
from sklearn.datasets import load_svmlight_file

from files import utils
reload(utils)

def init_experiment(func_name, dataset="Mushrooms", d=None, seed=18, c=False, alpha=0.1, **init_args):
    args = {}
    args['func_name'] = func_name
    if func_name == "Reg":
        np.random.seed(seed)
        L = init_args['L']
        mu = init_args['mu']
        args['A'] = utils.generate_matrix(d, mu, L)
        args['b'] = np.random.random(d)
        if c:
            args['c'] = np.random.random(1)
        else:
            args['c'] = 0
        args['dataset'] = "Synthetic"
            
    elif func_name == "LogReg":
        args['dataset'] = dataset
        data = load_svmlight_file(f"../data/{dataset}.txt")
        X, y = data[0].toarray(), data[1]
        y = y * 2 - 3
        args['X'] = X
        args['y'] = y
        args['alpha'] = alpha
        if dataset == "Mushrooms":
            d = 112
        else:
            d = 718
        
    elif func_name == "SVM":
        args['dataset'] = dataset
        data = load_svmlight_file(f"../data/{dataset}.txt")
        X, y = data[0].toarray(), data[1]
        y = y * 2 - 3
        args['X'] = X
        args['y'] = y
        args['alpha'] = alpha
        if dataset == "Mushrooms":
            d = 112
        else:
            d = 718
    
    return d, args
