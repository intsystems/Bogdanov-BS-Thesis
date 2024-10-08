import numpy as np
from importlib import reload

from files import utils
reload(utils)

class ZO_oracle:
    def __init__(self, func_name="Reg", sigma=0, rounding=None, oracle_mode="tpf", args=None):
        self.func_name=func_name
        if func_name not in ["Reg", "LogReg", "SVM"]:
            raise ValueError(f"Wrong function name {func_name}!")
        self.sigma = sigma
        self.rounding = rounding
        self.oracle_mode = oracle_mode
        if oracle_mode not in ["opf", "tpf"]:
            raise ValueError(f"Wrong oracle mode {oracle_mode}!")
        self.args = args
        self.dataset = args['dataset']
        if self.func_name == "Reg":
            self.A = self.args['A']
            self.b = self.args['b']
            self.c = self.args['c']
        elif func_name in ["LogReg", "SVM"]:
            self.X = self.args['X']
            self.y = self.args['y']
            self.alpha = self.args['alpha']
        self.name = f"{oracle_mode} oracle"

    def get_points(self, point_1, point_2):
        if self.func_name == "Reg":
            if self.oracle_mode == "opf":
                noise_1 = np.random.normal(loc=0, scale=self.sigma, size=1)
                noise_2 = np.random.normal(loc=0, scale=self.sigma, size=1)
                
                func_1 = utils.Reg_func(point_1, A=self.A, b=self.b, c=self.c) + noise_1
                func_2 = utils.Reg_func(point_2, A=self.A, b=self.b, c=self.c) + noise_2
            else:
                noise = np.random.normal(loc=0, scale=self.sigma, size=1)
                                    
                func_1 = utils.Reg_func(point_1, A=self.A, b=self.b, c=self.c) + noise
                func_2 = utils.Reg_func(point_2, A=self.A, b=self.b, c=self.c) + noise

            if self.rounding is not None:
                func_1 = np.round(func_1, int(-np.log10(self.rounding)))
                func_2 = np.round(func_2, int(-np.log10(self.rounding)))
            
        elif self.func_name == "LogReg":
            if self.oracle_mode == "opf":
                noise_1 = np.random.normal(loc=0, scale=self.sigma, size=1)
                noise_2 = np.random.normal(loc=0, scale=self.sigma, size=1)
                
                func_1 = utils.LogReg_func(point_1, X=self.X, y=self.y, alpha=self.alpha) + noise_1
                func_2 = utils.LogReg_func(point_2, X=self.X, y=self.y, alpha=self.alpha) + noise_2
            else:
                noise = np.random.normal(loc=0, scale=self.sigma, size=1)
                
                func_1 = utils.LogReg_func(point_1, X=self.X, y=self.y, alpha=self.alpha) + noise
                func_2 = utils.LogReg_func(point_2, X=self.X, y=self.y, alpha=self.alpha) + noise
                
            if self.rounding is not None:
                func_1 = np.round(func_1, int(-np.log10(self.rounding)))
                func_2 = np.round(func_2, int(-np.log10(self.rounding)))
                    
        elif self.func_name == "SVM":
            if self.oracle_mode == "opf":
                noise_1 = np.random.normal(loc=0, scale=self.sigma, size=1)
                noise_2 = np.random.normal(loc=0, scale=self.sigma, size=1)
                
                func_1 = utils.SVM_func(point_1, X=self.X, y=self.y, alpha=self.alpha) + noise_1
                func_2 = utils.SVM_func(point_2, X=self.X, y=self.y, alpha=self.alpha) + noise_2
            else:
                noise = np.random.normal(loc=0, scale=self.sigma, size=1)
                
                func_1 = utils.SVM_func(point_1, X=self.X, y=self.y, alpha=self.alpha) + noise
                func_2 = utils.SVM_func(point_2, X=self.X, y=self.y, alpha=self.alpha) + noise
                
            if self.rounding is not None:
                func_1 = np.round(func_1, int(-np.log10(self.rounding)))
                func_2 = np.round(func_2, int(-np.log10(self.rounding)))

        return func_1, func_2
        
class TrueGradientApproximator:
    def __init__(self, args=None):
        self.func_name = args['func_name']
        self.dataset = args['dataset']
        self.args = args
        if self.func_name in ["Reg"]:
            self.A = self.args['A']
            self.b = self.args['b']
            self.c = self.args['c']
        elif self.func_name in ["LogReg", "SVM"]:
            self.X = self.args['X']
            self.y = self.args['y']
            self.alpha = self.args['alpha']
        self.g_curr = None
        self.name = "true grad"
    
    def approx_gradient(self, x, k): ### true gradient ###
        if self.func_name == "Reg":
            grad = utils.Reg_grad(x, A=self.A, b=self.b)
        elif self.func_name == "LogReg":
            grad = utils.LogReg_grad(x, X=self.X, y=self.y, alpha=self.alpha)
        elif self.func_name == "SVM":
            grad = utils.SVM_grad(x, X=self.X, y=self.y, alpha=self.alpha)
        self.g_curr = np.copy(grad)
        
        return self.g_curr, 1

class JaguarApproximator:
    def __init__(self, ZO_oracle, gamma=1e-5, momentum_k=None, batch_size=1):
        self.ZO_oracle = ZO_oracle
        self.gamma = gamma
        self.momentum_k = momentum_k
        self.h_curr = None
        self.g_curr = None
        self.batch_size = batch_size
        self.name = "JAGUAR"
        
    def approx_gradient(self, x, k):
        d = len(x)
        if self.h_curr is None:
            self.h_curr = np.zeros_like(x)
            for i in range(d):
                e_i = np.zeros_like(x)
                e_i[i] = 1
                point_1 = x + self.gamma * e_i
                point_2 = x - self.gamma * e_i
                func_1, func_2 = self.ZO_oracle.get_points(point_1, point_2)
                self.h_curr += (func_1 - func_2) / (2 * self.gamma) * e_i
            
            self.g_curr = np.copy(self.h_curr) 
            oracle_calls = 2 * d
        else:
            approx_grad = np.zeros_like(x)
            batch_indices = np.random.choice(d, self.batch_size, replace=False)
            for i in batch_indices:
                e_i = np.zeros_like(x)
                e_i[i] = 1
                point_1 = x + self.gamma * e_i
                point_2 = x - self.gamma * e_i
                
                func_1, func_2 = self.ZO_oracle.get_points(point_1, point_2)
                approx_grad = (func_1 - func_2) / (2 * self.gamma) * e_i
                self.h_curr = self.h_curr - self.h_curr[i] * e_i + approx_grad
                
            if self.momentum_k is not None:
                eta_k = self.momentum_k(k)
                self.g_curr = (1 - eta_k) * self.g_curr + eta_k * self.h_curr
            else:
                self.g_curr = np.copy(self.h_curr)
                
            oracle_calls = 2 * self.batch_size

        return self.g_curr, oracle_calls
    
class LameApproximator:
    def __init__(self, ZO_oracle, gamma=1e-5):
        self.ZO_oracle = ZO_oracle
        self.gamma = gamma
        self.g_curr = None
        self.name = "$l_2$-smoothing"
        
    def approx_gradient(self, x, k):
        d = len(x)
        e = np.random.random(size=d)
        e = e / np.linalg.norm(e)
        point_1 = x + self.gamma * e
        point_2 = x - self.gamma * e

        func_1, func_2 = self.ZO_oracle.get_points(point_1, point_2)
        approx_grad = (func_1 - func_2) / (2 * self.gamma) * e
        
        self.g_curr = np.copy(approx_grad)

        return self.g_curr, 2

class TurtleApproximator:
    def __init__(self, ZO_oracle, gamma=1e-5):
        self.ZO_oracle = ZO_oracle
        self.gamma = gamma
        self.g_curr = None
        self.name = "full approximation"
        
    def approx_gradient(self, x, k):
        d = len(x)
        approx_grad = np.zeros_like(x)
        for i in range(d):
            e_i = np.zeros_like(x)
            e_i[i] = 1
            point_1 = x + self.gamma * e_i
            point_2 = x - self.gamma * e_i

            func_1, func_2 = self.ZO_oracle.get_points(point_1, point_2)
            approx_grad += (func_1 - func_2) / (2 * self.gamma) * e_i
        
        self.g_curr = np.copy(approx_grad)

        return self.g_curr, 2 * d
