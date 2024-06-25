import numpy as np
import tqdm
from importlib import reload

from files import gradient_approximation
from files import sets
from files import utils
reload(gradient_approximation)
reload(sets)
reload(utils)

class GDOptimizer:
    def __init__(self, gradient_approximator, learning_rate_k, x_0, sett=None, x_sol=None, max_oracle_calls=10**4, tol=0.000001, seed=18, err="f"):
        np.random.seed(seed)
        self.gradient_approximator = gradient_approximator
        self.set = sett
        self.learning_rate_k = learning_rate_k
        self.x_0 = x_0
        self.x_curr = np.copy(self.x_0)
        self.d = len(x_0)
        self.err = err
        
        try:
            self.func_name = gradient_approximator.ZO_oracle.func_name
            self.dataset = gradient_approximator.ZO_oracle.dataset
            if self.func_name == "Reg":
                self.A = gradient_approximator.ZO_oracle.A
                self.b = gradient_approximator.ZO_oracle.b
                self.c = gradient_approximator.ZO_oracle.c
            elif self.func_name in ["LogReg", "SVM"]:
                self.X = gradient_approximator.ZO_oracle.X
                self.y = gradient_approximator.ZO_oracle.y
                self.alpha = gradient_approximator.ZO_oracle.alpha
        except AttributeError:
            self.func_name = gradient_approximator.func_name
            if self.func_name == "Reg":
                self.A = gradient_approximator.A
                self.b = gradient_approximator.b
                self.c = gradient_approximator.c
            elif self.func_name in ["LogReg", "SVM"]:
                self.X = gradient_approximator.X
                self.y = gradient_approximator.y
                self.alpha = gradient_approximator.alpha

        self.R0 = self.get_error(x_0)
        self.max_oracle_calls = max_oracle_calls
        self.tol = tol
        self.name = "GD"

    def step(self, x, k):
        gamma_k = self.learning_rate_k(k)
        nabla_f, oracle_calls = self.gradient_approximator.approx_gradient(x, k)
        x_next = x - gamma_k * nabla_f
        x_next = self.set.projection(x_next)

        return x_next, oracle_calls
    
    def get_error(self, x):
        if self.err == "f":
            if self.func_name == "Reg":
                error = utils.Reg_func(x, A=self.A, b=self.b, c=self.c)
            elif self.func_name == "LogReg":
                error = utils.LogReg_func(x, X=self.X, y=self.y, alpha=self.alpha)
            elif self.func_name == "SVM":
                error = utils.SVM_func(x, X=self.X, y=self.y, alpha=self.alpha)
        elif self.err == "grad":
            if self.func_name == "Reg":
                error = utils.Reg_grad(x, A=self.A, b=self.b)
            elif self.func_name == "LogReg":
                error = utils.LogReg_grad(x, X=self.X, y=self.y, alpha=self.alpha)
            elif self.func_name == "SVM":
                error = utils.SVM_grad(x, X=self.X, y=self.y, alpha=self.alpha)
        elif self.err == "gap":
            if self.func_name == "Reg":
                error = self.set.gap(x, utils.Reg_grad(x, A=self.A, b=self.b))
            elif self.func_name == "LogReg":
                error = self.set.gap(x, utils.LogReg_grad(x, X=self.X, y=self.y, alpha=self.alpha))
            elif self.func_name == "SVM":
                error = self.set.gap(x, utils.SVM_grad(x, X=self.X, y=self.y, alpha=self.alpha))
        return error
    
    def optimize(self):
        if self.gradient_approximator.name == "JAGUAR":
            batch_size = self.gradient_approximator.batch_size
            num_iter = (self.max_oracle_calls - self.d) // (2 * batch_size) + 1
        if self.gradient_approximator.name == "$l_2$-smoothing":
            num_iter = self.max_oracle_calls // 2
        if self.gradient_approximator.name == "full approximation":
            num_iter = self.max_oracle_calls // (2 * self.d)
        if self.gradient_approximator.name == "true grad":
            num_iter = self.max_oracle_calls
        
        self.oracle_calls_list = [0]
        self.errors_list = [1.]
        for k in tqdm.trange(num_iter):
            self.x_curr, oracle_calls = self.step(self.x_curr, k)
            self.oracle_calls_list.append(self.oracle_calls_list[-1] + oracle_calls)
            error = self.get_error(self.x_curr) / self.R0
            self.errors_list.append(error)
            if error <= self.tol:
                print(f"Precision {self.tol} achieved at step {k}!")
                break

class FWOptimizer(GDOptimizer):
    def __init__(self, gradient_approximator, learning_rate_k, x_0, sett=None, x_sol=None, max_oracle_calls=10**4, tol=0.000001, seed=18, err="f"):
        super().__init__(gradient_approximator, learning_rate_k, x_0, sett, x_sol, max_oracle_calls, tol, seed, err)
        self.name = "FW"
    
    def step(self, x, k):
        gamma_k = self.learning_rate_k(k)
        nabla_f, oracle_calls = self.gradient_approximator.approx_gradient(x, k)
        s = self.set.fw_argmin(nabla_f)
        x_next = (1 - gamma_k) * x + gamma_k * s

        return x_next, oracle_calls
