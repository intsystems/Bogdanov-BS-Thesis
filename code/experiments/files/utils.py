import numpy as np
from matplotlib import pylab as plt

def generate_matrix(d, mu, L):
    diag = (L - mu) * np.random.random_sample(d) + mu
    sigma = np.diag(diag)
    sigma[0][0] = L
    sigma[d - 1][d - 1] = mu
    rand_matrix = np.random.rand(d, d)
    rand_ort, _, _ = np.linalg.svd(rand_matrix)
    matrix = rand_ort.T @ sigma @ rand_ort
    return matrix

def quadratic_func(w, A, b, c=None):
    if c is not None:
        return (1./2 * w.T @ A @ w - b.T @ w + c)[0]
    else:
        return (1./2 * w.T @ A @ w - b.T @ w)[0]

def quadratic_grad(w, A, b):
    return A @ w - b

def logreg_func(w, X, y, alpha = 0.1):
    return np.mean(np.log(1 + np.exp(- X * np.expand_dims(y, axis=1) @ np.expand_dims(w, axis=1)))) + alpha * np.square(np.linalg.norm(w))

def logreg_grad(w, X, y, alpha = 0.1):
    return -np.mean(X * np.expand_dims(y, axis=1) / np.expand_dims((1 + np.exp(X @ w * y)), axis=1), axis = 0) + 2 * alpha * w
    
def SVM_func(w, X, y, alpha = 0.1):
    w_new = w[:-1]
    b = w[-1]
    return 1 / 2 * np.square(np.linalg.norm(w_new)) + alpha * np.sum(np.maximum(0, y * (X @ w_new - b)))

def make_err_plot(optimizers_list, labels=None, title=None, markers=None, colors=None, save_name=None):
    if markers is None:
        markers = ['o', 'v', 's', 'p', 'x', 'P', 'D', '^', '<', '>']
    if colors is None:
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'black', 'olive', 'pink', 'brown']

    x_label = "The number of oracle calls"
    y_label = "$f(x^k) / f(x^0)$"
#    if optimizers_list[0].x_sol is not None:
#        y_label = r'$\frac{f(x^k) - f(x^*)}{f(x^0) - f(x^*)}$'
#    else:
#        if optimizers_list[0].func_name == "SVM":
#
#        else:
#            y_label = r'$||\nabla f(x^k)||$'
    
    if labels is None:
        labels = [optimizer.gradient_approximator.name for optimizer in optimizers_list]
        if title is None:
            if optimizers_list[0].func_name == "SVM":
                func_name = "SVM"
            elif optimizers_list[0].func_name == "logreg":
                func_name = "LogReg"
            elif optimizers_list[0].func_name == "quadratic":
                func_name = "Reg"
            else:
                raise ValueError("Unknown problem!")
            
            if optimizers_list[0].set.name == "R":
                sett = "$R^d$"
            elif optimizers_list[0].set.name == "L2 Ball":
                sett = "$l_2$-ball"
            elif optimizers_list[0].set.name == "L1 Ball":
                sett = "$l_1$-ball"
            elif optimizers_list[0].set.name == "Simplex":
                sett = "Simplex"
            else:
                raise ValueError("Unknown set!")
                
            title = f"{func_name} on {sett} \n {optimizers_list[0].dataset} dataset"
        else:
            raise ValueError("Enter labels to the plot!")

    if title is not None:
        plt.title(title, fontsize=25)
    plt.xlabel(x_label, fontsize=25)
    plt.ylabel(y_label, fontsize=25)

    for optimizer, label, color, marker in zip(optimizers_list, labels, colors, markers):
        oracles_calls = optimizer.oracle_calls_list
        errors = optimizer.errors_list
        plt.semilogy(oracles_calls, errors, color=color, label=label, markevery=0.05, marker=marker)

    plt.legend(fontsize=17)
    plt.grid(True)
    
    if save_name is not None:
        plt.tight_layout()
        plt.savefig(f"figures/{save_name}.pdf", format='pdf')
        plt.savefig(f"figures/{save_name}.png", format='png')
        
    plt.show()
