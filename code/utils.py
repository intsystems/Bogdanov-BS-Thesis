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

def quadratic_func(x, A, b, c=0):
    return 1./2 * x.T @ A @ x - b.T @ x + c

def quadratic_grad(x, A, b):
    return A @ x - b

def logreg_func(x, matrix, alpha = 0.1):
    x_exp = np.expand_dims(x, axis=1)
    loss = np.mean(np.log(1 + np.exp(- matrix @ x_exp))) + alpha * np.square(np.linalg.norm(x))
    return loss

def logreg_grad(x, matrix, alpha = 0.1):
    x_exp = np.expand_dims(x, axis=1)
    grad = - np.mean(matrix / (1 + np.exp(matrix @ x_exp)), axis = 0)
    return grad + 2 * alpha * x

def make_err_plot(optimizers_list, labels=None, title=None, 
                  markers=None, markersize=7, save_name=None, 
                  colors=None, linestyles=None):
    if markers is None:
        markers = [None] * 100
    if colors is None:
        colors = ['red', 'green', 'blue', 'orange', 'purple',
                  'cyan', 'black', 'olive', 'pink', 'brown']
    if linestyles is None:
        linestyles = ['solid'] * 100

    x_label = "Количесто вызовов оракула"
    if optimizers_list[0].x_sol is not None:
        y_label = r'$\frac{f(x^k) - f(x^*)}{f(x^0) - f(x^*)}$'
    else:
        y_label = r'$||\nabla f(x^k)||$'

    if labels is None:
        if len(set([optimizer.gradient_approximator.name for optimizer in optimizers_list])) > 1:
            labels = [optimizer.gradient_approximator.name for optimizer in optimizers_list]
            if title is None:
                #mode = optimizers_list[0].gradient_approximator.oracle_mode
                title = f"Различные апроксимации градиента, {optimizers_list[0].name} алгоритм"
        elif len(set([optimizer.name for optimizer in optimizers_list])) > 1:
            labels = [optimizer.name for optimizer in optimizers_list]
            if title is None:
                title = f"Различные алгоритмы, {optimizers_list.gradient_approximator.name} апроксимация"
        else:
            raise ValueError("Enter labels to the plot!")

    plt.figure(figsize=(12, 8))
    if title is not None:
        plt.title(title + "\n логарифмический масштаб по оси y", fontsize=15)
    plt.xlabel(x_label, fontsize=15)
    plt.ylabel(y_label, fontsize=15)

    for optimizer, label, color, marker, linestyle in \
            zip(optimizers_list, labels, colors, markers, linestyles):
        oracles_calls = optimizer.oracle_calls_list
        errors = optimizer.errors_list
        plt.semilogy(oracles_calls, errors, color=color, label=label, linewidth=2,
                     marker=marker, markersize=markersize, linestyle=linestyle)

    plt.grid()
    plt.legend(fontsize=15)
    plt.tight_layout()
    if save_name is not None:
        plt.savefig(f"figures/{save_name}.png")
    plt.show()
