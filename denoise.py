import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.neighbors import KernelDensity


def cov_to_corr(cov):
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr[corr < -1], corr[corr > 1] = -1, 1

    return corr


def get_pca(matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    indices = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]
    eigenvalues = np.diagflat(eigenvalues)

    return eigenvalues, eigenvectors


def find_max_eval(eigen_observations, tn_relation, kde_bwidth):
    optimization = minimize(pdf_fit, x0=np.array(0.5), args=(
        eigen_observations, tn_relation, kde_bwidth), bounds=((1e-5, 1 - 1e-5),))
    var = optimization['x'][0]
    maximum_eigen = var * (1 + (1 / tn_relation) ** (1 / 2)) ** 2

    return maximum_eigen, var


def pdf_fit(var, eigen_observations, tn_relation, kde_bwidth, num_points=1000):
    theoretical_pdf = mp_pdf(var, tn_relation, num_points)
    empirical_pdf = fit_kde(
        eigen_observations, kde_bwidth, eval_points=theoretical_pdf.index.values)
    sse = np.sum((empirical_pdf - theoretical_pdf) ** 2)

    return sse


def mp_pdf(var, tn_relation, num_points):
    if not isinstance(var, float):
        var = float(var)
    eigen_min = var * (1 - (1 / tn_relation) ** (1 / 2)) ** 2
    eigen_max = var * (1 + (1 / tn_relation) ** (1 / 2)) ** 2
    eigen_space = np.linspace(eigen_min, eigen_max, num_points)
    pdf = tn_relation * ((eigen_max - eigen_space) * (eigen_space - eigen_min)) ** (1 / 2) / \
        (2 * np.pi * var * eigen_space)
    pdf = pd.Series(pdf, index=eigen_space)

    return pdf


def fit_kde(observations, kde_bwidth=0.01, kde_kernel='gaussian', eval_points=None):
    observations = observations.reshape(-1, 1)
    kde = KernelDensity(kernel=kde_kernel,
                        bandwidth=kde_bwidth).fit(observations)
    if eval_points is None:
        eval_points = np.unique(observations).reshape(-1, 1)
    if len(eval_points.shape) == 1:
        eval_points = eval_points.reshape(-1, 1)
    log_prob = kde.score_samples(eval_points)
    pdf = pd.Series(np.exp(log_prob), index=eval_points.flatten())

    return pdf


def corr_to_cov(corr, std):
    cov = corr * np.outer(std, std)
    return cov


def denoise(cov, tn_relation, kde_bwidth=0.01, alpha=0.5):

    corr = cov_to_corr(cov)

    # Calculating eigenvalues and eigenvectors
    eigenval, eigenvec = get_pca(corr)

    # Calculating the maximum eigenvalue to fit the theoretical distribution
    maximum_eigen, _ = find_max_eval(
        np.diag(eigenval), tn_relation, kde_bwidth)

    # Calculating the threshold of eigenvalues that fit the theoretical distribution
    # from our set of eigenvalues
    num_facts = eigenval.shape[0] - \
        np.diag(eigenval)[::-1].searchsorted(maximum_eigen)

    # Getting the eigenvalues and eigenvectors related to signal
    eigenvalues_signal = eigenval[:num_facts, :num_facts]
    eigenvectors_signal = eigenvec[:, :num_facts]

    # Getting the eigenvalues and eigenvectors related to noise
    eigenvalues_noise = eigenval[num_facts:, num_facts:]
    eigenvectors_noise = eigenvec[:, num_facts:]

    # Calculating the correlation matrix from eigenvalues associated with signal
    corr_signal = np.dot(eigenvectors_signal, eigenvalues_signal).dot(
        eigenvectors_signal.T)

    # Calculating the correlation matrix from eigenvalues associated with noise
    corr_noise = np.dot(eigenvectors_noise, eigenvalues_noise).dot(
        eigenvectors_noise.T)

    # Calculating the De-noised correlation matrix
    corr = corr_signal + alpha * corr_noise + \
        (1 - alpha) * np.diag(np.diag(corr_noise))

    cov_denoised = corr_to_cov(corr, np.diag(cov) ** (1 / 2))

    cov_denoised = pd.DataFrame(
        cov_denoised, index=cov.index, columns=cov.columns)

    return cov_denoised
