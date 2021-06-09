# importing our required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import portfoliolab as pl


def cov2corr(cov):
    # Derive the correlation matrix from a covariance matrix
    std = np.sqrt(np.diag(cov))
    corr = cov/np.outer(std, std)
    corr[corr < -1], corr[corr > 1] = -1, 1  # numerical error
    return corr


def denoiseCov(cov, q, bWidth=0.01, detone=False):
    risk_estimators = pl.estimators.RiskEstimators()
    cov_denoised = risk_estimators.denoise_covariance(
        cov, q, denoise_method='target_shrink', detone=detone, kde_bwidth=bWidth, alpha=0.5)
    cov_denoised = pd.DataFrame(
        cov_denoised, index=cov.index, columns=cov.columns)
    return cov_denoised
