import numpy as np
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier

from denoise import cov_to_corr
from onc import clusterKMeansTop


def markowitz(cov, mu, bounds):
    ef = EfficientFrontier(
        mu, cov, weight_bounds=bounds)
    w = ef.max_sharpe()
    w = np.array(list(ef.clean_weights().values()))
    return w


def nco(cov, mu, bounds=(-1, 1)):
    corr = cov_to_corr(cov)
    _, clstrs, _ = clusterKMeansTop(corr, int(corr.shape[0]/2))
    print(clstrs)
    w_intra = pd.DataFrame(0, index=cov.index, columns=clstrs.keys())
    for i in clstrs:
        w_intra.loc[clstrs[i], i] = markowitz(
            cov.loc[clstrs[i], clstrs[i]], mu.loc[clstrs[i]], bounds)

    cov_inter = w_intra.T.dot(np.dot(cov, w_intra))
    mu_inter = w_intra.T.dot(mu)

    w_inter = pd.Series(markowitz(cov_inter, mu_inter, bounds),
                        index=cov_inter.index)

    w_nco = w_intra.mul(w_inter, axis=1).sum(
        axis=1).values.reshape(-1, 1)

    weights = pd.DataFrame(w_nco.T, columns=cov.columns)

    return weights.loc[0, :]
