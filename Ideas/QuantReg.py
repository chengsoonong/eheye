from statsmodels.compat.python import range
import numpy as np
import warnings
import scipy.stats as stats
from scipy.linalg import pinv
from scipy.stats import norm
from statsmodels.tools.tools import chain_dot
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import (RegressionModel,
                                                 RegressionResults,
                                                 RegressionResultsWrapper)
from statsmodels.tools.sm_exceptions import (ConvergenceWarning,
                                             IterationLimitWarning)

class QuantReg(RegressionModel):
    '''Quantile Regression

    Estimate a quantile regression model using iterative reweighted least
    squares.

    Parameters
    ----------
    endog : array or dataframe
        endogenous/response variable
    exog : array or dataframe
        exogenous/explanatory variable(s)

    Notes
    -----
    The Least Absolute Deviation (LAD) estimator is a special case where
    quantile is set to 0.5 (q argument of the fit method).

    The asymptotic covariance matrix is estimated following the procedure in
    Greene (2008, p.407-408), using either the logistic or gaussian kernels
    (kernel argument of the fit method).

    References
    ----------
    General:

    * Birkes, D. and Y. Dodge(1993). Alternative Methods of Regression, John Wiley and Sons.
    * Green,W. H. (2008). Econometric Analysis. Sixth Edition. International Student Edition.
    * Koenker, R. (2005). Quantile Regression. New York: Cambridge University Press.
    * LeSage, J. P.(1999). Applied Econometrics Using MATLAB,

    Kernels (used by the fit method):

    * Green (2008) Table 14.2

    Bandwidth selection (used by the fit method):

    * Bofinger, E. (1975). Estimation of a density function using order statistics. Australian Journal of Statistics 17: 1-17.
    * Chamberlain, G. (1994). Quantile regression, censoring, and the structure of wages. In Advances in Econometrics, Vol. 1: Sixth World Congress, ed. C. A. Sims, 171-209. Cambridge: Cambridge University Press.
    * Hall, P., and S. Sheather. (1988). On the distribution of the Studentized quantile. Journal of the Royal Statistical Society, Series B 50: 381-391.

    Keywords: Least Absolute Deviation(LAD) Regression, Quantile Regression,
    Regression, Robust Estimation.
    '''

    def __init__(self, endog, exog, **kwargs):
        super(QuantReg, self).__init__(endog, exog, **kwargs)

    def whiten(self, data):
        """
        QuantReg model whitener does nothing: returns data.
        """
        return data


    def fit(self, q=.5, vcov='robust', kernel='epa', bandwidth='hsheather',
            max_iter=1000, p_tol=1e-6, **kwargs):
        '''Solve by Iterative Weighted Least Squares

        Parameters
        ----------
        q : float
            Quantile must be between 0 and 1
        vcov : string, method used to calculate the variance-covariance matrix
            of the parameters. Default is ``robust``:

            - robust : heteroskedasticity robust standard errors (as suggested
              in Greene 6th edition)
            - iid : iid errors (as in Stata 12)

        kernel : string, kernel to use in the kernel density estimation for the
            asymptotic covariance matrix:

            - epa: Epanechnikov
            - cos: Cosine
            - gau: Gaussian
            - par: Parzene

        bandwidth: string, Bandwidth selection method in kernel density
            estimation for asymptotic covariance estimate (full
            references in QuantReg docstring):

            - hsheather: Hall-Sheather (1988)
            - bofinger: Bofinger (1975)
            - chamberlain: Chamberlain (1994)
        '''

        if q < 0 or q > 1:
            raise Exception('p must be between 0 and 1')

        kern_names = ['biw', 'cos', 'epa', 'gau', 'par']
        if kernel not in kern_names:
            raise Exception("kernel must be one of " + ', '.join(kern_names))
        else:
            kernel = kernels[kernel]

        if bandwidth == 'hsheather':
            bandwidth = hall_sheather
        elif bandwidth == 'bofinger':
            bandwidth = bofinger
        elif bandwidth == 'chamberlain':
            bandwidth = chamberlain
        else:
            raise Exception("bandwidth must be in 'hsheather', 'bofinger', 'chamberlain'")

        endog = self.endog
        exog = self.exog
        nobs = self.nobs
        exog_rank = np.linalg.matrix_rank(self.exog)
        self.rank = exog_rank
        self.df_model = float(self.rank - self.k_constant)
        self.df_resid = self.nobs - self.rank
        n_iter = 0
        xstar = exog

        beta = np.ones(exog_rank)
        # TODO: better start, initial beta is used only for convergence check

        # Note the following doesn't work yet,
        # the iteration loop always starts with OLS as initial beta
        #        if start_params is not None:
        #            if len(start_params) != rank:
        #                raise ValueError('start_params has wrong length')
        #            beta = start_params
        #        else:
        #            # start with OLS
        #            beta = np.dot(np.linalg.pinv(exog), endog)

        diff = 10
        cycle = False

        history = dict(params = [], mse=[])
        while n_iter < max_iter and diff > p_tol and not cycle:
            n_iter += 1
            beta0 = beta
            xtx = np.dot(xstar.T, exog)
            xty = np.dot(xstar.T, endog)
            beta = np.dot(pinv(xtx), xty)
            resid = endog - np.dot(exog, beta)

            mask = np.abs(resid) < .000001
            resid[mask] = ((resid[mask] >= 0) * 2 - 1) * .000001
            resid = np.where(resid < 0, q * resid, (1-q) * resid)
            resid = np.abs(resid)
            xstar = exog / resid[:, np.newaxis]
            diff = np.max(np.abs(beta - beta0))
            history['params'].append(beta)
            history['mse'].append(np.mean(resid*resid))

            if (n_iter >= 300) and (n_iter % 100 == 0):
                # check for convergence circle, shouldn't happen
                for ii in range(2, 10):
                    if np.all(beta == history['params'][-ii]):
                        cycle = True
                        warnings.warn("Convergence cycle detected", ConvergenceWarning)
                        break

        if n_iter == max_iter:
            warnings.warn("Maximum number of iterations (" + str(max_iter) +
                          ") reached.", IterationLimitWarning)

        e = endog - np.dot(exog, beta)
        # Greene (2008, p.407) writes that Stata 6 uses this bandwidth:
        # h = 0.9 * np.std(e) / (nobs**0.2)
        # Instead, we calculate bandwidth as in Stata 12
        iqre = stats.scoreatpercentile(e, 75) - stats.scoreatpercentile(e, 25)
        h = bandwidth(nobs, q)
        h = min(np.std(endog),
                iqre / 1.34) * (norm.ppf(q + h) - norm.ppf(q - h))

        fhat0 = 1. / (nobs * h) * np.sum(kernel(e / h))

        if vcov == 'robust':
            d = np.where(e > 0, (q/fhat0)**2, ((1-q)/fhat0)**2)
            xtxi = pinv(np.dot(exog.T, exog))
            xtdx = np.dot(exog.T * d[np.newaxis, :], exog)
            vcov = chain_dot(xtxi, xtdx, xtxi)
        elif vcov == 'iid':
            vcov = (1. / fhat0)**2 * q * (1 - q) * pinv(np.dot(exog.T, exog))
        else:
            raise Exception("vcov must be 'robust' or 'iid'")

        lfit = QuantRegResults(self, beta, normalized_cov_params=vcov)

        lfit.q = q
        lfit.iterations = n_iter
        lfit.sparsity = 1. / fhat0
        lfit.bandwidth = h
        lfit.history = history

        return RegressionResultsWrapper(lfit)