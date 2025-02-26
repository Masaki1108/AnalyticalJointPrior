import numpy as np
from scipy.special import erf

def calculate_Xp(s1x,s1y,s1z,s2x,s2y,s2z,q):

    a1 = np.sqrt(s1x**2+s1y**2+s1z**2)
    a2 = np.sqrt(s2x**2+s2y**2+s2z**2)
    cost1 = s1z/a1
    cost2 = s2z/a2
    sint1 = np.sqrt(1.-cost1**2)
    sint2 = np.sqrt(1.-cost2**2)

    return np.maximum(a1*sint1,((3.+4.*q)/(4.+3.*q))*q*a2*sint2)

def asym(x):
    return -np.exp(-x**2)/np.sqrt(np.pi)/x*(1.-1./(2.*x**2))

def calculate_Gaussian(x, mu, sigma2, low, high): 

    a = (low-mu)/np.sqrt(2*sigma2)
    b = (high-mu)/np.sqrt(2*sigma2)
    norm = np.sqrt(sigma2*np.pi/2)*(-erf(a) + erf(b))

    # If difference in error functions produce zero, overwrite with asymptotic expansion
    if np.isscalar(norm):
        if norm==0:
            norm = (np.sqrt(sigma2*np.pi/2)*(-asym(a) + asym(b)))
    elif np.any(norm==0):
        badInds = np.where(norm==0)
        norm[badInds] = (np.sqrt(sigma2*np.pi/2)*(-asym(a) + asym(b)))[badInds]

    # If differences remain zero, then our domain of interest (-1,1) is so many std. deviations
    # from the mean that our parametrization is unphysical. In this case, discount this hyperparameter.
    # This amounts to an additional condition in our hyperprior
    # NaNs occur when norm is infinitesimal, like 1e-322, such that 1/norm is set to inf and the exponential term is zero
    y = (1.0/norm)*np.exp((-1.0*(x-mu)*(x-mu))/(2.*sigma2))
    if np.any(norm==0) or np.any(y!=y):
        return np.zeros(x.size)

    else:
        y[x<low] = 0
        y[x>high] = 0
        return y

def calculate_Gaussian_2D(chiEff, chiP, mu_eff, sigma2_eff, mu_p, sigma2_p, cov, chi_min=-1): 

    """
    Function to evaluate our bivariate gaussian probability distribution on chiEff and chiP
    See e.g. http://mathworld.wolfram.com/BivariateNormalDistribution.html

    INPUTS
    chiEff:     Array of chi-effective values at which to evaluate probability distribution
    chiP:       Array of chi-p values
    mu_eff:     Mean of the BBH chi-effective distribution
    sigma2_eff: Variance of the BBH chi-effective distribution
    mu_p:       Mean of the BBH chi-p distribution
    sigma2_p:   Variance of the BBH chi-p distribution
    cov:        Degree of covariance (off-diagonal elements of the covariance matrix are cov*sigma_eff*sigma_p)

    RETURNS
    y:          Array of probability densities
    """

    dchi_p = 0.01
    dchi_eff = (1.-chi_min)/200

    chiEff_grid = np.arange(chi_min,1.+dchi_eff,dchi_eff)
    chiP_grid = np.arange(0.,1.+dchi_p,dchi_p)
    CHI_EFF,CHI_P = np.meshgrid(chiEff_grid,chiP_grid)


    # We need to truncate this distribution over the range chiEff=(-1,1) and chiP=(0,1)
    # Compute the correct normalization constant numerically, integrating over our precomputed grid from above
    norm_grid = np.exp(-0.5/(1.-cov**2.)*(
                    np.square(CHI_EFF-mu_eff)/sigma2_eff
                    + np.square(CHI_P-mu_p)/sigma2_p
                    - 2.*cov*(CHI_EFF-mu_eff)*(CHI_P-mu_p)/np.sqrt(sigma2_eff*sigma2_p)
                    ))
    norm = np.sum(norm_grid)*dchi_eff*dchi_p
    if norm<=1e-12:
        return np.zeros(chiEff.shape)

    # Now evaluate the gaussian at (chiEff,chiP)
    y = (1./norm)*np.exp(-0.5/(1.-cov**2.)*(
                            np.square(chiEff-mu_eff)/sigma2_eff
                            + np.square(chiP-mu_p)/sigma2_p
                            - (2.*cov)*(chiEff-mu_eff)*(chiP-mu_p)/np.sqrt(sigma2_eff*sigma2_p)
                            ))

    y[chiEff<chi_min] = 0.

    return y

