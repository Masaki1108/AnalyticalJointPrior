import numpy as np
import sys
from scipy.special import erfinv
from support import *

def inPrior_powerLawPeak(c,priorDict):

    """
    Helper function to implement prior on parameters of the Gaussian spin + power law peak model.
    """

    # Read parameters
    lmbda = c[0]
    mMax = c[1]
    m0 = c[2]
    sigM = c[3]
    fPeak = c[4]
    bq = c[5]
    kappa = c[6]
    mu_eff = c[7]
    sigma_eff = c[8]
    mu_p = c[9]
    sigma_p = c[10]
    rho = c[11]

    if lmbda<priorDict['lmbda'][0] or lmbda>priorDict['lmbda'][1]:
        return False
    elif mMax<priorDict['mMax'][0] or mMax>priorDict['mMax'][1]:
        return False
    elif m0<priorDict['m0'][0] or m0>priorDict['m0'][1]:
        return False
    elif sigM<priorDict['sigM'][0] or sigM>priorDict['sigM'][1]:
        return False
    elif fPeak<priorDict['fPeak'][0] or fPeak>priorDict['fPeak'][1]:
        return False
    elif bq<priorDict['bq'][0] or bq>priorDict['bq'][1]:
        return False
    elif mu_eff<priorDict['mu_eff'][0] or mu_eff>priorDict['mu_eff'][1]:
        return False
    elif sigma_eff<priorDict['sigma_eff'][0] or sigma_eff>priorDict['sigma_eff'][1]:
        return False
    elif mu_p<priorDict['mu_p'][0] or mu_p>priorDict['mu_p'][1]:
        return False
    elif sigma_p<priorDict['sigma_p'][0] or sigma_p>priorDict['sigma_p'][1]:
        return False
    elif rho<priorDict['rho'][0] or rho>priorDict['rho'][1]:
        return False
    else:
        return True

def dynesty_priorTransform_powerLawPeak(c,priorDict):

    # Read parameters
    lmbda = c[0]*(priorDict['lmbda'][1]-priorDict['lmbda'][0]) + priorDict['lmbda'][0]
    mMax = c[1]*(priorDict['mMax'][1]-priorDict['mMax'][0]) + priorDict['mMax'][0]
    m0 = c[2]*(priorDict['m0'][1]-priorDict['m0'][0]) + priorDict['m0'][0]
    sigM = c[3]*(priorDict['sigM'][1]-priorDict['sigM'][0]) + priorDict['sigM'][0]
    fPeak = c[4]*(priorDict['fPeak'][1]-priorDict['fPeak'][0]) + priorDict['fPeak'][0]
    bq = c[5]*(priorDict['bq'][1]-priorDict['bq'][0]) + priorDict['bq'][0]
    kappa = np.sqrt(2.)*priorDict['sig_kappa']*erfinv(-1.+2.*c[6])
    mu_eff = c[7]*(priorDict['mu_eff'][1]-priorDict['mu_eff'][0]) + priorDict['mu_eff'][0]
    sigma_eff = c[8]*(priorDict['sigma_eff'][1]-priorDict['sigma_eff'][0]) + priorDict['sigma_eff'][0]
    mu_p = c[9]*(priorDict['mu_p'][1]-priorDict['mu_p'][0]) + priorDict['mu_p'][0]
    sigma_p = c[10]*(priorDict['sigma_p'][1]-priorDict['sigma_p'][0]) + priorDict['sigma_p'][0]
    rho = c[11]*(priorDict['rho'][1]-priorDict['rho'][0]) + priorDict['rho'][0]

    return([lmbda,mMax,m0,sigM,fPeak,bq,kappa,mu_eff,sigma_eff,mu_p,sigma_p,rho])

def logp_powerLawPeak(c,sampleDict,injectionDict,priorDict,sampler='emcee',chi_min=-1):

    """
    Likelihood for gaussian spin + power-law peak model.

    INPUTS
    c: Array of proposed hyperparameters
    sampleDict: Dictionary of PE samples and prior weights for each event in consideration
    injectionDict: Dictionary of preprocessed injection information, as created by prep_injections.py
    priorDict: Dictionary defining boundaries on hyperparameters
    """

    logP = 0.

    # Read parameters
    lmbda = c[0]
    mMax = c[1]
    m0 = c[2]
    sigM = c[3]
    fPeak = c[4]
    bq = c[5]
    kappa = c[6]
    mu_eff = c[7]
    sigma_eff = c[8]
    mu_p = c[9]
    sigma_p = c[10]
    rho = c[11]

    # Call priors as necessary
    if sampler=="emcee":

        # Flat priors, reject samples past boundaries
        if not inPrior_powerLawPeak(c,priorDict):
            return -np.inf

        # Gaussian prior on kappa
        logP += -kappa**2./(2.*priorDict['sig_kappa']**2.)

    # Unpack injections
    m1_det = injectionDict['m1']
    m2_det = injectionDict['m2']
    z_det = injectionDict['z']
    dVdz_det = injectionDict['dVdz']
    Xeff_det = injectionDict['Xeff']
    Xp_det = injectionDict['Xp']
    pop_reweight = injectionDict['weights']
    q_det = m2_det/m1_det
    mtot_det = m1_det+m2_det

    mMin = priorDict['mMin']

    # Reweight injection probabilities
    p_det_Xeff_Xp = calculate_Gaussian_2D(Xeff_det, Xp_det, mu_eff, sigma_eff**2, mu_p, sigma_p**2., rho, chi_min=chi_min)
    p_det_m2 = (1.+bq)*np.power(m2_det,bq)/(np.power(m1_det,1.+bq)-mMin**(1.+bq))
    p_det_m2[m2_det<mMin] = 0.
    p_det_z = dVdz_det*np.power(1.+z_det,kappa-1.)

    # PLPeak distribution on p(m1)
    p_det_m1_pl = (1.+lmbda)*m1_det**lmbda/(mMax**(1.+lmbda) - mMin**(1.+lmbda))
    p_det_m1_pl[m1_det>mMax] = 0
    p_det_m1_peak = np.exp(-0.5*(m1_det-m0)**2./sigM**2)/np.sqrt(2.*np.pi*sigM**2.)
    p_det_m1 = fPeak*p_det_m1_peak + (1.-fPeak)*p_det_m1_pl

    # Construct full weighting factors
    det_weights = p_det_Xeff_Xp*p_det_m1*p_det_m2*p_det_z*pop_reweight
    if np.max(det_weights)==0:
        return -np.inf

    # Check for sufficient sampling size
    # Specifically require 4*Ndet effective detections, according to https://arxiv.org/abs/1904.10879
    nEvents = len(sampleDict)
    Nsamp = np.sum(det_weights)**2/np.sum(det_weights**2)
    if Nsamp<=4*nEvents:
        #if sampler=="emcee": print("Insufficient mock detections:",c)
        return -np.inf

    # Add detection efficiency term to log likelihood
    log_detEff = -nEvents*np.log(np.sum(det_weights))
    logP += log_detEff

    # Loop across samples
    for event in sampleDict:

        # Grab samples
        m1_sample = sampleDict[event]['m1']
        m2_sample = sampleDict[event]['m2']
        Xeff_sample = sampleDict[event]['Xeff']
        Xp_sample = sampleDict[event]['Xp']
        z_sample = sampleDict[event]['z']
        spin_prior = sampleDict[event]['joint_priors']
        weights = sampleDict[event]['weights']
        q_sample = m2_sample/m1_sample
        
        # Chi probability - Gaussian: P(chi_eff | mu, sigma2)
        p_Chi = calculate_Gaussian_2D(Xeff_sample, Xp_sample, mu_eff, sigma_eff**2., mu_p, sigma_p**2., rho, chi_min=chi_min)

        # p(m1)
        p_m1_pl = (1.+lmbda)*m1_sample**lmbda/(mMax**(1.+lmbda) - mMin**(1.+lmbda))
        p_m1_pl[m1_sample>mMax] = 0.
        p_m1_peak = np.exp(-0.5*(m1_sample-m0)**2./sigM**2)/np.sqrt(2.*np.pi*sigM**2.)
        p_m1 = fPeak*p_m1_peak + (1.-fPeak)*p_m1_pl
        old_m1_prior = np.ones(m1_sample.size)
        
        # p(m2)
        p_m2 = (1.+bq)*np.power(m2_sample,bq)/(np.power(m1_sample,1.+bq)-mMin**(1.+bq))
        old_m2_prior = np.ones(m2_sample.size)
        p_m2[m2_sample<mMin]=0

        # p(z)
        p_z = np.power(1.+z_sample,kappa-1.)
        old_pz_prior = (1.+z_sample)**(2.7-1.)
        
        # Evaluate marginalized likelihood
        nSamples = p_Chi.size

        integrand = p_Chi*p_m1*p_m2*p_z*weights/spin_prior/old_m1_prior/old_m2_prior/old_pz_prior
        neff_singleEvent = np.sum(integrand)**2/np.sum(integrand**2)
        if neff_singleEvent<10:
            #print(event,neff_singleEvent)
            return -np.inf
            
        pEvidence = np.sum(integrand)/nSamples

        # Summation
        logP += np.log(pEvidence)

    return logP

def inPrior_powerLawPeak_mMin(c,priorDict):

    """
    Helper function to implement prior on parameters of the Gaussian spin + power law peak model.
    """

    # Read parameters
    lmbda = c[0]
    mMin = c[1]
    mMax = c[2]
    m0 = c[3]
    sigM = c[4]
    fPeak = c[5]
    bq = c[6]
    kappa = c[7]
    mu_eff = c[8]
    sigma_eff = c[9]
    mu_p = c[10]
    sigma_p = c[11]
    rho = c[12]

    if lmbda<priorDict['lmbda'][0] or lmbda>priorDict['lmbda'][1]:
        return False
    elif mMin<priorDict['mMin'][0] or mMin>priorDict['mMin'][1]:
        return False
    elif mMax<priorDict['mMax'][0] or mMax>priorDict['mMax'][1]:
        return False
    elif m0<priorDict['m0'][0] or m0>priorDict['m0'][1]:
        return False
    elif sigM<priorDict['sigM'][0] or sigM>priorDict['sigM'][1]:
        return False
    elif fPeak<priorDict['fPeak'][0] or fPeak>priorDict['fPeak'][1]:
        return False
    elif bq<priorDict['bq'][0] or bq>priorDict['bq'][1]:
        return False
    elif mu_eff<priorDict['mu_eff'][0] or mu_eff>priorDict['mu_eff'][1]:
        return False
    elif sigma_eff<priorDict['sigma_eff'][0] or sigma_eff>priorDict['sigma_eff'][1]:
        return False
    elif mu_p<priorDict['mu_p'][0] or mu_p>priorDict['mu_p'][1]:
        return False
    elif sigma_p<priorDict['sigma_p'][0] or sigma_p>priorDict['sigma_p'][1]:
        return False
    elif rho<priorDict['rho'][0] or rho>priorDict['rho'][1]:
        return False
    else:
        return True

def logp_powerLawPeak_mMin(c,sampleDict,injectionDict,priorDict,sampler='emcee',chi_min=-1):

    """
    Likelihood for gaussian spin + power-law peak model.

    INPUTS
    c: Array of proposed hyperparameters
    sampleDict: Dictionary of PE samples and prior weights for each event in consideration
    injectionDict: Dictionary of preprocessed injection information, as created by prep_injections.py
    priorDict: Dictionary defining boundaries on hyperparameters
    """

    logP = 0.

    # Read parameters
    lmbda = c[0]
    mMin = c[1]
    mMax = c[2]
    m0 = c[3]
    sigM = c[4]
    fPeak = c[5]
    bq = c[6]
    kappa = c[7]
    mu_eff = c[8]
    sigma_eff = c[9]
    mu_p = c[10]
    sigma_p = c[11]
    rho = c[12]

    # Call priors as necessary
    if sampler=="emcee":

        # Flat priors, reject samples past boundaries
        if not inPrior_powerLawPeak_mMin(c,priorDict):
            return -np.inf

        # Gaussian prior on kappa
        logP += -kappa**2./(2.*priorDict['sig_kappa']**2.)

    # Unpack injections
    m1_det = injectionDict['m1']
    m2_det = injectionDict['m2']
    z_det = injectionDict['z']
    dVdz_det = injectionDict['dVdz']
    Xeff_det = injectionDict['Xeff']
    Xp_det = injectionDict['Xp']
    pop_reweight = injectionDict['weights']
    q_det = m2_det/m1_det
    mtot_det = m1_det+m2_det

    # Reweight injection probabilities
    p_det_Xeff_Xp = calculate_Gaussian_2D(Xeff_det, Xp_det, mu_eff, sigma_eff**2, mu_p, sigma_p**2., rho, chi_min=chi_min)
    p_det_m2 = (1.+bq)*np.power(m2_det,bq)/(np.power(m1_det,1.+bq)-mMin**(1.+bq))
    p_det_m2[m2_det<mMin] = 0.
    p_det_z = dVdz_det*np.power(1.+z_det,kappa-1.)

    # PLPeak distribution on p(m1)
    p_det_m1_pl = (1.+lmbda)*m1_det**lmbda/(mMax**(1.+lmbda) - mMin**(1.+lmbda))
    p_det_m1_pl[m1_det>mMax] = 0
    p_det_m1_peak = np.exp(-0.5*(m1_det-m0)**2./sigM**2)/np.sqrt(2.*np.pi*sigM**2.)
    p_det_m1 = fPeak*p_det_m1_peak + (1.-fPeak)*p_det_m1_pl

    # Construct full weighting factors
    det_weights = p_det_Xeff_Xp*p_det_m1*p_det_m2*p_det_z*pop_reweight
    if np.max(det_weights)==0:
        return -np.inf

    # Check for sufficient sampling size
    # Specifically require 4*Ndet effective detections, according to https://arxiv.org/abs/1904.10879
    nEvents = len(sampleDict)
    Nsamp = np.sum(det_weights)**2/np.sum(det_weights**2)
    if Nsamp<=4*nEvents:
        if sampler=="emcee": print("Insufficient mock detections:",c)
        return -np.inf

    # Add detection efficiency term to log likelihood
    log_detEff = -nEvents*np.log(np.sum(det_weights))
    logP += log_detEff

    # Loop across samples
    for event in sampleDict:

        # Grab samples
        m1_sample = sampleDict[event]['m1']
        m2_sample = sampleDict[event]['m2']
        Xeff_sample = sampleDict[event]['Xeff']
        Xp_sample = sampleDict[event]['Xp']
        z_sample = sampleDict[event]['z']
        spin_prior = sampleDict[event]['joint_priors']
        weights = sampleDict[event]['weights']
        q_sample = m2_sample/m1_sample
        
        # Chi probability - Gaussian: P(chi_eff | mu, sigma2)
        p_Chi = calculate_Gaussian_2D(Xeff_sample, Xp_sample, mu_eff, sigma_eff**2., mu_p, sigma_p**2., rho, chi_min=chi_min)

        # p(m1)
        p_m1_pl = (1.+lmbda)*m1_sample**lmbda/(mMax**(1.+lmbda) - mMin**(1.+lmbda))
        p_m1_pl[m1_sample>mMax] = 0.
        p_m1_peak = np.exp(-0.5*(m1_sample-m0)**2./sigM**2)/np.sqrt(2.*np.pi*sigM**2.)
        p_m1 = fPeak*p_m1_peak + (1.-fPeak)*p_m1_pl
        old_m1_prior = np.ones(m1_sample.size)
        
        # p(m2)
        p_m2 = (1.+bq)*np.power(m2_sample,bq)/(np.power(m1_sample,1.+bq)-mMin**(1.+bq))
        old_m2_prior = np.ones(m2_sample.size)
        p_m2[m2_sample<mMin]=0

        # p(z)
        p_z = np.power(1.+z_sample,kappa-1.)
        old_pz_prior = (1.+z_sample)**(2.7-1.)
        
        # Evaluate marginalized likelihood
        nSamples = p_Chi.size

        integrand = p_Chi*p_m1*p_m2*p_z*weights/spin_prior/old_m1_prior/old_m2_prior/old_pz_prior
        neff_singleEvent = np.sum(integrand)**2/np.sum(integrand**2)
        if neff_singleEvent<10:
            print(event,neff_singleEvent)
            return -np.inf
            
        pEvidence = np.sum(integrand)/nSamples

        # Summation
        logP += np.log(pEvidence)

    print(logP)
    return logP

def dynesty_priorTransform_mixtureModel(c,priorDict):

    # Read parameters
    lmbda = c[0]*(priorDict['lmbda'][1]-priorDict['lmbda'][0]) + priorDict['lmbda'][0]
    mMax = c[1]*(priorDict['mMax'][1]-priorDict['mMax'][0]) + priorDict['mMax'][0]
    m0 = c[2]*(priorDict['m0'][1]-priorDict['m0'][0]) + priorDict['m0'][0]
    sigM = c[3]*(priorDict['sigM'][1]-priorDict['sigM'][0]) + priorDict['sigM'][0]
    fPeak = c[4]*(priorDict['fPeak'][1]-priorDict['fPeak'][0]) + priorDict['fPeak'][0]
    bq = c[5]*(priorDict['bq'][1]-priorDict['bq'][0]) + priorDict['bq'][0]
    kappa = np.sqrt(2.)*priorDict['sig_kappa']*erfinv(-1.+2.*c[6])
    mu_eff = c[7]*(priorDict['mu_eff'][1]-priorDict['mu_eff'][0]) + priorDict['mu_eff'][0]
    sigma_eff = c[8]*(priorDict['sigma_eff'][1]-priorDict['sigma_eff'][0]) + priorDict['sigma_eff'][0]
    zeta = c[9]*(priorDict['zeta'][1]-priorDict['zeta'][0]) + priorDict['zeta'][0]

    return([lmbda,mMax,m0,sigM,fPeak,bq,kappa,mu_eff,sigma_eff,zeta])

def logp_mixtureModel(c,sampleDict,injectionDict,priorDict,sampler='emcee',zero_width=0.005,chi_min=-1,both=False):

    """
    Likelihood for gaussian spin + power-law peak model.

    INPUTS
    c: Array of proposed hyperparameters
    sampleDict: Dictionary of PE samples and prior weights for each event in consideration
    injectionDict: Dictionary of preprocessed injection information, as created by prep_injections.py
    priorDict: Dictionary defining boundaries on hyperparameters
    """

    logP = 0.

    # Read parameters
    lmbda = c[0]
    mMax = c[1]
    m0 = c[2]
    sigM = c[3]
    fPeak = c[4]
    bq = c[5]
    kappa = c[6]
    mu_eff = c[7]
    sigma_eff = c[8]
    zeta = c[9]

    # Call priors as necessary
    if sampler=="emcee":

        # Flat priors, reject samples past boundaries
        if not inPrior_mixtureModel(c,priorDict):
            return -np.inf

        # Gaussian prior on kappa
        logP += -kappa**2./(2.*priorDict['sig_kappa']**2.)

    # Unpack injections
    m1_det = injectionDict['m1']
    m2_det = injectionDict['m2']
    z_det = injectionDict['z']
    dVdz_det = injectionDict['dVdz']
    Xeff_det = injectionDict['Xeff']
    pop_reweight = injectionDict['weights_XeffOnly']
    q_det = m2_det/m1_det
    mtot_det = m1_det+m2_det

    mMin = priorDict['mMin']

    # Reweight injection probabilities
    if both==False:
        p_det_Xeff = zeta*calculate_Gaussian(Xeff_det, mu_eff, sigma_eff**2, chi_min, 1)\
                + (1.-zeta)*calculate_Gaussian(Xeff_det, 0, zero_width**2, -1, 1)
    else:
        p_det_Xeff = zeta*calculate_Gaussian(Xeff_det, mu_eff, sigma_eff**2, chi_min, 1)\
                + (1.-zeta)*calculate_Gaussian(Xeff_det, 0, zero_width**2, chi_min, 1)

    p_det_m2 = (1.+bq)*np.power(m2_det,bq)/(np.power(m1_det,1.+bq)-mMin**(1.+bq))
    p_det_m2[m2_det<mMin] = 0.
    p_det_z = dVdz_det*np.power(1.+z_det,kappa-1.)

    # PLPeak distribution on p(m1)
    p_det_m1_pl = (1.+lmbda)*m1_det**lmbda/(mMax**(1.+lmbda) - mMin**(1.+lmbda))
    p_det_m1_pl[m1_det>mMax] = 0
    p_det_m1_peak = np.exp(-0.5*(m1_det-m0)**2./sigM**2)/np.sqrt(2.*np.pi*sigM**2.)
    p_det_m1 = fPeak*p_det_m1_peak + (1.-fPeak)*p_det_m1_pl

    # Construct full weighting factors
    det_weights = p_det_Xeff*p_det_m1*p_det_m2*p_det_z*pop_reweight
    if np.max(det_weights)==0:
        return -np.inf

    # Check for sufficient sampling size
    # Specifically require 4*Ndet effective detections, according to https://arxiv.org/abs/1904.10879
    nEvents = len(sampleDict)
    Nsamp = np.sum(det_weights)**2/np.sum(det_weights**2)
    if Nsamp<=4*nEvents:
        if sampler=="emcee": print("Insufficient mock detections:",c)
        return -np.inf

    # Add detection efficiency term to log likelihood
    log_detEff = -nEvents*np.log(np.sum(det_weights))
    logP += log_detEff

    # Loop across samples
    for event in sampleDict:

        # Grab samples
        m1_sample = sampleDict[event]['m1']
        m2_sample = sampleDict[event]['m2']
        Xeff_sample = sampleDict[event]['Xeff']
        z_sample = sampleDict[event]['z']
        spin_prior = sampleDict[event]['Xeff_priors']
        weights = sampleDict[event]['weights']
        q_sample = m2_sample/m1_sample
        
        # Chi probability - Gaussian: P(chi_eff | mu, sigma2)
        if both==False:
            p_Chi = zeta*calculate_Gaussian(Xeff_sample, mu_eff, sigma_eff**2, chi_min, 1)\
                    + (1.-zeta)*calculate_Gaussian(Xeff_sample, 0, zero_width**2, -1, 1)
        else:
            p_Chi = zeta*calculate_Gaussian(Xeff_sample, mu_eff, sigma_eff**2, chi_min, 1)\
                    + (1.-zeta)*calculate_Gaussian(Xeff_sample, 0, zero_width**2, chi_min, 1)

        # p(m1)
        p_m1_pl = (1.+lmbda)*m1_sample**lmbda/(mMax**(1.+lmbda) - mMin**(1.+lmbda))
        p_m1_pl[m1_sample>mMax] = 0.
        p_m1_peak = np.exp(-0.5*(m1_sample-m0)**2./sigM**2)/np.sqrt(2.*np.pi*sigM**2.)
        p_m1 = fPeak*p_m1_peak + (1.-fPeak)*p_m1_pl
        old_m1_prior = np.ones(m1_sample.size)
        
        # p(m2)
        p_m2 = (1.+bq)*np.power(m2_sample,bq)/(np.power(m1_sample,1.+bq)-mMin**(1.+bq))
        old_m2_prior = np.ones(m2_sample.size)
        p_m2[m2_sample<mMin]=0

        # p(z)
        p_z = np.power(1.+z_sample,kappa-1.)
        old_pz_prior = (1.+z_sample)**(2.7-1.)
        
        # Evaluate marginalized likelihood
        nSamples = p_Chi.size
        pEvidence = np.sum(p_Chi*p_m1*p_m2*p_z*weights/spin_prior/old_m1_prior/old_m2_prior/old_pz_prior)/nSamples

        # Summation
        logP += np.log(pEvidence)

    print(logP)
    return logP

def inPrior_mixtureModel_variableMin(c,priorDict):

    """
    Helper function to implement prior on parameters of the Gaussian spin + power law peak model with a variable truncation bound.
    """

    # Read parameters
    lmbda = c[0]
    mMax = c[1]
    m0 = c[2]
    sigM = c[3]
    fPeak = c[4]
    bq = c[5]
    kappa = c[6]
    mu_eff = c[7]
    sigma_eff = c[8]
    zeta = c[9]
    chi_min = c[10]

    if lmbda<priorDict['lmbda'][0] or lmbda>priorDict['lmbda'][1]:
        return False
    elif mMax<priorDict['mMax'][0] or mMax>priorDict['mMax'][1]:
        return False
    elif m0<priorDict['m0'][0] or m0>priorDict['m0'][1]:
        return False
    elif sigM<priorDict['sigM'][0] or sigM>priorDict['sigM'][1]:
        return False
    elif fPeak<priorDict['fPeak'][0] or fPeak>priorDict['fPeak'][1]:
        return False
    elif bq<priorDict['bq'][0] or bq>priorDict['bq'][1]:
        return False
    elif mu_eff<priorDict['mu'][0] or mu_eff>priorDict['mu'][1]:
        return False
    elif sigma_eff<priorDict['sigma'][0] or sigma_eff>priorDict['sigma'][1]:
        return False
    elif zeta<priorDict['zeta'][0] or zeta>priorDict['zeta'][1]:
        return False
    elif chi_min<priorDict['chi_min'][0] or chi_min>priorDict['chi_min'][1]:
        return False
    elif chi_min>mu_eff:
        return False
    else:
        return True

def logp_mixtureModel_variableMin(c,sampleDict,injectionDict,priorDict,sampler='emcee',zero_width=0.005,both=False):

    """
    Likelihood for gaussian spin + power-law peak model.

    INPUTS
    c: Array of proposed hyperparameters
    sampleDict: Dictionary of PE samples and prior weights for each event in consideration
    injectionDict: Dictionary of preprocessed injection information, as created by prep_injections.py
    priorDict: Dictionary defining boundaries on hyperparameters
    """

    logP = 0.

    # Read parameters
    lmbda = c[0]
    mMax = c[1]
    m0 = c[2]
    sigM = c[3]
    fPeak = c[4]
    bq = c[5]
    kappa = c[6]
    mu_eff = c[7]
    sigma_eff = c[8]
    zeta = c[9]
    chi_min = c[10]

    # Call priors as necessary
    if sampler=="emcee":

        # Flat priors, reject samples past boundaries
        if not inPrior_mixtureModel_variableMin(c,priorDict):
            return -np.inf

        # Gaussian prior on kappa
        logP += -kappa**2./(2.*priorDict['sig_kappa']**2.)

    # Unpack injections
    m1_det = injectionDict['m1']
    m2_det = injectionDict['m2']
    z_det = injectionDict['z']
    dVdz_det = injectionDict['dVdz']
    Xeff_det = injectionDict['Xeff']
    pop_reweight = injectionDict['weights_XeffOnly']
    q_det = m2_det/m1_det
    mtot_det = m1_det+m2_det

    mMin = priorDict['mMin']

    # Reweight injection probabilities
    if both==False:
        p_det_Xeff = zeta*calculate_Gaussian(Xeff_det, mu_eff, sigma_eff**2, chi_min, 1)\
                + (1.-zeta)*calculate_Gaussian(Xeff_det, 0, zero_width**2, -1, 1)
    else:
        p_det_Xeff = zeta*calculate_Gaussian(Xeff_det, mu_eff, sigma_eff**2, chi_min, 1)\
                + (1.-zeta)*calculate_Gaussian(Xeff_det, 0, zero_width**2, chi_min, 1)
    p_det_m2 = (1.+bq)*np.power(m2_det,bq)/(np.power(m1_det,1.+bq)-mMin**(1.+bq))
    p_det_m2[m2_det<mMin] = 0.
    p_det_z = dVdz_det*np.power(1.+z_det,kappa-1.)

    # PLPeak distribution on p(m1)
    p_det_m1_pl = (1.+lmbda)*m1_det**lmbda/(mMax**(1.+lmbda) - mMin**(1.+lmbda))
    p_det_m1_pl[m1_det>mMax] = 0
    p_det_m1_peak = np.exp(-0.5*(m1_det-m0)**2./sigM**2)/np.sqrt(2.*np.pi*sigM**2.)
    p_det_m1 = fPeak*p_det_m1_peak + (1.-fPeak)*p_det_m1_pl

    # Construct full weighting factors
    det_weights = p_det_Xeff*p_det_m1*p_det_m2*p_det_z*pop_reweight
    if np.max(det_weights)==0:
        return -np.inf

    # Check for sufficient sampling size
    # Specifically require 4*Ndet effective detections, according to https://arxiv.org/abs/1904.10879
    nEvents = len(sampleDict)
    Nsamp = np.sum(det_weights)**2/np.sum(det_weights**2)
    if Nsamp<=4*nEvents:
        if sampler=="emcee": print("Insufficient mock detections:",c)
        return -np.inf

    # Add detection efficiency term to log likelihood
    log_detEff = -nEvents*np.log(np.sum(det_weights))
    logP += log_detEff

    # Loop across samples
    for event in sampleDict:

        # Grab samples
        m1_sample = sampleDict[event]['m1']
        m2_sample = sampleDict[event]['m2']
        Xeff_sample = sampleDict[event]['Xeff']
        z_sample = sampleDict[event]['z']
        spin_prior = sampleDict[event]['Xeff_priors']
        weights = sampleDict[event]['weights']
        q_sample = m2_sample/m1_sample
        
        # Chi probability - Gaussian: P(chi_eff | mu, sigma2)
        if both==False:
            p_Chi = zeta*calculate_Gaussian(Xeff_sample, mu_eff, sigma_eff**2, chi_min, 1)\
                    + (1.-zeta)*calculate_Gaussian(Xeff_sample, 0, zero_width**2, -1, 1)
        else:
            p_Chi = zeta*calculate_Gaussian(Xeff_sample, mu_eff, sigma_eff**2, chi_min, 1)\
                    + (1.-zeta)*calculate_Gaussian(Xeff_sample, 0, zero_width**2, chi_min, 1)

        # p(m1)
        p_m1_pl = (1.+lmbda)*m1_sample**lmbda/(mMax**(1.+lmbda) - mMin**(1.+lmbda))
        p_m1_pl[m1_sample>mMax] = 0.
        p_m1_peak = np.exp(-0.5*(m1_sample-m0)**2./sigM**2)/np.sqrt(2.*np.pi*sigM**2.)
        p_m1 = fPeak*p_m1_peak + (1.-fPeak)*p_m1_pl
        old_m1_prior = np.ones(m1_sample.size)
        
        # p(m2)
        p_m2 = (1.+bq)*np.power(m2_sample,bq)/(np.power(m1_sample,1.+bq)-mMin**(1.+bq))
        old_m2_prior = np.ones(m2_sample.size)
        p_m2[m2_sample<mMin]=0

        # p(z)
        p_z = np.power(1.+z_sample,kappa-1.)
        old_pz_prior = (1.+z_sample)**(2.7-1.)
        
        # Evaluate marginalized likelihood
        nSamples = p_Chi.size
        pEvidence = np.sum(p_Chi*p_m1*p_m2*p_z*weights/spin_prior/old_m1_prior/old_m2_prior/old_pz_prior)/nSamples

        # Summation
        logP += np.log(pEvidence)

    return logP

def inPrior_powerLawPeak_variableChiMin(c,priorDict):

    """
    Helper function to implement prior on parameters of the Gaussian spin + power law peak model with a variable truncation bound.
    """

    # Read parameters
    lmbda = c[0]
    mMax = c[1]
    m0 = c[2]
    sigM = c[3]
    fPeak = c[4]
    bq = c[5]
    kappa = c[6]
    mu_eff = c[7]
    sigma_eff = c[8]
    mu_p = c[9]
    sigma_p = c[10]
    rho = c[11]
    chi_min = c[12]

    if lmbda<priorDict['lmbda'][0] or lmbda>priorDict['lmbda'][1]:
        return False
    elif mMax<priorDict['mMax'][0] or mMax>priorDict['mMax'][1]:
        return False
    elif m0<priorDict['m0'][0] or m0>priorDict['m0'][1]:
        return False
    elif sigM<priorDict['sigM'][0] or sigM>priorDict['sigM'][1]:
        return False
    elif fPeak<priorDict['fPeak'][0] or fPeak>priorDict['fPeak'][1]:
        return False
    elif bq<priorDict['bq'][0] or bq>priorDict['bq'][1]:
        return False
    elif mu_eff<priorDict['mu_eff'][0] or mu_eff>priorDict['mu_eff'][1]:
        return False
    elif sigma_eff<priorDict['sigma_eff'][0] or sigma_eff>priorDict['sigma_eff'][1]:
        return False
    elif mu_p<priorDict['mu_p'][0] or mu_p>priorDict['mu_p'][1]:
        return False
    elif sigma_p<priorDict['sigma_p'][0] or sigma_p>priorDict['sigma_p'][1]:
        return False
    elif rho<priorDict['rho'][0] or rho>priorDict['rho'][1]:
        return False
    elif chi_min<priorDict['chi_min'][0] or chi_min>priorDict['chi_min'][1]:
        return False
    elif chi_min>mu_eff:
        return False
    else:
        return True

def logp_powerLawPeak_variableChiMin(c,sampleDict,injectionDict,priorDict):

    """
    Likelihood for gaussian spin + power-law peak model with a variable truncation bound on chi_eff

    INPUTS
    c: Array of proposed hyperparameters
    sampleDict: Dictionary of PE samples and prior weights for each event in consideration
    injectionDict: Dictionary of preprocessed injection information, as created by prep_injections.py
    priorDict: Dictionary defining boundaries on hyperparameters
    """

    # Flat priors, reject samples past boundaries
    if not inPrior_powerLawPeak_variableChiMin(c,priorDict):
        return -np.inf

    # If sample in prior range, evaluate
    else:

        # Read parameters
        lmbda = c[0]
        mMax = c[1]
        m0 = c[2]
        sigM = c[3]
        fPeak = c[4]
        bq = c[5]
        kappa = c[6]
        mu_eff = c[7]
        sigma_eff = c[8]
        mu_p = c[9]
        sigma_p = c[10]
        rho = c[11]
        chi_min = c[12]

        # Gaussian prior on kappa
        logP = 0.
        logP += -kappa**2./(2.*priorDict['sig_kappa']**2.)

        # Unpack injections
        m1_det = injectionDict['m1']
        m2_det = injectionDict['m2']
        z_det = injectionDict['z']
        dVdz_det = injectionDict['dVdz']
        Xeff_det = injectionDict['Xeff']
        Xp_det = injectionDict['Xp']
        pop_reweight = injectionDict['weights']
        q_det = m2_det/m1_det
        mtot_det = m1_det+m2_det

        mMin = priorDict['mMin']

        # Reweight injection probabilities
        p_det_Xeff_Xp = calculate_Gaussian_2D(Xeff_det, Xp_det, mu_eff, sigma_eff**2, mu_p, sigma_p**2., rho, chi_min=chi_min)
        p_det_m2 = (1.+bq)*np.power(m2_det,bq)/(np.power(m1_det,1.+bq)-mMin**(1.+bq))
        p_det_z = dVdz_det*np.power(1.+z_det,kappa-1.)

        # PLPeak distribution on p(m1)
        p_det_m1_pl = (1.+lmbda)*m1_det**lmbda/(mMax**(1.+lmbda) - mMin**(1.+lmbda))
        p_det_m1_pl[m1_det>mMax] = 0
        p_det_m1_peak = np.exp(-0.5*(m1_det-m0)**2./sigM**2)/np.sqrt(2.*np.pi*sigM**2.)
        p_det_m1 = fPeak*p_det_m1_peak + (1.-fPeak)*p_det_m1_pl

        # Construct full weighting factors
        det_weights = p_det_Xeff_Xp*p_det_m1*p_det_m2*p_det_z*pop_reweight
        if np.max(det_weights)==0:
            return -np.inf

        # Check for sufficient effective sample size
        # Specifically require 4*Ndet effective detections, according to https://arxiv.org/abs/1904.10879
        nEvents = len(sampleDict)
        Nsamp = np.sum(det_weights)**2/np.sum(det_weights**2)
        if Nsamp<=4*nEvents:
            print("Insufficient mock detections:",c)
            return -np.inf

        # Add selection term to log-likelihood
        log_detEff = -nEvents*np.log(np.sum(det_weights))
        logP += log_detEff

        # Loop across samples
        for event in sampleDict:

            # Grab samples
            m1_sample = sampleDict[event]['m1']
            m2_sample = sampleDict[event]['m2']
            Xeff_sample = sampleDict[event]['Xeff']
            Xp_sample = sampleDict[event]['Xp']
            z_sample = sampleDict[event]['z']
            spin_prior = sampleDict[event]['joint_priors']
            weights = sampleDict[event]['weights']
            q_sample = m2_sample/m1_sample
            
            # Effective spin probability
            p_Chi = calculate_Gaussian_2D(Xeff_sample, Xp_sample, mu_eff, sigma_eff**2., mu_p, sigma_p**2., rho, chi_min=chi_min)

            # p(m1)
            p_m1_pl = (1.+lmbda)*m1_sample**lmbda/(mMax**(1.+lmbda) - mMin**(1.+lmbda))
            p_m1_pl[m1_sample>mMax] = 0.
            p_m1_peak = np.exp(-0.5*(m1_sample-m0)**2./sigM**2)/np.sqrt(2.*np.pi*sigM**2.)
            p_m1 = fPeak*p_m1_peak + (1.-fPeak)*p_m1_pl
            old_m1_prior = np.ones(m1_sample.size)
            
            # p(m2)
            p_m2 = (1.+bq)*np.power(m2_sample,bq)/(np.power(m1_sample,1.+bq)-mMin**(1.+bq))
            old_m2_prior = np.ones(m2_sample.size)
            p_m2[m2_sample<mMin]=0

            # p(z)
            p_z = np.power(1.+z_sample,kappa-1.)
            old_pz_prior = (1.+z_sample)**(2.7-1.)
            
            # Evaluate marginalized likelihood
            nSamples = p_Chi.size
            pEvidence = np.sum(p_Chi*p_m1*p_m2*p_z*weights/spin_prior/old_m1_prior/old_m2_prior/old_pz_prior)/nSamples

            # Summation
            logP += np.log(pEvidence)

        return logP

def inPrior_powerLawPeak_variableChiMin_chiEffOnly(c,priorDict):

    """
    Helper function to implement prior on parameters of the Gaussian spin + power law peak model with a variable truncation bound,
    using only the marginal chi_effective distribution
    """

    # Read parameters
    lmbda = c[0]
    mMax = c[1]
    m0 = c[2]
    sigM = c[3]
    fPeak = c[4]
    bq = c[5]
    kappa = c[6]
    mu_eff = c[7]
    sigma_eff = c[8]
    chi_min = c[9]

    if lmbda<priorDict['lmbda'][0] or lmbda>priorDict['lmbda'][1]:
        return False
    elif mMax<priorDict['mMax'][0] or mMax>priorDict['mMax'][1]:
        return False
    elif m0<priorDict['m0'][0] or m0>priorDict['m0'][1]:
        return False
    elif sigM<priorDict['sigM'][0] or sigM>priorDict['sigM'][1]:
        return False
    elif fPeak<priorDict['fPeak'][0] or fPeak>priorDict['fPeak'][1]:
        return False
    elif bq<priorDict['bq'][0] or bq>priorDict['bq'][1]:
        return False
    elif mu_eff<priorDict['mu_eff'][0] or mu_eff>priorDict['mu_eff'][1]:
        return False
    elif sigma_eff<priorDict['sigma_eff'][0] or sigma_eff>priorDict['sigma_eff'][1]:
        return False
    elif chi_min<priorDict['chi_min'][0] or chi_min>priorDict['chi_min'][1]:
        return False
    elif chi_min>mu_eff:
        return False
    else:
        return True

def dynesty_priorTransform_variableChiMin(c,priorDict):

    # Read parameters
    lmbda = c[0]*(priorDict['lmbda'][1]-priorDict['lmbda'][0]) + priorDict['lmbda'][0]
    mMax = c[1]*(priorDict['mMax'][1]-priorDict['mMax'][0]) + priorDict['mMax'][0]
    m0 = c[2]*(priorDict['m0'][1]-priorDict['m0'][0]) + priorDict['m0'][0]
    sigM = c[3]*(priorDict['sigM'][1]-priorDict['sigM'][0]) + priorDict['sigM'][0]
    fPeak = c[4]*(priorDict['fPeak'][1]-priorDict['fPeak'][0]) + priorDict['fPeak'][0]
    bq = c[5]*(priorDict['bq'][1]-priorDict['bq'][0]) + priorDict['bq'][0]
    kappa = np.sqrt(2.)*priorDict['sig_kappa']*erfinv(-1.+2.*c[6])
    mu_eff = c[7]*(priorDict['mu_eff'][1]-priorDict['mu_eff'][0]) + priorDict['mu_eff'][0]
    sigma_eff = c[8]*(priorDict['sigma_eff'][1]-priorDict['sigma_eff'][0]) + priorDict['sigma_eff'][0]
    chi_min = c[9]*(min(priorDict['chi_min'][1],mu_eff) - priorDict['chi_min'][0]) + priorDict['chi_min'][0]

    return([lmbda,mMax,m0,sigM,fPeak,bq,kappa,mu_eff,sigma_eff,chi_min])


def logp_powerLawPeak_variableChiMin_chiEffOnly(c,sampleDict,injectionDict,priorDict,sampler='emcee'):

    """
    Likelihood for gaussian spin + power-law peak model with a variable truncation bound on chi_eff,
    using only the marginal chi_effective distribution

    INPUTS
    c: Array of proposed hyperparameters
    sampleDict: Dictionary of PE samples and prior weights for each event in consideration
    injectionDict: Dictionary of preprocessed injection information, as created by prep_injections.py
    priorDict: Dictionary defining boundaries on hyperparameters
    """

    logP = 0

    # Read parameters
    lmbda = c[0]
    mMax = c[1]
    m0 = c[2]
    sigM = c[3]
    fPeak = c[4]
    bq = c[5]
    kappa = c[6]
    mu_eff = c[7]
    sigma_eff = c[8]
    chi_min = c[9]

    # Call priors as necessary
    if sampler=="emcee":

        # Flat priors, reject samples past boundaries
        if not inPrior_powerLawPeak_variableChiMin_chiEffOnly(c,priorDict):
            return -np.inf

        # Gaussian prior on kappa
        logP += -kappa**2./(2.*priorDict['sig_kappa']**2.)

    # Unpack injections
    m1_det = injectionDict['m1']
    m2_det = injectionDict['m2']
    z_det = injectionDict['z']
    dVdz_det = injectionDict['dVdz']
    Xeff_det = injectionDict['Xeff']
    pop_reweight = injectionDict['weights_XeffOnly']
    q_det = m2_det/m1_det
    mtot_det = m1_det+m2_det

    mMin = priorDict['mMin']

    # Reweight injection probabilities
    p_det_Xeff = calculate_Gaussian(Xeff_det, mu_eff, sigma_eff**2., chi_min, 1.)
    p_det_m2 = (1.+bq)*np.power(m2_det,bq)/(np.power(m1_det,1.+bq)-mMin**(1.+bq))
    p_det_m2[m2_det<=mMin] = 0.
    p_det_z = dVdz_det*np.power(1.+z_det,kappa-1.)

    # PLPeak distribution on p(m1)
    p_det_m1_pl = (1.+lmbda)*m1_det**lmbda/(mMax**(1.+lmbda) - mMin**(1.+lmbda))
    p_det_m1_pl[m1_det>mMax] = 0
    p_det_m1_peak = np.exp(-0.5*(m1_det-m0)**2./sigM**2)/np.sqrt(2.*np.pi*sigM**2.)
    p_det_m1 = fPeak*p_det_m1_peak + (1.-fPeak)*p_det_m1_pl
    p_det_m1[m1_det<=mMin] = 0

    # Construct full weighting factors
    det_weights = p_det_Xeff*p_det_m1*p_det_m2*p_det_z*pop_reweight
    if np.max(det_weights)==0:
        return -np.inf

    # Check for sufficient effective sample size
    # Specifically require 4*Ndet effective detections, according to https://arxiv.org/abs/1904.10879
    nEvents = len(sampleDict)
    Nsamp = np.sum(det_weights)**2/np.sum(det_weights**2)
    if Nsamp<=4*nEvents:
        #print("Insufficient mock detections:",c)
        return -np.inf

    # Add selection term to log-likelihood
    log_detEff = -nEvents*np.log(np.sum(det_weights))
    logP += log_detEff

    # Loop across samples
    for event in sampleDict:

        # Grab samples
        m1_sample = sampleDict[event]['m1']
        m2_sample = sampleDict[event]['m2']
        Xeff_sample = sampleDict[event]['Xeff']
        z_sample = sampleDict[event]['z']
        Xeff_prior = sampleDict[event]['Xeff_priors']
        weights = sampleDict[event]['weights']
        q_sample = m2_sample/m1_sample
        
        # Effective spin probability
        p_Chi = calculate_Gaussian(Xeff_sample, mu_eff, sigma_eff**2., chi_min, 1.)

        # p(m1)
        p_m1_pl = (1.+lmbda)*m1_sample**lmbda/(mMax**(1.+lmbda) - mMin**(1.+lmbda))
        p_m1_pl[m1_sample>mMax] = 0.
        p_m1_peak = np.exp(-0.5*(m1_sample-m0)**2./sigM**2)/np.sqrt(2.*np.pi*sigM**2.)
        p_m1 = fPeak*p_m1_peak + (1.-fPeak)*p_m1_pl
        old_m1_prior = np.ones(m1_sample.size)
        
        # p(m2)
        p_m2 = (1.+bq)*np.power(m2_sample,bq)/(np.power(m1_sample,1.+bq)-mMin**(1.+bq))
        old_m2_prior = np.ones(m2_sample.size)
        p_m2[m2_sample<mMin]=0

        # p(z)
        p_z = np.power(1.+z_sample,kappa-1.)
        old_pz_prior = (1.+z_sample)**(2.7-1.)
        
        # Evaluate marginalized likelihood
        nSamples = p_Chi.size
        pEvidence = np.sum(p_Chi*p_m1*p_m2*p_z*weights/Xeff_prior/old_m1_prior/old_m2_prior/old_pz_prior)/nSamples

        # Summation
        logP += np.log(pEvidence)

    print(c,logP)
    return logP
