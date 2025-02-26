import numpy as np
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
    mu0 = c[7]
    log_sigma0 = c[8]
    alpha = c[9]
    beta = c[10]

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
    elif mu0<priorDict['mu0'][0] or mu0>priorDict['mu0'][1]:
        return False
    elif log_sigma0<priorDict['log_sigma0'][0] or log_sigma0>priorDict['log_sigma0'][1]:
        return False
    elif alpha<priorDict['alpha'][0] or alpha>priorDict['alpha'][1]:
        return False
    elif beta<priorDict['beta'][0] or beta>priorDict['beta'][1]:
        return False
    #elif log_sigma0+beta*(1.-0.5)<-1.75:
    #    return False
    else:
        return True

def logp_powerLawPeak(c,sampleDict,injectionDict,priorDict,neff_check_only=False):

    """
    Likelihood for gaussian spin + power-law peak model.

    INPUTS
    c: Array of proposed hyperparameters
    sampleDict: Dictionary of PE samples and prior weights for each event in consideration
    injectionDict: Dictionary of preprocessed injection information, as created by prep_injections.py
    priorDict: Dictionary defining boundaries on hyperparameters
    """

    # Flat priors, reject samples past boundaries
    if not inPrior_powerLawPeak(c,priorDict):
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
        mu0 = c[7]
        log_sigma0 = c[8]
        alpha = c[9]
        beta = c[10]

        # Gaussian prior on kappa
        logP = 0.
        logP += -kappa**2./(2.*priorDict['sig_kappa']**2.)
        #logP += -alpha**2./(2.*priorDict['sig_alpha']**2.)
        #logP += -beta**2./(2.*priorDict['sig_beta']**2.)

        # Unpack injections
        m1_det = injectionDict['m1']
        m2_det = injectionDict['m2']
        s1z_det = injectionDict['s1z']
        s2z_det = injectionDict['s2z']
        z_det = injectionDict['z']
        dVdz_det = injectionDict['dVdz']
        pop_reweight = injectionDict['weights_XeffOnly']
        q_det = m2_det/m1_det
        mtot_det = m1_det+m2_det
        X_det = (m1_det*s1z_det + m2_det*s2z_det)/(m1_det+m2_det)

        mMin = priorDict['mMin']

        # Reweight injection probabilities
        # Gaussian p(xeff) with q-dependent mean and std deviation
        mu_q = mu0 + alpha*(q_det-1.)
        log_sigma_q = log_sigma0 + beta*(q_det-1.)
        p_det_Xeff = calculate_Gaussian(X_det, mu_q, 10.**(2.*log_sigma_q),-1.,1.)

        # p(m2|m1) and p(z)
        # Note that `pop_reweight` already has appropriate factors of dV/dz
        p_det_m2 = (1.+bq)*np.power(m2_det,bq)/(np.power(m1_det,1.+bq)-mMin**(1.+bq))
        p_det_m2[m2_det<mMin] = 0
        p_det_z = dVdz_det*np.power(1.+z_det,kappa-1.)

        # PLPeak distribution on p(m1)
        p_det_m1_pl = (1.+lmbda)*m1_det**lmbda/(mMax**(1.+lmbda) - mMin**(1.+lmbda))
        p_det_m1_pl[m1_det>mMax] = 0
        p_det_m1_peak = np.exp(-0.5*(m1_det-m0)**2./sigM**2)/np.sqrt(2.*np.pi*sigM**2.)
        p_det_m1 = fPeak*p_det_m1_peak + (1.-fPeak)*p_det_m1_pl

        # Construct full weighting factors
        det_weights = p_det_Xeff*p_det_m1*p_det_m2*p_det_z*pop_reweight
        det_weights[np.where(det_weights!=det_weights)] = 0.
        if np.max(det_weights)==0:
            return -np.inf
        if np.any(det_weights<0):
            print("Negative injection weights!")
            sys.exit()

        # Check for sufficient sampling size
        nEvents = len(sampleDict)
        Nsamp = np.sum(det_weights)**2/np.sum(det_weights**2)
        if Nsamp<=4*nEvents:
            print("Insufficient mock detections:",c)
            return -np.inf

        if neff_check_only:
            return 0

        # Add detection efficiency term to log likelihood
        log_detEff = -nEvents*np.log(np.sum(det_weights))
        logP += log_detEff

        # Loop across samples
        for event in sampleDict:

            # Grab samples
            m1_sample = sampleDict[event]['m1']
            m2_sample = sampleDict[event]['m2']
            X_sample = sampleDict[event]['Xeff']
            z_sample = sampleDict[event]['z']
            Xeff_prior = sampleDict[event]['Xeff_priors']
            weights = sampleDict[event]['weights']
            q_sample = m2_sample/m1_sample
            
            # Chi probability - Gaussian
            mu_q = mu0 + alpha*(q_sample-1.)
            log_sigma_q = log_sigma0 + beta*(q_sample-1.)
            p_Chi = calculate_Gaussian(X_sample, mu_q, 10.**(2.*log_sigma_q),-1.,1.)

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

        if logP!=logP:
            print("!!!!!!!",c)

        return logP

