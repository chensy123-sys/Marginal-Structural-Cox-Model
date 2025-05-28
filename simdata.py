import numpy as np
import pandas as pd
from scipy.optimize import root_scalar
from scipy.stats import expon, binom, uniform, norm
from scipy.special import expit  # equivalent to R's plogis


def pi(OPA, deltaA):
    numerator1 = OPA * (2 - deltaA) + deltaA
    numerator2 = (OPA * (deltaA - 2) - deltaA)**2 + 4 * OPA * (1 - OPA) * (1 - deltaA)
    denumerator = 2 * (OPA - 1)
    return (numerator1 - np.sqrt(numerator2)) / denumerator


def sim_cox_instrumental(n, par, shape=1, scale=1):
    """
    Simulate survival data with an instrumental variable under a Cox model with treatment effect.

    This function generates a dataset suitable for evaluating causal inference methods using 
    instrumental variables (IVs) in survival analysis. The data-generating process involves:
    - Endogenous treatment assignment
    - IV-based compliance behavior
    - Time-to-event data with censoring
    - Structural hazard model with treatment effect

    Parameters
    ----------
    n : int
        Number of observations to simulate.

    par : dict
        Dictionary of user-supplied functions and parameters:
        - 'f': function (X1, X2) → P(Z=1), IV assignment model
        - 'deltaA': function (X1, X2, U) → treatment effect of IV on A
        - 'OPA': function (X1, X2, U) → baseline treatment probability
        - 'C_mean': function (Z, A, X1, X2, U) → mean of censoring distribution
        - 'tau': maximum follow-up time (right censoring bound)
        - 'psi': log hazard ratio for treatment effect

    shape : float, default=1
        Shape parameter for Weibull distribution (default = 1, i.e., exponential survival).

    scale : float, default=1
        Scale parameter for Weibull distribution.

    Returns
    -------
    dict
        A dictionary with two components:
        - 'simdat': pd.DataFrame with observable variables:
            - 'Z': Instrumental variable
            - 'A': Treatment assignment
            - 'X1', 'X2': Covariates
            - 'time': Observed event or censoring time
            - 'status': Event indicator (1=event, 0=censored)

        - 'simdat_unobs': pd.DataFrame with unobservable variables used internally:
            - 'U': Unmeasured confounder
            - 'deltaA': True treatment effect of IV on A
            - 'Z_probs': Probability of Z=1
            - 'D0', 'D1': Potential failure times under A=0 and A=1
            - 'D': True failure time
            - 'C': Censoring time
            - 'SC': Censoring survival function at observed time

    Notes
    -----
    - The IV Z affects treatment A via compliance functions `deltaA` and `OPA`.
    - Treatment A influences survival times via a proportional hazards model.
    - The failure time is modeled using a Weibull distribution with a shift for treatment.
    - Right censoring is applied via an exponential distribution bounded by `tau`.
    - The returned `simdat` is suitable for use in downstream causal hazard models such as 
      those implemented with `Marginal_Cox_Instrumental`.

    Examples
    --------
    >>> from scipy.stats import norm
    >>> par = {
    ...     'f': lambda x1, x2: norm.cdf(0.5 * x1 - x2),
    ...     'deltaA': lambda x1, x2, u: 0.3 + 0.2 * u,
    ...     'OPA': lambda x1, x2, u: 0.3 + 0.2 * x2,
    ...     'C_mean': lambda z, a, x1, x2, u: 3.0 + 0.5 * a,
    ...     'tau': 10,
    ...     'psi': np.log(2)
    ... }
    >>> data = sim_cox_instrumental(1000, par)
    """

    W1 = uniform.rvs(size=n) * 2 - 1
    W2 = uniform.rvs(size=n) * 2 - 1
    W3 = uniform.rvs(size=n) * 2 - 1
    
    X1 = 0.5 * W1 + W3
    X2 = W1 + 1.5 * W2**2 - 0.5
    U = W1 + W2


    Z_probs = par['f'](X1, X2)
    Z = binom.rvs(1, Z_probs)

    deltaA = par['deltaA'](X1, X2, U)
    OPA = par['OPA'](X1, X2, U)
    pA0 = pi(OPA, deltaA)
    pA1 = pA0 + deltaA

    A0 = binom.rvs(1, pA0)
    A1 = binom.rvs(1, pA1)
    A = Z * A1 + (1 - Z) * A0

    C_mean = par['C_mean'](Z, A, X1, X2, U)
    C = np.minimum(expon.rvs(scale=C_mean), par['tau'])

    D0 = (-np.log(0.5 + 0.5 * W1))**(1/shape) / scale
    D1 = (-np.log(0.5 + 0.5 * W1) * np.exp(-par['psi']))**(1/shape) / scale
    D = A * D1 + (1 - A) * D0

    time = np.minimum(D, C)
    status = (D <= C).astype(int)

    SC = np.exp(-time / C_mean)

    df = pd.DataFrame({
        'Z': Z,
        'A': A,
        'X1': X1,
        'X2': X2,
        'time': time,
        'status': status,
        'deltaA': deltaA,
        'U': U,
        'Z_probs': Z_probs,
        'D0': D0,
        'D1': D1,
        'D': D,
        'C': C,
        'SC': SC
    })

    simdat = df[['Z', 'A', 'X1', 'X2', 'time', 'status']]
    simdat_unobs = df[['U', 'deltaA', 'Z_probs', 'D0', 'D1', 'D', 'C', 'SC']]

    return {'simdat': simdat, 'simdat_unobs': simdat_unobs}












def sim_cox_instrumental2(n, par, shape=1, scale=1):

    W1 = norm.rvs(size=n)*0.5
    W2 = norm.rvs(size=n)*0.5
    W3 = norm.rvs(size=n)*0.5

    D0 = (-np.log(norm.cdf(W1/0.5)))**(1/shape) / scale
    D1 = (-np.log(norm.cdf(W1/0.5)) * np.exp(-par['psi']))**(1/shape) / scale
    
    

    X1 = 0.5 * W1 + W3
    X2 = W1 + 1.5 * W2**2 - 0.5
    U = W1 + W2


    Z_probs = par['f'](X1, X2)
    Z = binom.rvs(1, Z_probs)


    deltaA = par['deltaA'](X1, X2, U)
    OPA = par['OPA'](X1, X2, U)
    pA0 = pi(OPA, deltaA)
    pA1 = pA0 + deltaA

    A0 = binom.rvs(1, pA0)
    A1 = binom.rvs(1, pA1)
    A = Z * A1 + (1 - Z) * A0
    D = A * D1 + (1 - A) * D0


    C_mean = par['C_mean'](Z, A, X1, X2, U)
    C = np.minimum(expon.rvs(scale=C_mean), par['tau'])

    

    time = np.minimum(D, C)
    status = (D <= C).astype(int)

    SC = np.exp(-time / C_mean)

    df = pd.DataFrame({
        'Z': Z,
        'A': A,
        'X1': X1,
        'X2': X2,
        'time': time,
        'status': status,
        'deltaA': deltaA,
        'U': U,
        'Z_probs': Z_probs,
        'D0': D0,
        'D1': D1,
        'D': D,
        'C': C,
        'SC': SC
    })

    simdat = df[['Z', 'A', 'X1', 'X2', 'time', 'status']]
    simdat_unobs = df[['U', 'deltaA', 'Z_probs', 'D0', 'D1', 'D', 'C', 'SC']]

    return {'simdat': simdat, 'simdat_unobs': simdat_unobs}




def sim_cox_instrumental_exponential(n, par,shape=1, scale=1):
    data = []

    def check(par):
        left = sum(k for k in par['kappa'])
        right = min(-np.log(par['D_surv'](t=1, A=0, psi=par['psi'])),
                    -np.log(par['D_surv'](t=1, A=1, psi=par['psi'])))
        print(left)
        print(right)
        return left <= right
    print(check(par))
    def g(t, par, A, X1, X2, U, u,shape=shape,scale=scale):
        t = t**shape * scale
        res = (1 + par['kappa'][0] * t) * \
            (1 + par['kappa'][1] * t ) * \
            (1 + par['kappa'][2] * t )
        
        temp = -par['kappa'][0] * X1 - par['kappa'][1] * X2 - par['kappa'][2] * U
        res = res * np.exp(temp * t) * par['D_surv'](t, A, par['psi'])
        return res - 1 + u
    
    for _ in range(n):
        U = expon.rvs()
        X1 = expon.rvs()
        X2 = expon.rvs()
        
        Z_probs = par['f'](X1, X2)
        Z = binom.rvs(1, Z_probs)
        
        deltaA = par['deltaA'](X1, X2, U)
        OPA = par['OPA'](X1, X2, U)
        pA0 = pi(OPA, deltaA)
        pA1 = pA0 + deltaA
        
        A = Z * binom.rvs(1, pA1) + (1 - Z) * binom.rvs(1, pA0)
        
        C_mean = par['C_mean'](Z, A, X1, X2, U)
        C = expon.rvs(scale=C_mean)
        C = min(C, par['tau'])
        
        add = uniform.rvs()
        
        # Find D0
        try:
            sol0 = root_scalar(lambda t: g(t, par, 0, X1, X2, U, add), 
                              bracket=[1e-8, 1e3], method='brentq')
            D0 = sol0.root
        except:
            D0 = 1e3
            
        # Find D1
        try:
            sol1 = root_scalar(lambda t: g(t, par, 1, X1, X2, U, add), 
                              bracket=[1e-8, 1e3], method='brentq')
            D1 = sol1.root
        except:
            D1 = 1e3
            
        D = A * D1 + (1 - A) * D0
        time = min(D, C)
        status = 1 if D <= C else 0
        
        data.append({
            'Z': Z,
            'A': A,
            'X1': X1,
            'X2': X2,
            'time': time,
            'status': status,
            'U': U,
            'deltaA': deltaA,
            'Z_probs': Z_probs,
            'D0': D0,
            'D1': D1,
            'D': D,
            'C': C
        })
    
    df = pd.DataFrame(data)
    simdat = df[['Z', 'A', 'X1', 'X2', 'time', 'status']]
    simdat_unobs = df[['U', 'deltaA', 'Z_probs', 'D0', 'D1', 'D', 'C']]
    
    return {'simdat': simdat, 'simdat_unobs': simdat_unobs}











def sim_cox_instrumental_gaussian(n, par, shape=1, scale=1):
    data = []
    def check(par):
        left = sum(k for k in par['kappa'])
        right = min(-np.log(par['D_surv'](t=1, A=0, psi=par['psi'])),
                    -np.log(par['D_surv'](t=1, A=1, psi=par['psi'])))
        print(left)
        print(right)
        return left <= right
    
    print(check(par))
    def g(t, par, A, X1, X2, U, u, shape=shape,scale=scale):
        t = t**shape * scale
        temp = -par['kappa'][0] * X1**2 - par['kappa'][1] * X2**2 - par['kappa'][2] * U**2
        res = np.log(1+2*par['kappa'][0] * t) + np.log(1+2*par['kappa'][1] * t) + np.log(1+2*par['kappa'][2] * t)
        res = np.exp(temp * t + res * 0.5 ) * par['D_surv'](t, A, par['psi'])
        return res - 1 + u
    
    for _ in range(n):
        U = norm.rvs()
        X1 = norm.rvs()
        X2 = norm.rvs()
        
        Z_probs = par['f'](X1, X2)
        Z = binom.rvs(1, Z_probs)
        
        deltaA = par['deltaA'](X1, X2, U)
        OPA = par['OPA'](X1, X2, U)
        pA0 = pi(OPA, deltaA)
        pA1 = pA0 + deltaA
        
        A = Z * binom.rvs(1, pA1) + (1 - Z) * binom.rvs(1, pA0)
        
        C_mean = par['C_mean'](Z, A, X1, X2, U)
        # print(C_mean)
        C = expon.rvs(scale=C_mean)
        C = min(C, par['tau'])
        # print(C)
        add = uniform.rvs()
        
        # Find D0
        try:
            sol0 = root_scalar(lambda t: g(t, par, 0, X1, X2, U, add), 
                              bracket=[1e-8, 1e3], method='brentq')
            D0 = sol0.root
        except:
            D0 = 1e3
            
        # Find D1
        try:
            sol1 = root_scalar(lambda t: g(t, par, 1, X1, X2, U, add), 
                              bracket=[1e-8, 1e3], method='brentq')
            D1 = sol1.root
        except:
            D1 = 1e3
            
        D = A * D1 + (1 - A) * D0
        time = min(D, C)
        status = 1 if D <= C else 0
        
        data.append({
            'Z': Z,
            'A': A,
            'X1': X1,
            'X2': X2,
            'time': time,
            'status': status,
            'U': U,
            'deltaA': deltaA,
            'Z_probs': Z_probs,
            'D0': D0,
            'D1': D1,
            'D': D,
            'C': C,
            'SD': g(time, par, 1, X1, X2, U, 1),
            'SC': np.exp(- time / C_mean)
        })
    
    df = pd.DataFrame(data)
    simdat = df[['Z', 'A', 'X1', 'X2', 'time', 'status']]
    simdat_unobs = df[['U', 'deltaA', 'Z_probs', 'D0', 'D1', 'D', 'C', 'SD','SC']]
    
    return {'simdat': simdat, 'simdat_unobs': simdat_unobs}










def sim_cox_instrumental_uniform(n, par,shape=1,scale=1):
    data = []
    def check(par):
        left = sum(np.abs(k) for k in par['kappa'])
        right = min(-np.log(par['D_surv'](t=1, A=0, psi=par['psi'])),
                    -np.log(par['D_surv'](t=1, A=1, psi=par['psi'])))
        print(left)
        print(right)
        return left <= right
    
    print(check(par))
    def g(t, par, A, X1, X2, U, u, shape=shape,scale=scale):
        t = t**shape * scale
        temp = -par['kappa'][0] * X1 - par['kappa'][1] * X2 - par['kappa'][2] * U
        # res = np.log(1+2*par['kappa'][0] * t) + np.log(1+2*par['kappa'][1] * t) + np.log(1+2*par['kappa'][2] * t)
        # res0 = 2*par['kappa'][0]*t/(np.exp(par['kappa'][0]*t) - np.exp(-par['kappa'][0]*t))
        # res1 = 2*par['kappa'][1]*t/(np.exp(par['kappa'][1]*t) - np.exp(-par['kappa'][1]*t))
        # res2 = 2*par['kappa'][2]*t/(np.exp(par['kappa'][2]*t) - np.exp(-par['kappa'][2]*t))

        res0 = par['kappa'][0] * t / np.sinh(par['kappa'][0] * t)
        res1 = par['kappa'][1] * t / np.sinh(par['kappa'][1] * t)
        res2 = par['kappa'][2] * t / np.sinh(par['kappa'][2] * t)
        # print(res0, res1, res2)
        res = res0 * res1 * res2 * np.exp(temp * t) * par['D_surv'](t, A, par['psi'])
        return res - 1 + u
    
    for _ in range(n):
        U = uniform.rvs()*2-1
        X1 = uniform.rvs()*2-1
        X2 = uniform.rvs()*2-1
        
        Z_probs = par['f'](X1, X2)
        Z = binom.rvs(1, Z_probs)
        
        deltaA = par['deltaA'](X1, X2, U)
        OPA = par['OPA'](X1, X2, U)
        pA0 = pi(OPA, deltaA)
        pA1 = pA0 + deltaA
        
        A = Z * binom.rvs(1, pA1) + (1 - Z) * binom.rvs(1, pA0)
        
        C_mean = par['C_mean'](Z, A, X1, X2, U)
        # print(C_mean)
        C = expon.rvs(scale=C_mean)
        C = min(C, par['tau'])
        # print(C)
        add = uniform.rvs()
        
        # Find D0
        try:
            sol0 = root_scalar(lambda t: g(t, par, 0, X1, X2, U, add), 
                              bracket=[1e-8, 1e3], method='brentq')
            D0 = sol0.root
        except:
            D0 = 1e3
            
        # Find D1
        try:
            sol1 = root_scalar(lambda t: g(t, par, 1, X1, X2, U, add), 
                              bracket=[1e-8, 1e3], method='brentq')
            D1 = sol1.root
        except:
            D1 = 1e3
            
        D = A * D1 + (1 - A) * D0
        time = min(D, C)
        status = 1 if D <= C else 0
        
        data.append({
            'Z': Z,
            'A': A,
            'X1': X1,
            'X2': X2,
            'time': time,
            'status': status,
            'deltaA': deltaA,
            'U': U,
            'Z_probs': Z_probs,
            'D0': D0,
            'D1': D1,
            'D': D,
            'C': C,
            'SD': g(time, par, 1, X1, X2, U, 1),
            'SC': np.exp(- time / C_mean)
        })
    
    df = pd.DataFrame(data)
    simdat = df[['Z', 'A', 'X1', 'X2', 'time', 'status']]
    simdat_unobs = df[['U', 'deltaA', 'Z_probs', 'D0', 'D1', 'D', 'C', 'SD','SC']]
    
    return {'simdat': simdat, 'simdat_unobs': simdat_unobs}



