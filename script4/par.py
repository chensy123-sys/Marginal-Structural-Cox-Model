import simdata
import numpy as np
from scipy.special import expit
from MCI import train_nuisance_func, Marginal_Cox_Instrumental, plot_MCI_Hazard, plot_MCI_Surv
import matplotlib.pyplot as plt

n=1000; Looptime=500; tau = 1

par1 = {
    'psi': -0.5,  # hazards ratio of interest
    # 'D_surv': lambda t, A, psi: np.exp(- np.exp(A * psi) * t),
    # # baseline hazards ratio    
    'f': lambda X1, X2: expit(np.sin(X1) + 0.5 * np.sin(X2)),
    # generate Pr(Z=1|X)
    'deltaA': lambda X1, X2, U: np.tanh(1 + 0.4 * np.sin(2*X1)),
    # generate Pr(A=1|Z=1,X,U)-Pr(A=1|Z=0,X,U)
    'OPA': lambda X1, X2, U: np.exp(-2 - 2*U + X1),
    # generate Pr(A=1|Z=1,X,U)/(1-.)*Pr(A=1|Z=0,X,U)/(1-.)
    'C_mean': lambda Z, A, X1, X2, U: np.exp(-(0.5 * Z - 0.5 * A + 0.25 * X1 - 0.25 * X2)+1.25)
    # generate C|Z,A,X,U from Cox PH model
}

par2 = {
    'psi': -0.5,
    'f': lambda X1, X2: expit(0.5*X1 - X2),
    'deltaA': lambda X1, X2, U: np.tanh(0.4 + 0.3 * X1**2 + 0.3 * X2**2),
    'OPA': lambda X1, X2, U: np.exp(-2 - 2*U + X1),
    'C_mean': lambda Z, A, X1, X2, U: np.exp(-(0.5 * Z - 0.5 * A + 0.25 * X1 - 0.25 * X2)+1.25)
}

par3 = {
    'psi': -0.5,
    'f': lambda X1, X2: expit(0.5*X1 - X2),
    'deltaA': lambda X1, X2, U: np.tanh(1 + 0.4 * np.sin(2*X1)),
    'OPA': lambda X1, X2, U: np.exp(-2 - 2*U + X1),
    'C_mean': lambda Z, A, X1, X2, U: np.exp(-(0.5 * Z - 0.5 * A + 0.25 * X1 - 0.25 * X2)+1.25)
}

par4 = {
    'psi': -0.5,
    'f': lambda X1, X2: expit(np.sin(X1) + 0.5 * np.sin(X2)),
    'deltaA': lambda X1, X2, U: np.tanh(0.4 + 0.3 * X1**2 + 0.3 * X2**2),
    'OPA': lambda X1, X2, U: np.exp(-2 - 2*U + X1),
    'C_mean': lambda Z, A, X1, X2, U: np.exp(-(0.5 * Z - 0.5 * A + 0.25 * X1 - 0.25 * X2)+1.25)
}

