import numpy as np
from sklearn.model_selection import KFold
from scipy.interpolate import interp1d
from Train_nuisance import S_coxph, S_rfs, ps_spline, ps_nn, ps_kernel
import pandas as pd

def train_nuisance_func(simdat, Kfold=5, accuracy=500, min_val=10e-2,
                        method_SD=['rsf',100,20], method_SC=['rsf',100,20], method_ps=['gam',0.01,[50]]):
    """
    Train nuisance functions for semiparametric causal estimation using K-fold cross-fitting.

    This function estimates counterfactual survival and censoring functions, as well as propensity 
    scores, for use in doubly robust survival analysis settings. Outputs include quantities for 
    augmented, outcome regression (OR), and inverse probability weighting (IPW) estimators.

    Parameters
    ----------
    simdat : pd.DataFrame
        A dataset containing the observed data. Must include the columns:
        - 'A': Treatment assignment
        - 'Z': Instrumental variable
        - 'status': Event indicator (1 if event, 0 if censored)
        - 'time': Observed event or censoring time

    Kfold : int, default=5
        Number of folds for cross-fitting.

    accuracy : int, default=500
        Number of discrete time points used for evaluating survival functions.

    min_val : float, default=1e-2
        Minimum threshold used to regularize small denominators for numerical stability.

    method_SD : list, default=['rsf', 100, 20]
        Method for estimating the survival function S(t | A, Z).
        Format: [method_name, ntree, nodesize]
        - method_name: 'rsf' (Random Survival Forest) or 'cox' (Cox proportional hazards)
        - ntree: Number of trees for 'rsf'
        - nodesize: Minimum terminal node size for 'rsf'

    method_SC : list, default=['rsf', 100, 20]
        Method for estimating the censoring survival function C(t | A, Z), in the same format as method_SD.

    method_ps : list, default=['gam', 0.01, [50]]
        Method for estimating the propensity score and associated quantities.
        Format: [method_name, learning_rate, architecture]
        - method_name: One of 'gam', 'nn', or 'kernel'
        - For 'nn': 'architecture' is a list of hidden layer sizes; 'learning_rate' is the optimizer step size.

    Returns
    -------
    dict
        A dictionary with the following keys:
        - 'results': List of dictionaries for each fold, each containing:
            - 'time.interest': Time grid used for survival estimation
            - 'R0', 'R1': Augmented counterfactual survival estimates for A=0 and A=1
            - 'ipw0', 'ipw1': IPW-only survival estimates
            - 'gamma0', 'gamma1': Outcome regression-only components
            - 'fit_SD', 'fit_SC', 'fit_ps': Fitted nuisance models
        - 'folds': List of train-test index pairs used in K-fold CV
        - 'accuracy': Number of discrete time points for evaluation

    Notes
    -----
    This function implements a robust, doubly robust nuisance estimation strategy for use in
    instrumental variable survival analysis. It is typically followed by an estimation function
    such as `Marginal_Cox_Instrumental`, which uses the outputs to estimate causal hazard ratios.
    """

    n = simdat.shape[0]
    kf = KFold(n_splits=Kfold, shuffle=True, random_state=42)
    folds = list(kf.split(simdat))
    
    results = []

    for train_index, test_index in folds:
        if method_SD[0] == 'rsf':
            fit_SD = S_rfs(simdat, train_index, test_index, min_val=min_val, accuracy=accuracy,
                            ntree=method_SD[1], nodesize=method_SD[2], nsplit=3)
        elif method_SD[0] == 'cox':
            fit_SD = S_coxph(simdat, train_index, test_index, min_val=min_val, accuracy=accuracy)


        if method_SC[0] == 'rsf':
            fit_SC = S_rfs(simdat, train_index, test_index, reverse=True, min_val=min_val, accuracy=accuracy, 
                           ntree=method_SC[1], nodesize=method_SC[2], nsplit=3)
        elif method_SC[0] == 'cox':
            fit_SC = S_coxph(simdat, train_index, test_index, reverse=True, min_val=min_val,
                             accuracy=accuracy)

        if method_ps[0] == 'gam':
            fit_ps = ps_spline(simdat, train_index, test_index, min_val=min_val)
        elif method_ps[0] == 'nn':
            fit_ps = ps_nn(simdat, train_index, test_index, min_val=min_val,
                           hidden_layers=method_ps[2],patience=20, lr=method_ps[1])
        elif method_ps[0] == 'kernel':
            fit_ps = ps_kernel(simdat, train_index, test_index, min_val=min_val)
        
        
        pi0, pi1, f, delta, omega = fit_ps['pi0'], fit_ps['pi1'], fit_ps['f'], fit_ps['delta'], fit_ps['omega']
        S11, S10, S01, S00 = fit_SD['S11'], fit_SD['S10'], fit_SD['S01'], fit_SD['S00']
        C11, C10, C01, C00 = fit_SC['S11'], fit_SC['S10'], fit_SC['S01'], fit_SC['S00']
        time_interest = fit_SD['time.interest']

        A = simdat['A'].values[test_index]
        Z = simdat['Z'].values[test_index]
        status = simdat['status'].values[test_index]
        time = simdat['time'].values[test_index]

        SD = (Z*A)[:, None]*S11 + (Z*(1-A))[:, None]*S10 + ((1-Z)*A)[:, None]*S01 + ((1-Z)*(1-A))[:, None]*S00
        SC = (Z*A)[:, None]*C11 + (Z*(1-A))[:, None]*C10 + ((1-Z)*A)[:, None]*C01 + ((1-Z)*(1-A))[:, None]*C00

        SDT = np.array([
            np.interp(time[i], time_interest, SD[i], left=SD[i][0], right=SD[i][-1])
            for i in range(len(test_index))
        ])
        SCT = np.array([
            np.interp(time[i], time_interest, SC[i], left=SC[i][0], right=SC[i][-1])
            for i in range(len(test_index))
        ])


        dFC = -np.hstack([np.diff(SC), np.zeros((SC.shape[0], 1))])

        atRisk = (time[:, None] >= time_interest).astype(float)

        JO = ((1 - status)[:, None] * (1 - atRisk) / (SDT[:, None] * SCT[:, None])
                - np.cumsum(atRisk * dFC / (SD * SC**2 + 1e-10), axis=1))

        temp = Z / f - (1 - Z) / (1 - f + 1e-10)

        gamma1 = (pi1[:,None] * S11 - pi0[:,None] * S01) / delta[:,None]
        gamma0 = ((1 - pi0[:,None]) * S00 - (1 - pi1[:,None]) * S10) / delta[:,None]


        xi1 = omega[:,None] * S01
        xi0 = (1 - omega[:,None]) * S00

        # R0
        R0 = -temp[:, None] * (1 - A)[:, None] / delta[:, None] * atRisk / SC
        ipw0 = R0.copy()
        R0 += (1 - temp[:, None] * (A - omega)[:, None] / delta[:, None]) * gamma0
        R0 += temp[:, None] / delta[:, None] * xi0
        R0 -= temp[:, None] * (1 - A)[:, None] / delta[:, None] * JO * SD
        # R0_avg = R0.mean(axis=0)
        # gamma0_avg = gamma0.mean(axis=0)

        # R1
        R1 = temp[:, None] * A[:, None] / delta[:, None] * atRisk / SC
        ipw1 = R1.copy()
        R1 += (1 - temp[:, None] * (A - omega)[:, None] / delta[:, None]) * gamma1
        R1 -= temp[:, None] / delta[:, None] * xi1
        R1 += temp[:, None] * A[:, None] / delta[:, None] * JO * SD
        # R1_avg = R1.mean(axis=0)
        # gamma1_avg = gamma1.mean(axis=0)

        results.append({
            'time.interest': time_interest,
            'R0': R0,
            'R1': R1,
            'ipw1': ipw1,
            'ipw0': ipw0,
            'gamma0': gamma0,
            'gamma1': gamma1,
            'fit_SD': fit_SD,
            'fit_SC': fit_SC,
            'fit_ps': fit_ps
        })

    return {
        'results': results,
        'folds': folds,
        'accuracy': accuracy,
    }




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def replace_na_nearest(arr):
    arr = np.array(arr)
    isnan = np.isnan(arr)
    if not np.any(isnan):
        return arr
    not_nan_idx = np.where(~isnan)[0]
    for i in np.where(isnan)[0]:
        nearest = not_nan_idx[np.argmin(np.abs(not_nan_idx - i))]
        arr[i] = arr[nearest]
    return arr


def Marginal_Cox_Instrumental(simdat, model):
    """
    Estimate marginal causal hazard ratios using augmented, outcome regression (OR), 
    and inverse probability weighting (IPW) approaches in a Cox instrumental variable setting.

    This function aggregates fold-specific survival estimates and influence functions from 
    cross-fitting results, and computes point estimates and standard errors for the 
    marginal causal hazard ratio. It also returns survival curve estimates under treatment 
    and control across the different methods.

    Parameters
    ----------
    simdat : ndarray
        An (n x p) matrix or DataFrame containing the full dataset used for estimation.
    
    model : dict
        A dictionary containing model output from a cross-fitting procedure. Must include:
            - 'results': List of dictionaries with fold-specific survival estimates 
                         and components (R0, R1, gamma0, gamma1, ipw0, ipw1, time.interest).
            - 'folds': List of (train, test) index tuples or lists used in cross-fitting.
            - 'accuracy': Integer specifying the number of time points in the survival estimates.

    Returns
    -------
    dict
        A dictionary with the following components:
        
        - 'curve_surv0': Estimated survival curves under control:
            - 'gamma0_avg', 'R0_avg', 'ipw0_avg': Mean survival curves from OR, AUG, and IPW.
            - 'std_surv_*': Standard errors of each method.
            - 'target_time': Time points corresponding to the survival estimates.
        
        - 'curve_surv1': Estimated survival curves under treatment:
            - 'gamma1_avg', 'R1_avg', 'ipw1_avg': Mean survival curves from OR, AUG, and IPW.
            - 'std_surv_*': Standard errors of each method. (only consistent for std_surv_aug)
            - 'target_time': Time points corresponding to the survival estimates.
        
        - 'curve_hazard': Hazard ratio curves:
            - 'par_aug', 'par_or', 'par_ipw': Log hazard ratio estimates over time.
            - 'std_*': Standard errors based on the influence function. (only consistent for std_aug)
            - 'index_*': Selected time index where the estimate is closest to the median within [90%, 95%] interval.
            - 'target_time': Time points corresponding to the hazard estimates.
        
        - 'est': Point estimates and standard errors at the final time point:
            - 'par_*_est': Log hazard ratio estimate (final time). 
            - 'std_*_est': Standard error (influence-based) at final time. (only consistent for std_aug_est)
            - 'std_*_est_deg': Decomposed standard error (unadjusted variance-based).

    Notes
    -----
    This implementation assumes each fold in `results` contains correctly computed survival
    estimates and instrumental variable-adjusted quantities for R0/R1, gamma0/gamma1, and ipw0/ipw1.
    The function also uses a helper `replace_na_nearest` to impute NaNs in log hazard ratio curves.
    """

    n = simdat.shape[0]
    results = model['results']
    folds = model['folds']
    accuracy = model['accuracy']
    Kfold = len(folds)

    # AUG
    R0 = np.zeros((n, accuracy))
    R1 = np.zeros((n, accuracy))
    G_aug = np.zeros((n, accuracy))
    W_aug = np.zeros((n, accuracy))
    # Lambda_aug = np.zeros((1, accuracy))
    for i in range(Kfold):
        test = folds[i][1]
        R0_temp = results[i]['R0']
        R1_temp = results[i]['R1']
        R0[test, :] = R0_temp
        R1[test, :] = R1_temp
        G_aug[test, :] = np.apply_along_axis(
            lambda row: np.cumsum((np.mean(R1_temp, axis=0) / np.mean(R0_temp, axis=0)) * np.append(np.diff(-row), 0)),
            axis=1,
            arr=R0_temp
        )
        Lambda_aug = np.apply_along_axis(
            lambda row: np.cumsum(1 / np.mean(R0_temp, axis=0) * np.append(np.diff(-row), 0)),
            axis=1,
            arr=R0_temp
        ).mean(axis=0)
        W_aug[test, :] = np.cumsum((R1_temp - R0_temp * (np.mean(R1_temp, axis=0) / np.mean(R0_temp, axis=0))) \
                                   * np.append(np.diff(Lambda_aug), 0), axis=1)
    

    R0_avg = R0.mean(axis=0)
    R1_avg = R1.mean(axis=0)
    par_aug = np.log((R1_avg[0] - R1_avg) / G_aug.mean(axis=0))
    par_aug = replace_na_nearest(par_aug)
    G_aug *= np.exp(par_aug)
    U_aug = (np.repeat(R1[:, [0]], accuracy, axis=1) - R1) - G_aug
    W_aug = U_aug - W_aug * np.exp(par_aug)

    # OR
    gamma0 = np.zeros((n, accuracy))
    gamma1 = np.zeros((n, accuracy))
    G_or = np.zeros((n, accuracy))
    W_or = np.zeros((n, accuracy))
    # Lambda_or = np.zeros((1, accuracy))

    for i in range(Kfold):
        test = folds[i][1]
        gamma0_temp = results[i]['gamma0']
        gamma1_temp = results[i]['gamma1']
        gamma0[test, :] = gamma0_temp
        gamma1[test, :] = gamma1_temp
        G_or[test, :] = np.apply_along_axis(
            lambda row: np.cumsum((np.mean(gamma1_temp, axis=0) / np.mean(gamma0_temp, axis=0)) * np.append(np.diff(-row), 0)),
            axis=1,
            arr=gamma0_temp
        )
        Lambda_or = np.apply_along_axis(
            lambda row: np.cumsum(1 / np.mean(gamma0_temp, axis=0) * np.append(np.diff(-row), 0)),
            axis=1,
            arr=gamma0_temp
        ).mean(axis=0)
        W_or[test, :] = np.cumsum((gamma1_temp - gamma0_temp * (np.mean(gamma1_temp, axis=0) / np.mean(gamma0_temp, axis=0))) \
                                  * np.append(np.diff(Lambda_or), 0), axis=1)

    gamma0_avg = gamma0.mean(axis=0)
    gamma1_avg = gamma1.mean(axis=0)
    par_or = np.log((gamma1_avg[0] - gamma1_avg) / G_or.mean(axis=0))
    par_or = replace_na_nearest(par_or)
    G_or *= np.exp(par_or)
    U_or = (np.repeat(gamma1[:, [0]], accuracy, axis=1) - gamma1) - G_or
    W_or = U_or - W_or * np.exp(par_or)



    # IPW
    ipw0 = np.zeros((n, accuracy))
    ipw1 = np.zeros((n, accuracy))
    G_ipw = np.zeros((n, accuracy))
    W_ipw = np.zeros((n, accuracy))
    # Lambda_ipw = np.zeros((1, accuracy))

    for i in range(Kfold):
        test = folds[i][1]
        ipw0_temp = results[i]['ipw0']
        ipw1_temp = results[i]['ipw1']
        ipw0[test, :] = ipw0_temp
        ipw1[test, :] = ipw1_temp
        G_ipw[test, :] = np.apply_along_axis(
            lambda row: np.cumsum((np.mean(ipw1_temp, axis=0) / np.mean(ipw0_temp, axis=0)) * np.append(np.diff(-row), 0)),
            axis=1,
            arr=ipw0_temp
        )
        Lambda_ipw = np.apply_along_axis(
            lambda row: np.cumsum(1 / np.mean(ipw0_temp, axis=0) * np.append(np.diff(-row), 0)),
            axis=1,
            arr=ipw0_temp
        ).mean(axis=0)
        W_ipw[test, :] = np.cumsum((ipw1_temp - ipw0_temp * (np.mean(ipw1_temp, axis=0) / np.mean(ipw0_temp, axis=0))) \
                                   * np.append(np.diff(Lambda_ipw), 0), axis=1)

    ipw0_avg = ipw0.mean(axis=0)
    ipw1_avg = ipw1.mean(axis=0)
    par_ipw = np.log((ipw1_avg[0] - ipw1_avg) / G_ipw.mean(axis=0))
    par_ipw = replace_na_nearest(par_ipw)
    G_ipw *= np.exp(par_ipw)
    U_ipw = (np.repeat(ipw1[:, [0]], accuracy, axis=1) - ipw1) - G_ipw
    W_ipw = U_ipw - W_ipw * np.exp(par_ipw)
    
    # print(np.mean(U_or,axis=0))
    # print(np.mean(U_ipw,axis=0))
    # print(np.mean(U_aug,axis=0))
    
    
    start = int(accuracy * 0.9)
    end = int(accuracy * 0.95)

    std_aug = np.sqrt(U_aug.var(axis=0) / G_aug.mean(axis=0)**2 / n)
    std_or = np.sqrt(U_or.var(axis=0) / G_or.mean(axis=0)**2 / n)
    std_ipw = np.sqrt(U_ipw.var(axis=0) / G_ipw.mean(axis=0)**2 / n)

    std_aug2 = np.sqrt(W_aug.var(axis=0) / G_aug.mean(axis=0)**2 / n)
    std_or2 = np.sqrt(W_or.var(axis=0) / G_or.mean(axis=0)**2 / n)
    std_ipw2 = np.sqrt(W_ipw.var(axis=0) / G_ipw.mean(axis=0)**2 / n)

    # plt.subplot(211)
    # plt.plot(Lambda_aug)
    # plt.subplot(212)
    # plt.plot(U_aug.var(axis=0))
    # plt.plot(W_aug.var(axis=0))
    # plt.show()

    # plt.plot(std_aug)
    # plt.plot(std_aug2)
    # plt.show()

    std_surv_aug0 = np.std(R0, axis=0, ddof=1)/np.sqrt(n)
    std_surv_aug1 = np.std(R1, axis=0, ddof=1)/np.sqrt(n)
    std_surv_ipw0 = np.std(ipw0, axis=0, ddof=1)/np.sqrt(n)
    std_surv_ipw1 = np.std(ipw1, axis=0, ddof=1)/np.sqrt(n)
    std_surv_or0 = np.std(gamma0, axis=0, ddof=1)/np.sqrt(n)
    std_surv_or1 = np.std(gamma1, axis=0, ddof=1)/np.sqrt(n)

    par_aug_est = np.median(par_aug[start:end])
    par_or_est = np.median(par_or[start:end])
    par_ipw_est = np.median(par_ipw[start:end])

    index_aug = np.argmin(np.abs(par_aug[start:end] - par_aug_est))
    index_or = np.argmin(np.abs(par_or[start:end] - par_or_est))
    index_ipw = np.argmin(np.abs(par_ipw[start:end] - par_ipw_est))

    return {
        "curve_surv0": {
            "gamma0_avg": gamma0_avg,
            "R0_avg": R0_avg,
            "ipw0_avg": ipw0_avg,
            "std_surv_aug0": std_surv_aug0,
            "std_surv_ipw0": std_surv_ipw0,
            "std_surv_or0": std_surv_or0,
            "target_time": results[0]['time.interest']
        },
        "curve_surv1": {
            "gamma1_avg": gamma1_avg,
            "R1_avg": R1_avg,
            "ipw1_avg": ipw1_avg,
            "std_surv_aug1": std_surv_aug1,
            "std_surv_ipw1": std_surv_ipw1,
            "std_surv_or1": std_surv_or1,
            "target_time": results[0]['time.interest']
        },
        "curve_hazard": {
            "par_aug": par_aug,
            "par_or": par_or,
            "par_ipw": par_ipw,
            "std_aug": std_aug2,
            "std_or": std_or2,
            "std_ipw": std_ipw2,
            "index_aug": index_aug,
            "index_or": index_or,
            "index_ipw": index_ipw,
            "target_time": results[0]['time.interest']
        },
        "est": {
            "std_aug_est": std_aug2[-1],
            "std_or_est": std_or2[-1],
            "std_ipw_est": std_ipw2[-1],
            "std_aug_est_deg": std_aug[-1],
            "std_or_est_deg": std_or[-1],
            "std_ipw_est_deg": std_ipw[-1],
            "par_aug_est": par_aug[-1],
            "par_or_est": par_or[-1],
            "par_ipw_est": par_ipw[-1],
        }
    }

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from lifelines import KaplanMeierFitter
from scipy.interpolate import interp1d





def plot_MCI_Surv(fit_MCI, par, shape=1):
    color_dict = {
            'AIPW': 'red',
            'OR': 'blue',
            'IPW': 'green'
        }

    curve_surv0 = fit_MCI['curve_surv0']
    curve_surv1 = fit_MCI['curve_surv1']
    target_time = curve_surv0['target_time']

    df0_aug = pd.DataFrame({
        'time': target_time,
        'mean': curve_surv0['R0_avg'],
        'lower': curve_surv0['R0_avg'] - 1.96 * curve_surv0['std_surv_aug0'],
        'upper': curve_surv0['R0_avg'] + 1.96 * curve_surv0['std_surv_aug0'],
        'group': 'AIPW'
    })
    df0_ipw = pd.DataFrame({
        'time': target_time,
        'mean': curve_surv0['ipw0_avg'],
        'lower': curve_surv0['ipw0_avg'] - 1.96 * curve_surv0['std_surv_ipw0'],
        'upper': curve_surv0['ipw0_avg'] + 1.96 * curve_surv0['std_surv_ipw0'],
        'group': 'IPW'
    })
    df0_or = pd.DataFrame({
        'time': target_time,
        'mean': curve_surv0['gamma0_avg'],
        'lower': curve_surv0['gamma0_avg'] - 1.96 * curve_surv0['std_surv_or0'],
        'upper': curve_surv0['gamma0_avg'] + 1.96 * curve_surv0['std_surv_or0'],
        'group': 'OR'
    })
    df0_true = pd.DataFrame({
        'time': target_time,
        'mean': par['D_surv'](target_time**shape,0,par['psi']),
        'lower': par['D_surv'](target_time**shape,0,par['psi']),
        'upper': par['D_surv'](target_time**shape,0,par['psi']),
        'group': 'TRUE'
    })
    df0 = pd.concat([df0_aug, df0_or, df0_ipw, df0_true], ignore_index=True)



    # 构造 DataFrame
    df1_aug = pd.DataFrame({
        'time': target_time,
        'mean': curve_surv1['R1_avg'],
        'lower': curve_surv1['R1_avg'] - 1.96 * curve_surv1['std_surv_aug1'],
        'upper': curve_surv1['R1_avg'] + 1.96 * curve_surv1['std_surv_aug1'],
        'group': 'AIPW'
    })
    df1_ipw = pd.DataFrame({
        'time': target_time,
        'mean': curve_surv1['ipw1_avg'],
        'lower': curve_surv1['ipw1_avg'] - 1.96 * curve_surv1['std_surv_ipw1'],
        'upper': curve_surv1['ipw1_avg'] + 1.96 * curve_surv1['std_surv_ipw1'],
        'group': 'IPW'
    })
    df1_or = pd.DataFrame({
        'time': target_time,
        'mean': curve_surv1['gamma1_avg'],
        'lower': curve_surv1['gamma1_avg'] - 1.96 * curve_surv1['std_surv_or1'],
        'upper': curve_surv1['gamma1_avg'] + 1.96 * curve_surv1['std_surv_or1'],
        'group': 'OR'
    })
    df1_true = pd.DataFrame({
        'time': target_time,
        'mean': par['D_surv'](target_time**shape,1,par['psi']),
        'lower': par['D_surv'](target_time**shape,1,par['psi']),
        'upper': par['D_surv'](target_time**shape,1,par['psi']),
        'group': 'TRUE'
    })
    df1 = pd.concat([df1_aug, df1_or, df1_ipw, df1_true], ignore_index=True)


    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    sns.set(style="whitegrid")

    for key, grp in df1[df1['group'].isin(['AIPW', 'OR', 'IPW'])].groupby('group'):
        color = color_dict.get(key, 'gray')
        plt.fill_between(grp['time'], grp['lower'], grp['upper'], alpha=0.3, label=f"{key} CI",color=color)
        plt.plot(grp['time'], grp['mean'], label=f"{key} Mean", linewidth=2,color=color)
    

    df_true = df1[df1['group'] == 'TRUE']
    plt.plot(df_true['time'], df_true['mean'], color='black', linewidth=2, label='TRUE')
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.title("Estimated Survival Curves with Confidence Intervals")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.ylim([0,1.2])


    plt.subplot(1, 2, 2)
    sns.set(style="whitegrid")

    for key, grp in df0[df0['group'].isin(['AIPW', 'OR', 'IPW'])].groupby('group'):
        color = color_dict.get(key, 'gray')
        plt.fill_between(grp['time'], grp['lower'], grp['upper'], alpha=0.3, label=f"{key} CI",color=color)
        plt.plot(grp['time'], grp['mean'], label=f"{key} Mean", linewidth=2,color=color)
    
    df_true = df0[df0['group'] == 'TRUE']
    plt.plot(df_true['time'], df_true['mean'], color='black', linewidth=2, label='TRUE')
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.title("Estimated Survival Curves with Confidence Intervals")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.ylim([0,1.2])
    plt.show()


def plot_MCI_Hazard(fit_MCI, par):
    curve = fit_MCI['curve_hazard']
    target_time = curve['target_time']


    color_dict = {
        'AIPW': 'red',
        'OR': 'blue',
        'IPW': 'green'
    }
    
    df_aug = pd.DataFrame({
        'time': target_time,
        'mean': curve['par_aug'],
        'lower': curve['par_aug'] - 1.96 * curve['std_aug'],
        'upper': curve['par_aug'] + 1.96 * curve['std_aug'],
        'group': 'AIPW'
    })

    df_or = pd.DataFrame({
        'time': target_time,
        'mean': curve['par_or'],
        'lower': curve['par_or'] - 1.96 * curve['std_or'],
        'upper': curve['par_or'] + 1.96 * curve['std_or'],
        'group': 'OR'
    })

    df_ipw = pd.DataFrame({
        'time': target_time,
        'mean': curve['par_ipw'],
        'lower': curve['par_ipw'] - 1.96 * curve['std_ipw'],
        'upper': curve['par_ipw'] + 1.96 * curve['std_ipw'],
        'group': 'IPW'
    })

    df = pd.concat([df_aug, df_or, df_ipw], ignore_index=True)

    start = np.quantile(target_time, 0.1)
    df = df[df['time'] > start]

    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")

    for key, grp in df.groupby('group'):
        color = color_dict.get(key, 'gray')  
        plt.fill_between(grp['time'], grp['lower'], grp['upper'], alpha=0.3, label=f"{key} CI", color=color)
        plt.plot(grp['time'], grp['mean'], label=f"{key} Mean", linewidth=2, color=color)


    plt.axhline(y=par['psi'], linestyle='--', color='black', label='psi')

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Hazard Curve")
    plt.xlim(start, par['tau'])
    plt.ylim(-2, 2)
    plt.legend()
    plt.tight_layout()
    plt.show()
