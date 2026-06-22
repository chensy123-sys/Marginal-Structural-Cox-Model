import numpy as np; import pandas as pd
import seaborn as sns; import copy
from sklearn.model_selection import KFold
from scipy.interpolate import interp1d
from lifelines import KaplanMeierFitter
from Train_nuisance import S_coxph, S_rfs, ps_spline #, ps_nn, ps_kernel
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

def process_fold(
    simdat,
    train_index,
    test_index,
    tau, accuracy,
    min_val,
    method_SD,
    method_SC,
    method_pi,
    method_f,
):  

    # -----------------------------
    # nuisance estimation
    # -----------------------------
    if method_SD[0] == 'rsf':
        fit_SD = S_rfs(
            simdat,
            train_index,
            test_index,
            covariate=method_SD[1],
            min_val=min_val,
            tau=tau,accuracy=accuracy,
            ntree=method_SD[2],
            nodesize=method_SD[3],
            nsplit=3
        )
    else:
        fit_SD = S_coxph(
            simdat,
            train_index,
            test_index,
            covariate=method_SD[1],
            min_val=min_val,
            tau=tau,accuracy=accuracy
        )

    if method_SC[0] == 'rsf':
        fit_SC = S_rfs(
            simdat,
            train_index,
            test_index,
            covariate=method_SC[1],
            reverse=True,
            min_val=min_val,
            tau=tau,accuracy=accuracy,
            ntree=method_SC[2],
            nodesize=method_SC[3],
            nsplit=3
        )
    else:
        fit_SC = S_coxph(
            simdat,
            train_index,
            test_index,
            covariate=method_SC[1],
            reverse=True,
            min_val=min_val,
            tau=tau,accuracy=accuracy
        )

    if method_f[0] == 'gam':
        fit_f = ps_spline(
            simdat,
            train_index,
            test_index,
            min_val=min_val,
            nuisance = 'instrument',covariate=method_f[1]
        )
    
    if method_pi[0] == 'gam':
        fit_pi = ps_spline(
            simdat,
            train_index,
            test_index,
            min_val=min_val,
            nuisance = 'treatment',covariate=method_pi[1]
        )


    # =====================================
    # 下面直接复制你原来的计算部分
    # =====================================

    pi0, pi1, f, delta, omega = (
        fit_pi['pi0'],
        fit_pi['pi1'],
        fit_f['f'],
        fit_pi['delta'],
        fit_pi['omega']
    )

    S11, S10, S01, S00 = (
        fit_SD['S11'],
        fit_SD['S10'],
        fit_SD['S01'],
        fit_SD['S00']
    )

    C11, C10, C01, C00 = (
        fit_SC['S11'],
        fit_SC['S10'],
        fit_SC['S01'],
        fit_SC['S00']
    )

    time_interest = fit_SD['time.interest']

    A = simdat['A'].values[test_index]
    Z = simdat['Z'].values[test_index]
    status = simdat['status'].values[test_index]
    time = simdat['time'].values[test_index]

    SD = (
        (Z*A)[:,None]*S11
        + (Z*(1-A))[:,None]*S10
        + ((1-Z)*A)[:,None]*S01
        + ((1-Z)*(1-A))[:,None]*S00
    )

    SC = (
        (Z*A)[:,None]*C11
        + (Z*(1-A))[:,None]*C10
        + ((1-Z)*A)[:,None]*C01
        + ((1-Z)*(1-A))[:,None]*C00
    )

    SDT = np.array([
        np.interp(time[i], time_interest, SD[i])
        for i in range(len(test_index))
    ])

    SCT = np.array([
        np.interp(time[i], time_interest, SC[i])
        for i in range(len(test_index))
    ])
    # print(SDT.shape, SCT.shape)

    dFC = -np.hstack([np.diff(SC),np.zeros((SC.shape[0], 1))])
    dSD = np.hstack([np.diff(SD),np.zeros((SD.shape[0], 1))])

    atRisk = (time[:,None] >= time_interest).astype(float)

    JO = (
        (1-status)[:,None]
        * (1-atRisk)
        / (SDT[:,None]*SCT[:,None])
        -
        np.cumsum(
            atRisk*dFC/(SD*SC**2 + 1e-10),
            axis=1
        )
    )

    temp = Z/f - (1-Z)/(1-f)

    gamma1 = (
        pi1[:,None]*S11
        - pi0[:,None]*S01
    ) / delta[:,None]

    gamma0 = (
        (1-pi0[:,None])*S00
        - (1-pi1[:,None])*S10
    ) / delta[:,None]

    xi1 = omega[:,None]*S01
    xi0 = (1-omega[:,None])*S00

    # R01
    R01 = temp[:,None]*(A-1)[:,None]/delta[:,None]*atRisk/SC
    ipw01 = R01.copy()
    R01 += (1-temp[:,None]*(A-omega)[:,None]/delta[:,None])*gamma0
    R01 += temp[:,None]/delta[:,None]*xi0
    R01 += temp[:,None]*(A-1)[:,None]/delta[:,None]*JO*SD

    # R00
    R00 = temp[:,None]*(A-1)[:,None]/delta[:,None]*(atRisk-1)*status[:,None]/SCT[:,None]
    ipw00 = R00.copy()
    R00 += (1-temp[:,None]*(A-omega)[:,None]/delta[:,None])*gamma0
    R00 += temp[:,None]/delta[:,None]*xi0
    R00 += temp[:,None]*(A-1)[:,None]/delta[:,None] * np.cumsum(JO * dSD, axis=1)

    # R11
    R11 = temp[:,None]*A[:,None]/delta[:,None]*atRisk/SC
    ipw11 = R11.copy()
    R11 += (1-temp[:,None]*(A-omega)[:,None]/delta[:,None])*gamma1
    R11 -= temp[:,None]/delta[:,None]*xi1
    R11 += temp[:,None]*A[:,None]/delta[:,None]*JO*SD

    # R10
    R10 = temp[:,None]*A[:,None]/delta[:,None]*(atRisk-1)*status[:,None]/SCT[:,None]
    ipw10 = R10.copy()
    R10 += (1-temp[:,None]*(A-omega)[:,None]/delta[:,None])*gamma1
    R10 -= temp[:,None]/delta[:,None]*xi1
    R10 += temp[:,None]*A[:,None]/delta[:,None] * np.cumsum(JO * dSD, axis=1)

    return {
        'time.interest': time_interest,
        'R00': R00,
        'R01': R01,
        'R11': R11,
        'R10': R10,
        'ipw01': ipw01,
        'ipw00': ipw00,
        'ipw11': ipw11,
        'ipw10': ipw10,
        'gamma0': gamma0,
        'gamma1': gamma1,
    }



def train_nuisance_func(
    simdat,
    Kfold=5,
    tau=-1,accuracy=500,
    min_val=1e-2,
    method_SD=['rsf',100,20],
    method_SC=['rsf',100,20],
    method_pi=['gam',['X1','X2']],
    method_f=['gam',['X1','X2']],
):
    n_jobs = Kfold
    kf = KFold(
        n_splits=Kfold,
        shuffle=True,
        random_state=42
    )

    folds = list(kf.split(simdat))

    results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        verbose=10
    )(
        delayed(process_fold)(
            simdat,
            train_index,
            test_index,
            tau, accuracy,
            min_val,
            method_SD,
            method_SC,
            method_pi,
            method_f,
        )
        for train_index, test_index in folds
    )

    return {
        "results": results,
        "folds": folds,
        "tau": tau,
        "accuracy": accuracy
    }




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
    n = simdat.shape[0]
    results = model['results']
    folds = model['folds']
    accuracy = model['accuracy']
    Kfold = len(folds)

    R01 = np.zeros((n, accuracy)); R11 = np.zeros((n, accuracy)); R00 = np.zeros((n, accuracy)); R10 = np.zeros((n, accuracy))
    ipw01 = np.zeros((n, accuracy));ipw11 = np.zeros((n, accuracy)); ipw00 = np.zeros((n, accuracy));ipw10 = np.zeros((n, accuracy)) 
    or0 = np.zeros((n, accuracy)); or1 = np.zeros((n, accuracy))
    G_aug = np.zeros((n, accuracy)); G_or = np.zeros((n, accuracy)); G_ipw = np.zeros((n, accuracy))
    W_aug = np.zeros((n, accuracy)); W_or = np.zeros((n, accuracy)); W_ipw = np.zeros((n, accuracy))
    for i in range(Kfold):
        test = folds[i][1]
        R01_temp = results[i]['R01']; R01[test, :] = R01_temp
        R11_temp = results[i]['R11']; R11[test, :] = R11_temp
        R00_temp = results[i]['R00']; R00[test, :] = R00_temp
        R10_temp = results[i]['R10']; R10[test, :] = R10_temp
        or0_temp = results[i]['gamma0'];or0[test, :] = or0_temp
        or1_temp = results[i]['gamma1'];or1[test, :] = or1_temp
        ipw01_temp = results[i]['ipw01']; ipw01[test, :] = ipw01_temp
        ipw11_temp = results[i]['ipw11']; ipw11[test, :] = ipw11_temp
        ipw00_temp = results[i]['ipw00']; ipw00[test, :] = ipw00_temp
        ipw10_temp = results[i]['ipw10']; ipw10[test, :] = ipw10_temp

        G_aug[test, :] = np.apply_along_axis(
            lambda row: np.cumsum((np.mean(R11_temp, axis=0) / np.mean(R01_temp, axis=0)) * np.append(np.diff(-row), 0)),
            axis=1, arr=R00_temp)
        G_or[test, :] = np.apply_along_axis(
            lambda row: np.cumsum((np.mean(or1_temp, axis=0) / np.mean(or0_temp, axis=0)) * np.append(np.diff(-row), 0)),
            axis=1,arr=or0_temp)
        G_ipw[test, :] = np.apply_along_axis(
            lambda row: np.cumsum((np.mean(ipw11_temp, axis=0) / np.mean(ipw01_temp, axis=0)) * np.append(np.diff(-row), 0)),
            axis=1, arr=ipw00_temp)

        Lambda_aug = np.apply_along_axis(
            lambda row: np.cumsum(1 / np.mean(R01_temp, axis=0) * np.append(np.diff(-row), 0)),
            axis=1, arr=R00_temp).mean(axis=0)
        Lambda_or = np.apply_along_axis(
            lambda row: np.cumsum(1 / np.mean(or0_temp, axis=0) * np.append(np.diff(-row), 0)),
            axis=1,arr=or0_temp).mean(axis=0)
        Lambda_ipw = np.apply_along_axis(
            lambda row: np.cumsum(1 / np.mean(ipw01_temp, axis=0) * np.append(np.diff(-row), 0)),
            axis=1, arr=ipw00_temp).mean(axis=0)

        W_aug[test, :]= np.cumsum((R11_temp - R01_temp * (np.mean(R11_temp, axis=0) / np.mean(R01_temp, axis=0))) \
            * np.append(np.diff(Lambda_aug), 0), axis=1)
        W_or[test, :] = np.cumsum((or1_temp - or0_temp * (np.mean(or1_temp, axis=0) / np.mean(or0_temp, axis=0))) \
            * np.append(np.diff(Lambda_or), 0), axis=1)
        W_ipw[test, :]= np.cumsum((ipw11_temp - ipw01_temp * (np.mean(ipw11_temp, axis=0) / np.mean(ipw01_temp, axis=0))) \
            * np.append(np.diff(Lambda_ipw), 0), axis=1)
    

    R01_avg = R01.mean(axis=0); R11_avg = R11.mean(axis=0); R00_avg = R00.mean(axis=0); R10_avg = R10.mean(axis=0) 
    or0_avg = or0.mean(axis=0); or1_avg = or1.mean(axis=0)
    ipw01_avg = ipw01.mean(axis=0); ipw11_avg = ipw11.mean(axis=0); ipw00_avg = ipw00.mean(axis=0); ipw10_avg = ipw10.mean(axis=0)

    par_aug = np.log((R10_avg[0] - R10_avg) / G_aug.mean(axis=0)) 
    par_or = np.log((or1_avg[0] - or1_avg) / G_or.mean(axis=0))
    par_ipw = np.log((ipw10_avg[0] - ipw10_avg) / G_ipw.mean(axis=0))
    par_aug = replace_na_nearest(par_aug); G_aug *= np.exp(par_aug)
    par_or = replace_na_nearest(par_or); G_or *= np.exp(par_or)
    par_ipw = replace_na_nearest(par_ipw); G_ipw *= np.exp(par_ipw)

    W_aug = (np.repeat(R10[:, [0]], accuracy, axis=1) - R10) - G_aug - W_aug * np.exp(par_aug)
    W_or = (np.repeat(or1[:, [0]], accuracy, axis=1) - or1) - G_or - W_or * np.exp(par_or)
    W_ipw = (np.repeat(ipw10[:, [0]], accuracy, axis=1) - ipw10) - G_ipw - W_ipw * np.exp(par_ipw)

    std_aug2 = np.sqrt(W_aug.var(axis=0) / G_aug.mean(axis=0)**2 / n)
    std_or2  = np.sqrt(W_or.var(axis=0) / G_or.mean(axis=0)**2 / n)
    std_ipw2 = np.sqrt(W_ipw.var(axis=0) / G_ipw.mean(axis=0)**2 / n)

    return {
        "curve_surv0": {
            "gamma0_avg": or0_avg,
            "R0_avg": R01_avg,
            "ipw0_avg": ipw01_avg,
            "std_surv_aug0": np.std(R01, axis=0, ddof=1)/np.sqrt(n),
            "std_surv_ipw0": np.std(ipw01, axis=0, ddof=1)/np.sqrt(n),
            "std_surv_or0": np.std(or0, axis=0, ddof=1)/np.sqrt(n),
            "target_time": results[0]['time.interest']
        },
        "curve_surv1": {
            "gamma1_avg": or1_avg,
            "R1_avg": R11_avg,
            "ipw1_avg": ipw11_avg,
            "std_surv_aug1": np.std(R11, axis=0, ddof=1)/np.sqrt(n),
            "std_surv_ipw1": np.std(ipw11, axis=0, ddof=1)/np.sqrt(n),
            "std_surv_or1": np.std(or1, axis=0, ddof=1)/np.sqrt(n),
            "target_time": results[0]['time.interest']
        },
        "curve_hazard": {
            "par_aug": par_aug,
            "par_or": par_or,
            "par_ipw": par_ipw,
            "std_aug": std_aug2,
            "std_or": std_or2,
            "std_ipw": std_ipw2,
            "target_time": results[0]['time.interest']
        },
        "est": {
            "std_aug_est": std_aug2[-1],
            "std_or_est": std_or2[-1],
            "std_ipw_est": std_ipw2[-1],
            "par_aug_est": par_aug[-1],
            "par_or_est": par_or[-1],
            "par_ipw_est": par_ipw[-1],
        }
    }

def Marginal_Cox_Instrumental2(simdat, model):
    n = simdat.shape[0]
    results = model['results']
    folds = model['folds']
    accuracy = model['accuracy']
    Kfold = len(folds)

    R01 = np.zeros((n, accuracy)); ipw01 = np.zeros((n, accuracy)); or01 = np.zeros((n, accuracy))
    R11 = np.zeros((n, accuracy)); ipw11 = np.zeros((n, accuracy)); or11 = np.zeros((n, accuracy))
    R00 = np.zeros((n, accuracy)); ipw00 = np.zeros((n, accuracy)); or00 = np.zeros((n, accuracy))
    R10 = np.zeros((n, accuracy)); ipw10 = np.zeros((n, accuracy)); or10 = np.zeros((n, accuracy))
    G_aug_up=np.zeros((n, accuracy)); G_ipw_up=np.zeros((n, accuracy)); G_or_up=np.zeros((n, accuracy))
    G_aug_low=np.zeros((n, accuracy)); G_ipw_low=np.zeros((n, accuracy)); G_or_low=np.zeros((n, accuracy))
    W_aug1 = np.zeros((n, accuracy)); W_ipw1 = np.zeros((n, accuracy)); W_or1 = np.zeros((n, accuracy))
    W_aug2 = np.zeros((n, accuracy)); W_ipw2 = np.zeros((n, accuracy)); W_or2 = np.zeros((n, accuracy))
    W_aug3 = np.zeros((n, accuracy)); W_ipw3 = np.zeros((n, accuracy)); W_or3 = np.zeros((n, accuracy))
    W_aug4 = np.zeros((n, accuracy)); W_ipw4 = np.zeros((n, accuracy)); W_or4 = np.zeros((n, accuracy))
    for i in range(Kfold):
        test = folds[i][1]
        R01_temp = results[i]['R01']; R01[test, :] = R01_temp
        R11_temp = results[i]['R11']; R11[test, :] = R11_temp
        R00_temp = results[i]['R00']; R00[test, :] = R00_temp
        R10_temp = results[i]['R10']; R10[test, :] = R10_temp
        ipw01_temp = results[i]['ipw01']; ipw01[test, :] = ipw01_temp
        ipw11_temp = results[i]['ipw11']; ipw11[test, :] = ipw11_temp
        ipw00_temp = results[i]['ipw00']; ipw00[test, :] = ipw00_temp
        ipw10_temp = results[i]['ipw10']; ipw10[test, :] = ipw10_temp
        or01_temp = results[i]['gamma0']; or01[test, :] = or01_temp
        or11_temp = results[i]['gamma1']; or11[test, :] = or11_temp
        or00_temp = results[i]['gamma0']; or00[test, :] = or00_temp
        or10_temp = results[i]['gamma1']; or10[test, :] = or10_temp

        G_aug_up[test, :] = np.apply_along_axis(
            lambda row: np.cumsum(np.mean(R01_temp, axis=0)  * np.append(np.diff(row), 0)),
            axis=1, arr=R10_temp)
        G_aug_low[test, :] = np.apply_along_axis(
            lambda row: np.cumsum(np.mean(R11_temp, axis=0)  * np.append(np.diff(row), 0)),
            axis=1, arr=R00_temp)
        G_ipw_up[test, :] = np.apply_along_axis(
            lambda row: np.cumsum(np.mean(ipw01_temp, axis=0)  * np.append(np.diff(row), 0)),
            axis=1, arr=ipw10_temp)
        G_ipw_low[test, :] = np.apply_along_axis(
            lambda row: np.cumsum(np.mean(ipw11_temp, axis=0)  * np.append(np.diff(row), 0)),
            axis=1, arr=ipw00_temp)
        G_or_up[test, :] = np.apply_along_axis(
            lambda row: np.cumsum(np.mean(or01_temp, axis=0)  * np.append(np.diff(row), 0)),
            axis=1, arr=or10_temp)
        G_or_low[test, :] = np.apply_along_axis(
            lambda row: np.cumsum(np.mean(or11_temp, axis=0)  * np.append(np.diff(row), 0)),
            axis=1, arr=or00_temp)

        
        W_aug1[test, :] = np.cumsum(
            (R11_temp - np.mean(R11_temp, axis=0)) \
                * np.append(np.diff(np.mean(R00_temp, axis=0)), 0), axis=1)
        W_aug2[test, :] = np.apply_along_axis(
            lambda row: np.cumsum(np.mean(R11_temp, axis=0)  * np.append(np.diff(row), 0)),
            axis=1, arr=R00_temp-R00_temp.mean(axis=0))
        W_aug3[test, :] = np.cumsum(
            (R01_temp - np.mean(R01_temp, axis=0)) \
                * np.append(np.diff(np.mean(R10_temp, axis=0)), 0), axis=1)
        W_aug4[test, :] = np.apply_along_axis(
            lambda row: np.cumsum(np.mean(R01_temp, axis=0)  * np.append(np.diff(row), 0)),
            axis=1, arr=R10_temp-R10_temp.mean(axis=0))

        W_ipw1[test, :] = np.cumsum(
            (ipw11_temp - np.mean(ipw11_temp, axis=0)) \
                * np.append(np.diff(np.mean(ipw00_temp, axis=0)), 0), axis=1)
        W_ipw2[test, :] = np.apply_along_axis(
            lambda row: np.cumsum(np.mean(ipw11_temp, axis=0)  * np.append(np.diff(row), 0)),
            axis=1, arr=ipw00_temp-ipw00_temp.mean(axis=0))
        W_ipw3[test, :] = np.cumsum(
            (ipw01_temp - np.mean(ipw01_temp, axis=0)) \
                * np.append(np.diff(np.mean(ipw10_temp, axis=0)), 0), axis=1)
        W_ipw4[test, :] = np.apply_along_axis(
            lambda row: np.cumsum(np.mean(ipw01_temp, axis=0)  * np.append(np.diff(row), 0)),
            axis=1, arr=ipw10_temp-ipw10_temp.mean(axis=0))
        
        W_or1[test, :] = np.cumsum(
            (or11_temp - np.mean(or11_temp, axis=0)) \
                * np.append(np.diff(np.mean(or00_temp, axis=0)), 0), axis=1)
        W_or2[test, :] = np.apply_along_axis(
            lambda row: np.cumsum(np.mean(or11_temp, axis=0)  * np.append(np.diff(row), 0)),
            axis=1, arr=or00_temp-or00_temp.mean(axis=0))
        W_or3[test, :] = np.cumsum(
            (or01_temp - np.mean(or01_temp, axis=0)) \
                * np.append(np.diff(np.mean(or10_temp, axis=0)), 0), axis=1)
        W_or4[test, :] = np.apply_along_axis(
            lambda row: np.cumsum(np.mean(or01_temp, axis=0)  * np.append(np.diff(row), 0)),
            axis=1, arr=or10_temp-or10_temp.mean(axis=0))
        

    par_aug  = np.log(G_aug_up.mean(axis=0)/G_aug_low.mean(axis=0)); par_aug = replace_na_nearest(par_aug)
    par_ipw  = np.log(G_ipw_up.mean(axis=0)/G_ipw_low.mean(axis=0)); par_ipw = replace_na_nearest(par_ipw)
    par_or   = np.log(G_or_up.mean(axis=0)/G_or_low.mean(axis=0));   par_or  = replace_na_nearest(par_or)
    std_aug2 = np.sqrt(((W_aug1+W_aug2-(W_aug3 + W_aug4)*np.exp(-par_aug))**2).mean(axis=0)/(G_aug_low.mean(axis=0)**2)/n)
    std_ipw2 = np.sqrt(((W_ipw1+W_ipw2-(W_ipw3 + W_ipw4)*np.exp(-par_ipw))**2).mean(axis=0)/(G_ipw_low.mean(axis=0)**2)/n)
    std_or2  = np.sqrt(((W_or1 +W_or2 -(W_or3  + W_or4 )*np.exp(-par_or ))**2).mean(axis=0)/(G_or_low.mean(axis=0)**2)/n)

    return {
        "curve_surv0": {
            "gamma0_avg":   or01.mean(axis=0),
            "R0_avg":       R01.mean(axis=0),
            "ipw0_avg":     ipw01.mean(axis=0),
            "std_surv_aug0":R01.std(axis=0)/np.sqrt(n),
            "std_surv_ipw0":ipw01.std(axis=0)/np.sqrt(n),
            "std_surv_or0": or01.std(axis=0)/np.sqrt(n),
            "target_time":  results[0]['time.interest']
        },
        "curve_surv1": {
            "gamma1_avg":   or11.mean(axis=0),
            "R1_avg":       R11.mean(axis=0),
            "ipw1_avg":     ipw11.mean(axis=0),
            "std_surv_aug1":R11.std(axis=0)/np.sqrt(n),
            "std_surv_ipw1":ipw11.std(axis=0)/np.sqrt(n),
            "std_surv_or1": or11.std(axis=0)/np.sqrt(n),
            "target_time":  results[0]['time.interest']
        },
        "curve_hazard": {
            "par_aug": par_aug,
            "par_or": par_or,
            "par_ipw": par_ipw,
            "std_aug": std_aug2,
            "std_or": std_or2,
            "std_ipw": std_ipw2,
            "target_time": results[0]['time.interest']
        },
        "est": {
            "std_aug_est": std_aug2[-1],
            "std_or_est": std_or2[-1],
            "std_ipw_est": std_ipw2[-1],
            "par_aug_est": par_aug[-1],
            "par_or_est": par_or[-1],
            "par_ipw_est": par_ipw[-1],
        }
    }

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
        plt.fill_between(grp['time']-par['D_surv'](target_time**shape,1,par['psi']), grp['lower']-par['D_surv'](target_time**shape,1,par['psi']), grp['upper']-par['D_surv'](target_time**shape,1,par['psi']), alpha=0.3, label=f"{key} CI",color=color)
        plt.plot(grp['time']-par['D_surv'](target_time**shape,1,par['psi']), grp['mean']-par['D_surv'](target_time**shape,1,par['psi']), label=f"{key} Mean", linewidth=2,color=color)

    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.title("Estimated Survival Curves with Confidence Intervals")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.subplot(1, 2, 2)
    sns.set(style="whitegrid")

    for key, grp in df0[df0['group'].isin(['AIPW', 'OR', 'IPW'])].groupby('group'):
        color = color_dict.get(key, 'gray')
        plt.fill_between(grp['time']-par['D_surv'](target_time**shape,0,par['psi']), grp['lower']-par['D_surv'](target_time**shape,0,par['psi']), grp['upper']-par['D_surv'](target_time**shape,0,par['psi']), alpha=0.3, label=f"{key} CI",color=color)
        plt.plot(grp['time']-par['D_surv'](target_time**shape,0,par['psi']), grp['mean']-par['D_surv'](target_time**shape,0,par['psi']), label=f"{key} Mean", linewidth=2,color=color)
    
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.title("Estimated Survival Curves with Confidence Intervals")
    plt.legend(loc='upper right')
    plt.tight_layout()
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

    start = np.quantile(target_time, 0.05)
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
    plt.xlim(start, target_time[-1])
    # plt.ylim(-2, 2)
    plt.legend()
    plt.tight_layout()
    plt.show()
