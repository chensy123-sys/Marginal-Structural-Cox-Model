import simdata
import numpy as np
from scipy.special import expit
from MCI import train_nuisance_func, Marginal_Cox_Instrumental, plot_MCI_Hazard, plot_MCI_Surv
from MCI import Marginal_Cox_Instrumental2
import matplotlib.pyplot as plt
import random
import pickle; import copy
import os
import contextlib
from par import *


np.random.seed(2025); random.seed(2025);par =par1
res = {
    'std': np.zeros(Looptime),
    'std_fSC': np.zeros(Looptime),
    'std_SD': np.zeros(Looptime),
    'std2': np.zeros(Looptime),
    'std2_fSC': np.zeros(Looptime),
    'std2_SD': np.zeros(Looptime),
    'par': np.zeros(Looptime),
    'par_fSC': np.zeros(Looptime),
    'par_SD': np.zeros(Looptime),
    'par2': np.zeros(Looptime),
    'par2_fSC': np.zeros(Looptime),
    'par2_SD': np.zeros(Looptime),
}
for i in range(Looptime):
    sim = simdata.sim_cox_instrumental(n, par)
    simdat = sim['simdat']
    with open(os.devnull, "w") as fnull:
        with contextlib.redirect_stdout(fnull), \
            contextlib.redirect_stderr(fnull):

            model_true = train_nuisance_func(
                simdat=simdat,accuracy=200,tau=tau,
                min_val=0.05,Kfold=5,
                method_SD=['rsf',['X1','X2','Z','A'],100,10], method_SC=['cox',['Z','A','X1','X2'],100,10], 
                method_pi=['gam',['X1','X2']], method_f=['gam',['X1','X2']]
            )
            model_fSC = train_nuisance_func(
                simdat=simdat,accuracy=200,tau=tau,
                min_val=0.05,Kfold=5,
                method_SD=['cox',[],100,10], method_SC=['cox',['Z','A','X1','X2'],100,10],
                method_pi=['gam',['X1','X2']], method_f=['gam',['X1','X2']]
            )
            model_SD = train_nuisance_func(
                simdat=simdat,accuracy=200,tau=tau,
                min_val=0.05,Kfold=5,
                method_SD=['rsf',['Z','A','X1','X2'],100,10], method_SC=['cox',[],100,10],
                method_pi=['gam',['X1','X2']], method_f=['gam',[]]
            )

            fit_MCI = Marginal_Cox_Instrumental(simdat, model_true)
            fit_MCI_fSC = Marginal_Cox_Instrumental(simdat, model_fSC)
            fit_MCI_SD = Marginal_Cox_Instrumental(simdat, model_SD)

            fit_MCI2 = Marginal_Cox_Instrumental2(simdat, model_true)
            fit_MCI2_fSC = Marginal_Cox_Instrumental2(simdat, model_fSC)
            fit_MCI2_SD = Marginal_Cox_Instrumental2(simdat, model_SD)

    res['std'][i] = fit_MCI['est']['std_aug_est']
    res['std_fSC'][i] = fit_MCI_fSC['est']['std_aug_est']
    res['std_SD'][i] = fit_MCI_SD['est']['std_aug_est']
    res['std2'][i] = fit_MCI2['est']['std_aug_est']
    res['std2_fSC'][i] = fit_MCI2_fSC['est']['std_aug_est']
    res['std2_SD'][i] = fit_MCI2_SD['est']['std_aug_est']
    
    res['par'][i] = fit_MCI['est']['par_aug_est']
    res['par_fSC'][i] = fit_MCI_fSC['est']['par_aug_est']
    res['par_SD'][i] = fit_MCI_SD['est']['par_aug_est']
    res['par2'][i] = fit_MCI2['est']['par_aug_est']
    res['par2_fSC'][i] = fit_MCI2_fSC['est']['par_aug_est']
    res['par2_SD'][i] = fit_MCI2_SD['est']['par_aug_est']
    print(i + 1, flush=True)
    
    # RES.append(res); j+=1


with open('scenario1_'+str(n)+'.pkl', 'wb') as f:
    pickle.dump(res, f)
