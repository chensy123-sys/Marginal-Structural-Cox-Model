# Marginal Structural Cox Model  

## Script Overview  

### `Example.ipynb`  
This notebook provides a practical demonstration of how to use the `MCI.py`, `simdata.py`, and `Train_nuisance.py` modules.  

### `Train_nuisance.py`  
This script implements various machine learning models for nuisance function estimation:  
- **Survival models**:  
  - Random Survival Forest (RSF) and Cox Proportional Hazards (PH) model for estimating:  
    - $S_D(t|Z,A,X) := Pr(D \geq t | Z, A, X)$ (time-to-event survival function)  
    - $S_C(t|Z,A,X) := Pr(C \geq t | Z, A, X)$ (censoring survival function)  
- **Treatment and instrument models**:  
  - Random Forest (RF), Neural Network (NN), Kernel Regression (KR), and Generalized Additive Model (GAM) for estimating:  
    - $\pi(X,Z) := Pr(A=1|X,Z)$ (propensity score)  
    - $f(Z) := Pr(Z=1|X)$ (instrument propensity score)  

### `MCI.py`  
This script implements three estimators for causal inference:  
1. **Outcome Regression (OR)**  
2. **Inverse Probability Weighting (IPW)**  
3. **Augmented Inverse Probability Weighting (AIPW)**  

Key features:  
- Uses K-fold cross-fitting in nuisance function estimation (`Marginal_Cox_Instrumental()`).  
- Estimates nuisance functions via `train_nuisance_func()`.  
- Provides visualization tools:  
  - `plot_MCI_Surv()`: Plots potential survival curves for all three estimators.  
  - `plot_MCI_Hazard()`: Plots causal hazard curves.  

**Note on Standard Errors**:  
- The standard errors for IPW and OR estimators may not be consistent.  
- Only `std_aug_est` (for AIPW) provides consistent asymptotic variance estimation.  

### `simdat.py`  
Implements five different simulation scenarios for evaluating the methods.  

---

## Real Data Analysis: Illinois Unemployment Incentive Experiments  

This analysis examines the causal effects of two interventions from the 1984–1985 Illinois Department of Employment Security study:  
1. **Job Search Incentive Experiment (JSIE)**  
2. **Hiring Incentive Experiment (HIE)**  

The outcome of interest is the hazard rate of reemployment (`REHIREDT`).  

### How to Reproduce  
- Run `HIE.ipynb` and `JSIR.ipynb` to generate results stored in:  
  - `bootstrap_results_HIE.pkl`  
  - `bootstrap_results_JSIE.pkl`  

### Included Covariates  
The analysis adjusts for the following confounders:  
```python
['AGE', 'AVPREARN', 'CLAIMDT', 'DEPALLOW', 'ELIG2', 'L1QWAGES',
'L2QWAGES', 'L3QWAGES', 'L4QWAGES', 'MALE', 'NOHSUB', 'NOHSUB1',
'POSPEARN', 'POSQEARN', 'PREPEARN', 'RHIREARN', 'WGETOT1', 'WGETOT2',
'WGETOT3', 'WGETOT4', 'WGETOT5', 'WGETOT6',
'RACE_Hispanic', 'RACE_NativeAmerican', 'RACE_Other', 'RACE_White']
```

### Bootstrap Procedure  
The analysis was repeated **1,000 times** to construct confidence bands for the AIPW, IPW, and OR estimators.  

---

## Simulation Results  

Contains all simulation results from the accompanying article. Each setting evaluates eight different nuisance function estimation methods.  

### File Naming Convention  
Example: `cox_rsf_gam1000.pkl`  
- **First term (`cox`)** → $S_C(t|Z,A,X)$ estimated via Cox PH regression.  
- **Second term (`rsf`)** → $S_D(t|Z,A,X)$ estimated via Random Survival Forest.  
- **Third term (`gam`)** → $\pi(X,Z)$ and $f(X)$ estimated via Generalized Additive Model.  
- **Suffix (`1000`)** → Repeated times.  

This structured naming allows easy interpretation of the estimation approaches used in each simulation.
