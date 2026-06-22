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
['AGE', 'CLAIMDT', 'MALE',
'AVPREARN', 'PREPEARN',
'RACE_Hispanic', 'RACE_NativeAmerican', 'RACE_Other', 'RACE_White']
```

### Bootstrap Procedure  
The analysis was repeated **500 times** to construct confidence bands for the AIPW, IPW, and OR estimators.  
