import numpy as np
import pandas as pd
from sksurv.ensemble import RandomSurvivalForest
from scipy.interpolate import interp1d

from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

from pygam import LogisticGAM, s
from sklearn.preprocessing import StandardScaler

from pygam import LogisticGAM, s
from sklearn.model_selection import GridSearchCV
import copy

def ps_spline(simdat, train_index, test_index, min_val=10e-2):

    
    # Prepare training data subsets
    train_data = simdat.iloc[train_index].copy()
    n_splines=20
    # Fit GAM for A|Z=1
    train_z1 = train_data[train_data['Z'] == 1].drop(columns=['Z', 'time', 'status'])
    gam_a_z1 = LogisticGAM(s(0, n_splines=n_splines) + s(1, n_splines=n_splines),lam=5).fit(train_z1[['X1', 'X2']], train_z1['A'])
    # gam_a_z1 = optimize_gam(train_z1[['X1', 'X2']], train_z1['A'])

    # Fit GAM for A|Z=0
    train_z0 = train_data[train_data['Z'] == 0].drop(columns=['Z', 'time', 'status'])
    gam_a_z0 = LogisticGAM(s(0, n_splines=n_splines) + s(1, n_splines=n_splines),lam=5).fit(train_z0[['X1', 'X2']], train_z0['A'])
    # gam_a_z0 = optimize_gam(train_z0[['X1', 'X2']], train_z0['A'])
    
    # Fit GAM for Z
    train_z = train_data.drop(columns=['A', 'time', 'status'])
    gam_z = LogisticGAM(s(0, n_splines=n_splines) + s(1, n_splines=n_splines),lam=5).fit(train_z[['X1', 'X2']], train_z['Z'])
    # gam_z = optimize_gam(train_z[['X1', 'X2']], train_z['Z'])

    # Prepare test data
    test_data = simdat.iloc[test_index].copy()
    
    # Make predictions
    pi1 = gam_a_z1.predict_proba(test_data[['X1', 'X2']])
    pi0 = gam_a_z0.predict_proba(test_data[['X1', 'X2']])
    f = gam_z.predict_proba(test_data[['X1', 'X2']])
    
    # Clip values to avoid extremes
    f_clipped = np.clip(f, min_val, 1-min_val)
    
    # Calculate delta
    delta = np.maximum(min_val, np.abs(pi1 - pi0)) * np.sign(pi1 - pi0)
    
    return {
        'pi0': pi0,
        'pi1': pi1,
        'f': f_clipped,
        'delta': delta,
        'omega': pi0
    }


import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist

def adaptive_kernel_logistic_predict(X_train, y_train, X_test, k=20):
    """
    Adaptive bandwidth kernel logistic regression estimator for estimating P(Y=1 | X)

    Parameters
    ----------
    X_train : ndarray of shape (n_samples, n_features)
        Covariates of the training data

    y_train : ndarray of shape (n_samples,)
        Binary response variable (0 or 1)

    X_test : ndarray of shape (m_samples, n_features)
        Test points to make predictions on

    k : int
        Number of nearest neighbors used to determine the adaptive bandwidth

    Returns
    -------
    probs : ndarray of shape (m_samples,)
        Predicted probabilities P(Y=1 | X) for each test point
    """
    # 1. Compute the distance to the k-th nearest neighbor for each test point, used as bandwidth
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_train)
    distances_k, _ = nbrs.kneighbors(X_test)
    adaptive_bandwidths = distances_k[:, -1] + 1e-6  # Add small value to prevent division by zero

    # 2. Compute Euclidean distances between each test point and all training points
    dists = cdist(X_test, X_train, 'euclidean')  # shape: (m, n)

    # 3. Compute Gaussian kernel weights
    weights = np.exp(-0.5 * (dists / adaptive_bandwidths[:, None]) ** 2)

    # 4. Kernel-smoothed estimate of P(Y=1 | X)
    numer = (weights * y_train).sum(axis=1)
    denom = weights.sum(axis=1)
    probs = numer / (denom + 1e-8)
    return probs


def ps_kernel(simdat, train_index, test_index, min_val=10e-2):
    # Prepare training data subsets
    train_data = simdat.iloc[train_index].copy()
    n_splines=20
    # Fit GAM for A|Z=1
    train_z1_X = np.array(train_data[train_data['Z'] == 1].drop(columns=['A', 'Z', 'time', 'status']))
    train_z1_y = np.array(train_data[train_data['Z'] == 1]['A'])

    # Fit GAM for A|Z=0
    train_z0_X = np.array(train_data[train_data['Z'] == 0].drop(columns=['A','Z', 'time', 'status']))
    train_z0_y = np.array(train_data[train_data['Z'] == 0]['A'])
    
    # Fit GAM for Z
    train_z_X = np.array(train_data.drop(columns=['Z', 'A', 'time', 'status']))
    train_z_y = np.array(train_data['Z'])

    # Prepare test data
    test_data = np.array(simdat.iloc[test_index].drop(columns=['A','Z', 'time', 'status']).copy())
    
    # Make predictions
    # print(train_z1_X)
    # print(train_z1_y)
    # print(test_data)

    k = int(train_data.shape[0]*0.1)
    pi1 = adaptive_kernel_logistic_predict(train_z1_X, train_z1_y, test_data, k)
    pi0 = adaptive_kernel_logistic_predict(train_z0_X, train_z0_y, test_data, k)
    f = adaptive_kernel_logistic_predict(train_z_X, train_z_y, test_data, k)
    
    # Clip values to avoid extremes
    f_clipped = np.clip(f, min_val, 1-min_val)
    
    # Calculate delta
    delta = np.maximum(min_val, np.abs(pi1 - pi0)) * np.sign(pi1 - pi0)
    
    return {
        'pi0': pi0,
        'pi1': pi1,
        'f': f_clipped,
        'delta': delta,
        'omega': pi0
    }






def S_coxph(simdat, train_index, test_index, tau=-1, accuracy=500, reverse=False, min_val=10e-2):
    # Prepare time grid
    simdat = copy.deepcopy(simdat)
    if tau == -1:
        target_time = np.linspace(0, simdat['time'].max(), accuracy)
    else:
        target_time = np.linspace(0, tau, accuracy)
    
    m = len(simdat.loc[test_index, 'Z'])
    
    if reverse:
        simdat['status'] = 1 - simdat['status']
        max_time = simdat['time'].max()
        simdat.loc[simdat['time'] == max_time, 'status'] = 1 - simdat.loc[simdat['time'] == max_time, 'status']
    
    # Prepare training data
    train_data = simdat.iloc[train_index].copy()
    
    # Fit Cox model
    cph = CoxPHFitter()
    cph.fit(train_data, duration_col='time', event_col='status')
    # print(cph.summary)
    # Get baseline survival function
    baseline_survival = cph.baseline_survival_
    time_grid = baseline_survival.index.values
    base_surv_vals = baseline_survival.values.flatten()
    
    # Prepare test data variants
    simdat_test = simdat.iloc[test_index].copy()
    
    simdat_test11 = simdat_test.copy()
    simdat_test11['Z'] = 1
    simdat_test11['A'] = 1
    
    simdat_test10 = simdat_test.copy()
    simdat_test10['Z'] = 1
    simdat_test10['A'] = 0
    
    simdat_test01 = simdat_test.copy()
    simdat_test01['Z'] = 0
    simdat_test01['A'] = 1
    
    simdat_test00 = simdat_test.copy()
    simdat_test00['Z'] = 0
    simdat_test00['A'] = 0
    
    # Get linear predictors
    lp11 = cph.predict_log_partial_hazard(simdat_test11)
    lp10 = cph.predict_log_partial_hazard(simdat_test10)
    lp01 = cph.predict_log_partial_hazard(simdat_test01)
    lp00 = cph.predict_log_partial_hazard(simdat_test00)
    
    # Calculate survival functions
    pred_11 = np.array([base_surv_vals ** np.exp(lp) for lp in lp11])
    pred_10 = np.array([base_surv_vals ** np.exp(lp) for lp in lp10])
    pred_01 = np.array([base_surv_vals ** np.exp(lp) for lp in lp01])
    pred_00 = np.array([base_surv_vals ** np.exp(lp) for lp in lp00])
    
    # Interpolate to target time points
    def interpolate_survival(pred, time_grid, target_time, m):
        S = np.zeros((m, len(target_time)))
        for i in range(m):
            # Add time=0 point with survival=1
            extended_time = np.concatenate([[0], time_grid])
            extended_surv = np.concatenate([[1], pred[i,:]])
            
            # Create interpolation function
            interp_func = interp1d(extended_time, extended_surv, kind='linear', 
                                 bounds_error=False, fill_value=(1, extended_surv[-1]))
            
            # Evaluate at target times
            S[i,:] = interp_func(target_time)
        return S
    
    S11 = interpolate_survival(pred_11, time_grid, target_time, m)
    S10 = interpolate_survival(pred_10, time_grid, target_time, m)
    S01 = interpolate_survival(pred_01, time_grid, target_time, m)
    S00 = interpolate_survival(pred_00, time_grid, target_time, m)
    

    
    # Apply minimum value threshold
    S11 = np.maximum(S11, min_val)
    S10 = np.maximum(S10, min_val)
    S01 = np.maximum(S01, min_val)
    S00 = np.maximum(S00, min_val)
    
    return {
        'time.interest': target_time,
        'S11': S11,
        'S10': S10,
        'S01': S01,
        'S00': S00
    }

def S_rfs(simdat, train_index, test_index, tau=-1, accuracy=500,
           reverse=False, min_val=10e-2, ntree=200, nodesize=20,
           nsplit=10, importance=True):
    
    # simdat = copy.deepcopy(simdat)
    # Create target time points
    if tau == -1:
        target_time = np.linspace(0, simdat['time'].max(), accuracy)
    else:
        target_time = np.linspace(0, tau, accuracy)
    
    print(simdat['Z'])
    print(test_index)
    m = len(simdat.loc[test_index, 'Z'])
    
    # Prepare data for survival analysis
    if reverse:
        status = 1 - simdat['status']
        max_time = simdat['time'].max()
        status[simdat['time'] == max_time] = 1 - status[simdat['time'] == max_time]
    else:
        status = simdat['status']
    
    # Create structured array for survival analysis
    y_train = np.empty(len(train_index), dtype=[('status', 'bool'), ('time', 'float64')])
    y_train['status'] = status[train_index].astype(bool)
    y_train['time'] = simdat.loc[train_index, 'time']
    
    # X_train = simdat.loc[train_index].drop(['time', 'status','X1','A','Z'], axis=1)
    X_train = simdat.loc[train_index].drop(['time', 'status'], axis=1)
    # Fit Random Survival Forest
    rsf = RandomSurvivalForest(n_estimators=ntree,
                               min_samples_leaf=nodesize,
                               max_features='sqrt',
                               random_state=0)
    rsf.fit(X_train, y_train)
    
    # Prepare test data variants
    simdat_test = simdat.loc[test_index].copy()
    
    # Create four scenarios
    test_data = {
        '11': simdat_test.assign(Z=1, A=1).drop(['time', 'status'], axis=1),
        '10': simdat_test.assign(Z=1, A=0).drop(['time', 'status'], axis=1),
        '01': simdat_test.assign(Z=0, A=1).drop(['time', 'status'], axis=1),
        '00': simdat_test.assign(Z=0, A=0).drop(['time', 'status'], axis=1)
    }
    
    results = {}
    for key, data in test_data.items():
        surv_funcs = rsf.predict_survival_function(data, return_array=True)
        unique_times = rsf.unique_times_
        
        # Interpolate to target time points
        interp_surv = np.zeros((m, len(target_time)))
        for i in range(m):
            # Add t=0 with survival=1
            extended_times = np.concatenate([[0], unique_times])
            extended_surv = np.concatenate([[1], surv_funcs[i]])
            
            # Create interpolation function
            f = interp1d(extended_times, extended_surv, 
                         kind='linear', bounds_error=False, 
                         fill_value=(1, extended_surv[-1]))
            
            interp_surv[i] = f(target_time)
        
        results[f'S{key}'] = np.maximum(interp_surv, min_val)
    
    return {
        'time.interest': target_time,
        'S11': results['S11'],
        'S10': results['S10'],
        'S01': results['S01'],
        'S00': results['S00']
    }


















import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler

class FlexibleNN(nn.Module):
    def __init__(self, input_dim=2, hidden_layers=[10]):
        super(FlexibleNN, self).__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

def train_nn(X_train, y_train, X_val, y_val, hidden_layers, epochs=1000, 
             lr=0.01, batch_size=5000, patience=10):
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = FlexibleNN(input_dim=X_train.shape[1], hidden_layers=hidden_layers)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = np.inf
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        for xb, yb in dataloader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        # Evaluate on validation set after every epoch
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            val_preds = model(X_val_tensor)
            val_loss = criterion(val_preds, y_val_tensor)

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0  # Reset counter
        else:
            epochs_without_improvement += 1

        # Stop training if no improvement for `patience` epochs
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    return model

def ps_nn(simdat, train_index, test_index, hidden_layers=[10], min_val=10e-2, patience=10, lr=0.01):
    scaler = StandardScaler()
    train_data = simdat.iloc[train_index].copy()
    test_data = simdat.iloc[test_index].copy()

    X_train = scaler.fit_transform(train_data[['X1', 'X2']])
    X_test = scaler.transform(test_data[['X1', 'X2']])
    y_train = train_data['A'].values
    y_test = test_data['A'].values
    z_train = train_data['Z'].values
    z_test = test_data['Z'].values

    # Validation set from training data
    val_index = int(len(train_data) * 0.8)  # 80% for training, 20% for validation
    X_val = X_train[val_index:]
    y_val = y_train[val_index:]
    z_val = z_train[val_index:]

    X_train = X_train[:val_index]
    y_train = y_train[:val_index]
    z_train = z_train[:val_index]

    # A | Z = 1
    z1_data = train_data[train_data['Z'] == 1]
    model_a_z1 = train_nn(
        scaler.transform(z1_data[['X1', 'X2']]),
        z1_data['A'].values,
        X_val[z_val == 1],
        y_val[z_val == 1],
        hidden_layers=hidden_layers,
        patience=patience,
        lr=lr
    )

    # A | Z = 0
    z0_data = train_data[train_data['Z'] == 0]
    model_a_z0 = train_nn(
        scaler.transform(z0_data[['X1', 'X2']]),
        z0_data['A'].values,
        X_val[z_val == 0],
        y_val[z_val == 0],
        hidden_layers=hidden_layers,
        patience=patience,
        lr=lr
    )

    # print(X_val.shape,z_val.shape)
    # print(X_train.shape,train_data['Z'].values.shape)
    # Z ~ X
    model_z = train_nn(
        X_train,
        z_train,
        X_val,
        z_val,
        hidden_layers=hidden_layers,
        patience=patience,
        lr=lr
    )
    # print('#################')
    # Prediction
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    pi1 = model_a_z1(X_test_tensor).detach().numpy().flatten()
    pi0 = model_a_z0(X_test_tensor).detach().numpy().flatten()
    f = model_z(X_test_tensor).detach().numpy().flatten()

    f_clipped = np.clip(f, min_val, 1 - min_val)
    delta = np.maximum(min_val, np.abs(pi1 - pi0)) * np.sign(pi1 - pi0)

    return {
        'pi0': pi0,
        'pi1': pi1,
        'f': f_clipped,
        'delta': delta,
        'omega': pi0
    }
