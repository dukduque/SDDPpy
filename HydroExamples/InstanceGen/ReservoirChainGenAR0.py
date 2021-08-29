'''
Created on Jan 2, 2018

@author: dduque

'''
if __name__ == '__main__':
    from os import path
    import sys
    sys.path.append(path.abspath('/Users/dduque/Dropbox/WORKSPACE/SDDP'))
    sys.path.append(path.abspath('/Users/dduque/Dropbox/WORKSPACE/SDDP/HydroExamples'))
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import pickle
from HydroModel import hydro_path


def gen_instance(num_reservoirs=10, T=12, num_outcomes=30, lognormal=True, simulate=False):
    '''
    Generate a random instance consisting of:
        - Autoregresive matrices (stored as dictionaries)
        - Initial inflow vector (matrix for lag > 0)
        - Innovations of the autoregressive process
    '''
    np.random.seed(0)
    season = 12
    
    active_reservoirs = [0, 3, 6]  # list(range(num_reservoirs))
    b = [5, 5, 5, 15, 5, 5, 5, 5, 5, 10, 5, 5]
    mu = [0.6, 0.6, 0.6, 1.5, 0.6, 0.6, 0.6, 0.6, 0.6, 1, 0.6, 0.5]
    sigma = [0.3, 0.3, 0.3, 0.5, 0.3, 0.3, 0.3, 0.3, 0.3, 0.5, 0.3, 0.4]
    # b = [10, 10, 10, 27, 10, 10, 10, 10, 15, 20, 15, 10]
    # mu = [1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1, 1.5, 1, 1.6]
    # sigma = [0.3, 0.3, 0.3, 0.7, 0.3, 0.3, 0.3, 0.3, 0.5, 0.6, 0.5, 0.3]
    
    RHS_noise = np.zeros(shape=(num_reservoirs, num_outcomes, T))
    for t in range(T):
        month = t if t < season else t % season
        mu_t = mu[month] * np.ones(nr)
        cov_mat = np.zeros((nr, nr))
        for i in range(nr):
            for j in range(nr):
                if i == j:
                    cov_mat[i, j] = sigma[month]**2
                else:
                    cov_mat[i, j] = sigma[month]**2 * 0.9
        if lognormal:
            RHS_corralated = np.exp(np.random.multivariate_normal(mu_t, cov_mat, size=(num_outcomes)))
            RHS_corralated = RHS_corralated.transpose()
            if t < season:
                RHS_noise[active_reservoirs, :, t] = np.maximum(0, b[t] - RHS_corralated[active_reservoirs, :])
            else:
                RHS_noise[:, :, t] = RHS_noise[:, :, t - season]
        else:
            if t < season:
                RHS_noise[active_reservoirs, :, t] = np.random.uniform(0, b[t], size=(num_outcomes))
            else:
                RHS_noise[:, :, t] = RHS_noise[:, :, t - season]
    if simulate:
        simulate_model(RHS_noise, T, num_reservoirs)
    
    instance = HydroRndInstance(None, None, RHS_noise)
    return instance


def simulate_model(RHS_noise, T, nrs):
    prices = [18 + round(5 * np.sin(0.5 * (x - 2)), 2) for x in range(0, T)]
    num_reservoirs = nr
    plt.figure(1)
    num_reps = 300
    res_ref = [0, 1, 3]
    np.random.seed(res_ref)
    mean_res_ref = {rr: np.zeros((T)) for rr in res_ref}
    for replica in range(num_reps):
        
        plot_data = {rr: [] for rr in res_ref}
        inflows = list()
        for t in range(0, T):
            outcome = np.random.randint(len(RHS_noise[0, :, t]))
            new_inflows = RHS_noise[:, outcome, t].copy()
            for rr in res_ref:
                plot_data[rr].append(new_inflows[rr])
        for (i, rr) in enumerate(res_ref):
            mean_res_ref[rr] = mean_res_ref[rr] + np.array(plot_data[rr])
            plotpos = int('%i1%i' % (len(res_ref), i + 1))
            plt.subplot(plotpos)
            plt.plot(plot_data[rr], alpha=0.5)
        data_replica = np.array([plot_data[r1] for r1 in res_ref])
    for (i, rr) in enumerate(res_ref):
        mean_res_ref[rr] = mean_res_ref[rr] / num_reps
        plotpos = int('%i1%i' % (len(res_ref), i + 1))
        plt.subplot(plotpos)
        plt.plot(mean_res_ref[rr], linewidth=2, color='black', linestyle='--')
        #plt.plot(prices, linewidth=2, color='red', linestyle='--')
        plt.grid()
    plt.show()


class HydroRndInstance():
    def __init__(self, ar_matrices, initial_inflows, RHS_noise):
        self.ar_matrices = ar_matrices
        self.inital_inflows = initial_inflows
        self.RHS_noise = RHS_noise


def read_instance(file_name='hydro_rnd_instance_R200_UD1_T120_LAG1_OUT10K_AR.pkl', lag=None):
    '''
        Read instance from file and returns a HydroRndInstance object.
    '''
    file_name_path = hydro_path + '/data/' + file_name
    if lag is None:
        file_name_path = hydro_path + '/data/' + 'hydro_rnd_instance_R1000_UD1_T120_LAG%i_OUT30_AR.pkl' % (lag)
    with open(file_name_path, 'rb') as input:
        instance = pickle.load(input)
        return instance


if __name__ == '__main__':
    nr = 10
    T = 48
    outcomes = 10_000
    file_name_pat = None
    file_name_path = hydro_path + f'/data/hydro_rnd_instance_R{nr}_T{T}_OUT10K_AR0_UNIFORM.pkl'
    with open(file_name_path, 'wb') as output:
        instance = gen_instance(nr, T, num_outcomes=outcomes, lognormal=False, simulate=True)
        pickle.dump(instance, output, pickle.HIGHEST_PROTOCOL)
