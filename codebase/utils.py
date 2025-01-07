import numpy as np
import matplotlib.pyplot as plt

def simulate_sir_simple(lam, mu, I_0, T, C_returned=False):
    # initialize arrays for S, I, R
    S = np.zeros(T+1)
    I = np.zeros(T+1)
    R = np.zeros(T+1)
    N_arr = np.zeros(T+1)
    # define odes for S, I, R
    def dS(S, I):
        return -lam*S*I
    def dI(S, I):
        return lam*S*I - mu*I
    def dR(I):
        return mu*I
    
    # initial conditions
    S[0] = 1 - I_0  
    I[0] = I_0
    R[0] = 0
    N_arr[0] = S[0] + I[0] + R[0]

    # simulate the ODEs
    for t in range(T):
        S[t+1] = S[t] + dS(S[t], I[t])
        I[t+1] = I[t] + dI(S[t], I[t])
        R[t+1] = R[t] + dR(I[t])
        N_arr[t+1] = S[t+1] + I[t+1] + R[t+1]
    # calculate the daily changes
    delta_S = np.zeros(T)
    delta_R = np.zeros(T)
    for t in range(1, T):
        delta_S[t] = S[t-1] - S[t]
        delta_R[t] = R[t-1] - R[t]

    X = np.array([(delta_S[i], delta_R[i]) for i in range(T)])
    C = np.array([(S[i], I[i], R[i]) for i in range(T)])
    # if not np.all(N_arr == 1.):
    #     print("Error: N is not constant")
    if C_returned:
        return X, C
    else:
        return X
    

# plot the min and max parameters to see, for the later prior definition  (the C is removed from the function output, add to test new parameters)
def plot_sir_single(lam, mu, I0, T=100, title='SIR Simulation'):
    # simulate parameters 
    _ , C= simulate_sir_simple(lam, mu, I0, T, C_returned=True)

    # plot 
    plt.plot(C[:, 0], label='S')
    plt.plot(C[:, 1], label='I')
    plt.plot(C[:, 2], label='R')
    plt.title(title)
    plt.legend()
    plt.show()


# # plot the min and max parameters to see, for the later prior definition  (the C is removed from the function output, add to test new parameters)
# def plot_sir_minmax(lam_min, lam_max, mu_min, mu_max, I0_min, I0_max, T=100, N = 10000):
#     # simulate min parameters 
#     _ , C_min = simulate_sir_simple(lam_min, mu_min, I0_min, T, N=N)

#     # simulate max parameters
#     _ , C_max = simulate_sir_simple(lam_max, mu_max, I0_max, T, N=N)

#     # plot 
#     fig, ax = plt.subplots(1, 2, figsize=(12, 6))
#     ax[0].plot(C_min[:, 0], label='S')
#     ax[0].plot(C_min[:, 1], label='I')
#     ax[0].plot(C_min[:, 2], label='R')
#     ax[0].set_title('Min Parameters')
#     ax[0].legend()
#     ax[1].plot(C_max[:, 0], label='S')
#     ax[1].plot(C_max[:, 1], label='I')
#     ax[1].plot(C_max[:, 2], label='R')
#     ax[1].set_title('Max Parameters')
#     ax[1].legend()
#     plt.show()

# plot_sir_minmax(lam_min=0.095, lam_max=0.2, mu_min=0.04, mu_max=0.07, I0_min=0.01, I0_max=0.05)

