import numpy as np
import matplotlib.pyplot as plt
import gmcmc
import sys
from numpy import exp, log

num_samples = 50000
h_metropolis_array = np.array([20000, 1, 1, 10])
np.random.seed(2)
# This is the number of observation days
global_T = 21
m = 4
d = global_T
MAXVICTIMS = 50000

# Simple exmplicit general logistic solution
def Xt (Q, q, v, X0, t):
    A = (-1. + Q/X0) ** v
    res = (1 + A*exp(-q * t)) ** (1. / v)
    return Q / res

# G goes from R^3, x= (Q, q, v, X0) to R^T, i.e. number of observed days
def G(x):
    res = []
    for i in range(global_T):
        res.append(Xt(x[0], x[1], x[2], x[3], i))
    return np.asanyarray(res)

# Checking now the domain of G is taylored on this problem of dimension 4
def inDomainG(x):
    if x[0] < 500 or x[0] > MAXVICTIMS:
        return False
    if x[1] < 0 or x[1] > 2:
        return False
    if x[2] < 0 or x[2] > 2:
        return False
    if x[3] < 100 or x[3] > MAXVICTIMS:
        return False
    return True

def gradX(Q, q, v, X0, t):
    # For a FIXED t, take the partial derivatives w.r.t Q, then q, v, X0
    # and store each in the i-th component
    # Some local variables will be very useful as a shortcut
    gradient = np.zeros(4)
    A = (Q / X0 - 1.) ** v
    D = (1 + A*exp(-q * t))
    # w.r.t Q
    gradient[0] = \
    D ** (1./v) - Q * (D**((1-v)/v)) * exp(-q*t)/X0 * ((Q/X0 -1)**(v-1))
    gradient[0] /= D**(2/v)
    # w.r.t q
    gradient[1] = (t/v) * exp(-q*t) * A * Q * (D ** (-(1+v)/v))
    # w.r.t v
    star3 = exp(v * log(Q/X0 -1) - q*t) * log(Q/X0 -1)
    star2 = star3 / (1 + exp(v * log(Q/X0 -1) - q*t))
    right = log(1 + exp(v*log(Q/X0 -1) * q * t))/(v**2)
    star1 = right - star2/v
    star0 = Q*exp(-1/v * log(1 + exp(v*log(Q/X0 -1) -q*t)))
    gradient[2] = star0 * star1
    # w.r.t X0
    gradient[3] = ((Q/X0) **2) * exp(-q*t) *((Q/X0 -1) ** v-1) * (D**(-(1+v)/v))
    return gradient

# JacobianG should be implemented now, but depends on gradientX above TO DO
# The only "true" step to implement is the jacobian of G
def jacobianG(x):
    # Memento: x[0] = Q, x[1] = q, x[2] = v, x[3] = X0
    res = np.asanyarray([gradX(x[0],x[1],x[2],x[3],i) for i in range(global_T)])
    return res.reshape(global_T, 4)

TOY_MODEL = True
if TOY_MODEL:
    tmpx0 = np.random.uniform(100, 20000)
    true_x = np.array([tmpx0 + np.random.uniform(500,25000),
                    np.random.uniform(0, 2),
                    np.random.uniform(0, 2), tmpx0])
    print("True x: ", true_x)
    y = G(true_x)
    print("y = ", y)
    sigmas = y / 20.
    cov_matrix = np.identity(d)
    cov_matrix = np.array([cov_matrix[i] * sigmas[i] for i in range(d)])
    y = G(true_x) + np.random.multivariate_normal(np.zeros(d), cov_matrix)
    print("Perturbed y: ", y)
else:
    import pandas as pd
    df = pd.read_csv("italy_10Apr.csv")
    y = np.asanyarray(df['Victims'])
    sigmas = y / 20.
    cov_matrix = np.identity(d)
    cov_matrix = np.array([cov_matrix[i] * sigmas[i] for i in range(d)])


skip_n_samples = 1 # With 1, no samples are skipped
parallel = True
conv_samples = 1000

#### --- end of the common section with read ---- #
SAMPLING_SINGLE_CHAIN = True #False
SAMPLING_TO_CHECK_CONVERGENCE = False #True #alse #True


if SAMPLING_SINGLE_CHAIN:
    print("Constructing a single full chain")
    start_x = np.array([np.random.uniform(y[-1], MAXVICTIMS),
                    np.random.uniform(0, 2),
                    np.random.uniform(0, 2), y[0]])
    print("Starting point: ", start_x)
    X, runtime, _ , _ = gmcmc.chain_rwMetropolis(start_x,
       h_metropolis_array, y, G, sigmas, inDomainG, num_samples, skip_n_samples)

    info_str = "INFOSIMU: chain_rwMetropolis, h = " + str(h_metropolis_array)+\
            " runtime: " + runtime + " n_samples = " + str(num_samples) + '\n'

    # Store the samples into a separate file, modular approach
    filename = "covid_chain.smp"
    if (len(sys.argv) == 2):
        filename = str(sys.argv[1]) + "_chain.smp"
    samples_file = open(filename, "w")
    samples_file.write(info_str)
    for x in X:
        for i in range(len(x)):
            if i < (len(x) - 1):
                print(x[i], file = samples_file, end = ' ')
            else:
                print(x[i], file = samples_file, end = '\n')
    samples_file.close()
    print("Single chain samples stored in " + filename)

       
if SAMPLING_TO_CHECK_CONVERGENCE:
    print("Convergence analysis for the Markov Chain")
    start_x = np.array([np.random.uniform(y[-1], MAXVICTIMS),
                    np.random.uniform(0, 2),
                    np.random.uniform(0, 2), y[0]])
    print("Starting point ", start_x)
    X, a_rate = gmcmc.convergenceMetropolis(start_x,
        h_metropolis_array,y, G, sigmas, inDomainG, num_samples, skip_n_samples,
        conv_samples, parallel)

    info_str = "CONVERGENCE OF: chain_rwMetropolis, h = " + \
            str(h_metropolis_array) + \
        " n_samples = " + str(num_samples) + " Average acceptance rate: " + \
        str(a_rate) + "%" + " skip rate: " + str(skip_n_samples) + \
        " #chains for studying convergence: " + \
        str(conv_samples) + "\n"

    # Store the samples into a separate file, to incentivate a chain approach
    filename = "covid_convergence.smp"
    if (len(sys.argv) == 2):
        filename = str(sys.argv[1]) + "_convergence.smp"
    samples_file = open(filename, "w")
    samples_file.write(info_str)
    for x in X:
        for i in range(len(x)):
            if i < (len(x) - 1):
                print(x[i], file = samples_file, end = ' ')
            else:
                print(x[i], file = samples_file, end = '\n')
    samples_file.close()
    print("Convergence information stored in " + filename)
