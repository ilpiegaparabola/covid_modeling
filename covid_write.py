# Modeling the Coronavirus fatal rate by using a generalized logistic map 
import numpy as np
import sys
import matplotlib.pyplot as plt
import gmcmc
from numpy import exp

np.random.seed(0)

# This is the number of observation days
global_T = 21

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

def jacobianG(x):
    print("ERROR: JACOBIAN STILL TO IMPLEMENT")
    return 0


# Checking now the domain of G is taylored on this problem of dimension 4
def inDomainG(x):
    if x[0] < 100 or x[0] > 5000:
        return False
    if x[1] < 0 or x[1] > 2:
        return False
    if x[2] < 0 or x[2] > 2:
        return False
    if x[3] < 100 or x[3] > 5000:
        return False
    return True

# m = 3, d = 2
m = 4
d = global_T
true_x = np.array([3000., 0.2, 0.5, 200.])
sigmas = np.ones(global_T) * 0.5

cov_matrix = np.identity(d)
cov_matrix = np.array([cov_matrix[i] * sigmas[i] for i in range(d)])
y = G(true_x) + np.random.multivariate_normal(np.zeros(d), cov_matrix)

print("True x: ", true_x)
print("y = ", y)
#input("-- press a key to continue---")
#quit()

h_metropolis_array = np.array([5, 0.001, 0.005, 5])
num_samples = 50000
skip_n_samples = 5 # With 1, no samples are skipped
parallel = True

conv_samples = 1000

SAMPLING_SINGLE_CHAIN = True #False
SAMPLING_TO_CHECK_CONVERGENCE = True #alse #True


if SAMPLING_SINGLE_CHAIN:
    print("Constructing a single full chain")
    X, runtime, _ , _ = gmcmc.chain_rwMetropolis(np.array([1000, 1, 1, 100]),
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
    X, a_rate = gmcmc.convergenceMetropolis(np.array([1000, 1, 1, 100]),
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
