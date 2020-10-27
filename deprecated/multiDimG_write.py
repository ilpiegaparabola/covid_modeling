import numpy as np
import sys
import matplotlib.pyplot as plt
import gmcmc

np.random.seed(0)

# G: R^3 -> R^2
def G(x):
    return x[0] + x[1], x[0] + x[2]


def inDomainG(x):
    for i in x:
        if i < -50 or i > 50:
            return False
    return True

# m = 3, d = 2
m = 3
d = 2
true_x = np.array([1., 3., 5.])
sigmas = [0.3, 0.3]

cov_matrix = np.identity(d)
cov_matrix = np.array([cov_matrix[i] * sigmas[i] for i in range(d)])
y = G(true_x) + np.random.multivariate_normal(np.zeros(d), cov_matrix)

print("y = ", y)
input("-- press a key to continue---")

h_metropolis = 4
num_samples = 5000
skip_n_samples = 5 # With 1, no samples are skipped
parallel = True

conv_samples = 500

SAMPLING_SINGLE_CHAIN = False #True #False
SAMPLING_TO_CHECK_CONVERGENCE = True


if SAMPLING_SINGLE_CHAIN:
    print("Constructing a single full chain")
    X, runtime, _ , _ = gmcmc.chain_rwMetropolis(np.zeros(m),
       h_metropolis, y, G, sigmas, inDomainG, num_samples, skip_n_samples)

    info_str = "INFOSIMU: chain_rwMetropolis, h = " + str(h_metropolis) + \
            " runtime: " + runtime + " n_samples = " + str(num_samples) + '\n'

    # Store the samples into a separate file, modular approach
    filename = "multiDimG_chain.smp"
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
    X, a_rate = gmcmc.convergenceMetropolis(np.zeros(m),
        h_metropolis, y, G, sigmas, inDomainG, num_samples, skip_n_samples,
        conv_samples, parallel)

    info_str = "CONVERGENCE OF: chain_rwMetropolis, h = " + \
            str(h_metropolis) + \
        " n_samples = " + str(num_samples) + " Average acceptance rate: " + \
        str(a_rate) + "%" + " skip rate: " + str(skip_n_samples) + \
        " #chains for studying convergence: " + \
        str(conv_samples) + "\n"

    # Store the samples into a separate file, to incentivate a chain approach
    filename = "multiDimG_convergence.smp"
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
