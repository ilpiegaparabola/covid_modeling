import numpy as np
import sys
import matplotlib.pyplot as plt
import gmcmc

def G(x):
    return x**2.

def inDomainG(x):
    for i in x:
        if i < -500 or i > 500:
            return False
    return True

true_x = 4
y = G(np.array([true_x]))
sigmas = [0.5]

h_metropolis = 8
num_samples = 10000
skip_n_samples = 10 # With 1, no samples are skipped
parallel = True

conv_samples = 500

SAMPLING_SINGLE_CHAIN = True #False
CONVERGENCE_ANALYSIS = True


if SAMPLING_SINGLE_CHAIN:
    print("Constructing a single full chain")
    X, runtime, _ , _ = gmcmc.chain_rwMetropolis(np.array([0]),
       h_metropolis, y, G, sigmas, inDomainG, num_samples, skip_n_samples)

    info_str = "INFOSIMU: chain_rwMetropolis, h = " + str(h_metropolis) + \
            " runtime: " + runtime + " n_samples = " + str(num_samples) + '\n'

    # Store the samples into a separate file, modular approach
    filename = "oneDimG_chain.smp"
    if (len(sys.argv) == 2):
        filename = str(sys.argv[1]) + "_chain.smp"
    samples_file = open(filename, "w")
    samples_file.write(info_str)
    for x in X:
        print(x[0], file = samples_file)
    samples_file.close()
    print("Single chain samples stored in " + filename)

       
if CONVERGENCE_ANALYSIS:
    print("Convergence analysis for the Markov Chain")
    X, a_rate = gmcmc.convergenceMetropolis(np.array([0]),
        h_metropolis, y, G, sigmas, inDomainG, num_samples, skip_n_samples,
        conv_samples, parallel)

    info_str = "CONVERGENCE OF: chain_rwMetropolis, h = " + \
            str(h_metropolis) + \
        " n_samples = " + str(num_samples) + " Average acceptance rate: " + \
        str(a_rate) + "%" + " skip rate: " + str(skip_n_samples) + \
        " #chains for studying convergence: " + \
        str(conv_samples) + "\n"

    # Store the samples into a separate file, to incentivate a chain approach
    filename = "oneDimG_convergence.smp"
    if (len(sys.argv) == 2):
        filename = str(sys.argv[1]) + "_convergence.smp"
    samples_file = open(filename, "w")
    samples_file.write(info_str)
    for x in X:
        print(x[0], file = samples_file)
    samples_file.close()
    print("Convergence information stored in " + filename)
