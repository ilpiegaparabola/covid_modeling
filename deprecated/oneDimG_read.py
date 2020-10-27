import numpy as np
import matplotlib.pyplot as plt
import gmcmc
import sys


def G(x):
    return x**2

def jacobianG(x):
    return np.array([[2. * x]])

def inDomainG(x):
    for i in x:
        if i < -500 or i > 500:
            return False
    return True



y = G(np.array([4]))
sigmas = [0.5]

# Steps gradient descent
eta = 0.0001

h_metropolis = 8
num_samples = 10000
skip_n_samples = 10 # With 1, no samples are skipped
parallel = True

conv_samples = 500

STUDY_SINGLE_CHAIN = True #False
CONVERGENCE_ANALYSIS = True#True

if STUDY_SINGLE_CHAIN:
    # Open the file containing the list of samples
    filename = "oneDimG_chain.smp"
    if (len(sys.argv) == 2):
        filename = str(sys.argv[1])
    print("Loading:", filename)
    samples_file = open(filename, "r")

    # Its first line contains information about the simulation
    info_str = samples_file.readline()
    print(info_str[:-1])

    # Collect all the samples into X
    X = []
    for x in samples_file:
        X.append(np.array(x[0:-1].split(' ')).astype("float64"))
    X = np.asanyarray(X)
    samples_file.close()
    print("Read", len(X), "samples of dimension", len(X[0]))

    # Plot the sampling empirical distribution
    d = len(X[0])
    for i in range(d):
        plt.subplot(d, 1, i+1)
        plt.hist(X[:,i], 50, density=True)
    plt.suptitle("Complete distribution")
    plt.show()


    # Find the optimal number of clustering
#    gmcmc.elbow_search(X, 1, 30)
#    ncent = int(input("Enter the number of centroids: "))
    ncent = 2
    # Store the clusters, which will be candidate modes
    centroids, freq = gmcmc.detailed_clustering (X, ncent, y, G, sigmas)
    freq = freq / 100.

    # Perform gradient descent on each centroid to identify the modes
    print("\nSearch for the modes: ")
    for i in range(ncent):
        print("Gradient descent on candidate mode number", i)
        if(gmcmc.simple_descent(centroids[i], y, G, sigmas, jacobianG, eta)):
            print("MODE FOUND: centroid number ", i)

    for i in range(d):
        plt.subplot(d, 1, i+1)
        plt.scatter(centroids[:,i], freq, marker="*")
    plt.suptitle("Clustered distribution")
    plt.show()


if CONVERGENCE_ANALYSIS:
    print("Reading the results about CONVERGENCE")
    # Open the file containing the list of samples
    filename = "oneDimG_convergence.smp"
    if (len(sys.argv) == 2):
        filename = str(sys.argv[1])
    print("Loading:", filename)
    samples_file = open(filename, "r")

    # Its first line contains information about the simulation
    info_str = samples_file.readline()
    print(info_str[:-1])

    # Collect all the samples into X
    X = []
    for x in samples_file:
        X.append(np.array(x[0:-1].split(' ')).astype("float64"))
    X = np.asanyarray(X)
    samples_file.close()
    print("Read", len(X), "samples of dimension", len(X[0]))

    # Compute the 95% confidence interval
    mean, sigma = gmcmc.mean_variance1d(X)
    print("Mean: ", mean, "sigma: ", sigma)
    print("95% Confidence Interval: [", mean-2.*sigma, " ", mean+2.*sigma, "]")

    # Plot the sampling distribution
    d = len(X[0])
    for i in range(d):
        plt.subplot(d, 1, i+1)
        plt.hist(X[:,i], 50, density=True)
    plt.suptitle("Convergence analysis. Gaussian = WIN")
    plt.show()
