import numpy as np
import matplotlib.pyplot as plt
import gmcmc
import sys


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
input("--- press a key to continue --- ")


# To perform Gradient descent, need the Jacobian matrix
def jacobianG(x):
    res = np.zeros([d, m])
    res[0,0] = 1
    res[0,1] = 1
    res[0,2] = 0
    res[1,0] = 1
    res[1,1] = 0
    res[1,2] = 1
    return res

# Steps gradient descent
eta = 0.0001

h_metropolis = 4
num_samples = 10000
skip_n_samples = 10 # With 1, no samples are skipped
parallel = True

conv_samples = 500

STUDY_SINGLE_CHAIN = False
CONVERGENCE_ANALYSIS = True #False #True

if STUDY_SINGLE_CHAIN:
    # Open the file containing the list of samples
    filename = "multiDimG_chain.smp"
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
    for i in range(m):
        plt.subplot(m, 1, i+1)
        plt.hist(X[:,i], 50, density=True)
    plt.suptitle("Complete distribution")
    plt.show()


    # Find the optimal number of clustering
    gmcmc.elbow_search(X, 1, 30)
    ncent = int(input("Enter the number of centroids: "))
#    ncent = 2
    # Store the clusters, which will be candidate modes
    centroids, freq = gmcmc.detailed_clustering (X, ncent, y, G, sigmas)
    freq = freq / 100.

    # Perform gradient descent on each centroid to identify the modes
    print("\nSearch for the modes: ")
    for i in range(ncent):
        print("Gradient descent on candidate mode number", i)
        if(gmcmc.simple_descent(centroids[i], y, G, sigmas, jacobianG, eta)):
            print("MODE FOUND: centroid number ", i)

    for i in range(m):
        plt.subplot(m, 1, i+1)
        plt.scatter(centroids[:,i], freq, marker="*")
    plt.suptitle("Clustered distribution")
    plt.show()


if CONVERGENCE_ANALYSIS:
    print("Reading the results about CONVERGENCE")
    # Open the file containing the list of samples
    filename = "multiDimG_convergence.smp"
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

    # Computing the confidence interval for each marginal
    for i in range(m):
        # Compute the 95% confidence interval
        mean, sigma = gmcmc.mean_variance1d(X[:,i])
        print("Merginal number #", i)
        print("Mean: ", mean, "sigma: ", sigma)
        print("95% Confidence Interval: [", 
                        mean-2.*sigma, " ", mean+2.*sigma, "]")

    # Plot the expectations' distribution
    for i in range(m):
        plt.subplot(m, 1, i+1)
        plt.hist(X[:,i], 50, density=True)
    plt.suptitle("Convergence analysis. Gaussian = WIN")
    plt.show()
