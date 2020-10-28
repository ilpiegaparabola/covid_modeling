import numpy as np
import matplotlib.pyplot as plt
import gmcmc
import sys
from numpy import exp, log

np.random.seed(0)
# This is the number of observation days
global_T = 21
m = 4
d = global_T


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
    if x[0] < 500 or x[0] > 50000:
        return False
    if x[1] < 0 or x[1] > 2:
        return False
    if x[2] < 0 or x[2] > 2:
        return False
    if x[3] < 100 or x[3] > 5000:
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


h_metropolis_array = np.array([10, 0.01, 0.01, 5])
num_samples = 5000
skip_n_samples = 5 # With 1, no samples are skipped
parallel = True
conv_samples = 500

### ------------------------ end of the common section with write -----#

STUDY_SINGLE_CHAIN = True #False
CONVERGENCE_ANALYSIS = True #False #True #False #True

if STUDY_SINGLE_CHAIN:
    # Open the file containing the list of samples
    filename = "covid_chain.smp"
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
#    gmcmc.elbow_search(X, 1, 30)
    ncent = int(input("Enter the number of centroids: "))
#    ncent = 2
    # Store the clusters, which will be candidate modes
    centroids, freq = gmcmc.detailed_clustering (X, ncent, y, G, sigmas)
    freq = freq / 100.

    # Perform gradient descent on each centroid to identify the modes
#    print("\nSearch for the modes: ")
#    eta = 0.5
#    for i in range(ncent):
#        print("Gradient descent on candidate mode number", i)
#        if(gmcmc.simple_descent(centroids[i], y, G, sigmas, jacobianG, eta)):
#            print("MODE FOUND: centroid number ", i)

    for i in range(m):
        plt.subplot(m, 1, i+1)
        plt.scatter(centroids[:,i], freq, marker="*")
    plt.suptitle("Clustered distribution")
    plt.show()


if CONVERGENCE_ANALYSIS:
    print("Reading the results about CONVERGENCE")
    # Open the file containing the list of samples
    filename = "covid_convergence.smp"
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
