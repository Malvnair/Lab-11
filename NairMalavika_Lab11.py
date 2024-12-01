import numpy as np
import matplotlib.pyplot as plt


# Import the spectral radius function from NairMalavika_SamoylovaAlona_Lab10
def spectral_radius(A):
    """Calculates the maxium absolute eigenvalue of a 2-D array A

    Args:
        A : 2D array from part 1

    Returns:
        maximum absolute value of eigenvalues
    """    
    
    #determine the eigenvalues, only get that value from the tuple
    eigenvalues, _ = np.linalg.eig(A)
    
    #determine the maxiumum absolute value
    max_value = max(abs(eigenvalues))
    
    return max_value


# Import the make initial condition function from NairMalavika_SamoylovaAlona_Lab10
def make_initialcond(sigma_0=0.2, k_0=35, x_i = None):
    """
    Function to solve advection equation for the time using Lax method and traffic simulation.

    Parameters:
    x_i => The spatial grid with positions
    sigma_0 => The width of the wave packet . 0.2 is used as default value
    k_0 => The average wave number. 35 is used as default value

    Returns: 
        the initial condition a=> a(x,0)

    """
    
    # Ensure that a spatial grid is provided, raise an error if not
    if x_i is None:
        raise ValueError("x_i (spatial grid) must be provided.")
    
    a = (np.exp( (-x_i**2) / (2*sigma_0**2))*np.cos(k_0*x_i))
    return a


# Import the make tridiagonal function from NairMalavika_SamoylovaAlona_Lab10
def make_tridiagonal(N, b, d, a):
    """
    Creates a tridiagonal matrix of size N x N with with d on the diagonal, b one below
    the diagonal, and a, one above.
    
    Parameters:
        N: The size of the matrix (N x N).
        b: The value for the diagonal below the main diagonal.
        d: The value for the main diagonal.
        a : The value for the diagonal above the main diagonal.
    
    Returns:
        A tridiagonal matrix with the given parameters.
    """
    
    #main diagonal
    A = d * np.eye(N)
    #below diagonal
    A += b * np.eye(N, k=-1)
    #above diagonal
    A += a * np.eye(N, k=1)
    
    return A


def advection1d(method, nspace, ntime, tau_rel, params):
    L, c = params
    h = L / nspace  
    tau = tau_rel * h / c  
    x = np.linspace(-L / 2, L / 2, nspace)  
    t = np.linspace(0, ntime * tau, ntime)  
    
    
    # Initial condition
    a = np.zeros((nspace, ntime))
    a[:, 0] = make_initialcond(x_i=x)


    return a, x, t



