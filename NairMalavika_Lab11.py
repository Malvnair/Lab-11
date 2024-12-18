# Github Link: https://github.com/Malvnair/Lab-11.git

# Imported the necessary libraries
import numpy as np
import matplotlib.pyplot as plt


# Import the spectral radius function from NairMalavika_SamoylovaAlona_Lab10
def spectral_radius(A):
    """Calculates the maxium absolute eigenvalue of a 2-D array A

    Args:
        A : Sqaure 2D array.

    Returns:
        Maximum absolute value of the eigenvalues of A.
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
    x_i: The spatial grid with positions
    sigma_0: The width of the wave packet . 0.2 is used as default value
    k_0: The average wave number. 35 is used as default value

    Returns: 
        The initial condition a=> a(x,0). Will raise a ValueError if the spatial grid not provided.

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
    """Solve the 1D advection equation using matrix multiplication.

    Supports the FTCS and Lax numerical methods. The solution is advanced in time
    using a matrix multiplication approach.

    Args:
        method: Numerical method that will be used ('ftcs' or 'lax', raises error if other).
        nspace: Number of spatial grid points.
        ntime: Number of time steps.
        tau_rel: Time step relative to the critical time step (tau / tau_crit).
        params: List containing the length of the  domain and wave speed.

    Returns:
        A tuple containing:
            - a : 2D array of wave amplitudes, shape (nspace, ntime).
            - x: 1D array of spatial grid points.
            - t: 1D array of time grid points.
    """
    L, c = params

    # Spatial step size
    h = L / nspace  
    # Time step size
    tau = tau_rel * h / c  
    # Spatial grid
    x = np.linspace(-L / 2, L / 2, nspace)  
    # Time grid
    t = np.linspace(0, ntime * tau, ntime)  
    
    
    # Initial condition
    a = np.zeros((nspace, ntime))
    a[:, 0] = make_initialcond(x_i=x)



    

    # Construct the matrix A
    if method == 'ftcs':
        alpha = c * tau / (2 * h)
        A = make_tridiagonal(nspace, b=alpha, d=1.0, a=-alpha)
    elif method == 'lax':
        alpha = c * tau / (2 * h)
        # Account for averaging (0.5) and advection (±alpha)
        A = make_tridiagonal(nspace, b=0.5 + alpha, d=0.0, a=0.5 - alpha)
    else:
        raise ValueError("Invalid method. Choose 'ftcs' or 'lax'.")

    # Apply periodic boundary conditions
    if method == 'ftcs':
        # Last point affects the first
        A[0, -1] = alpha
        # First point affects the last
        A[-1, 0] = -alpha
    elif method == 'lax':
        # Last point affects the first
        A[0, -1] = 0.5 + alpha
        # First point affects the last
        A[-1, 0] = 0.5 - alpha
        
        
    # Check stability for FTCS
    if method == 'ftcs':
        spectral_radius_A = spectral_radius(A)
        if spectral_radius_A > 1:
            print("FTCS method is unstable.")   
            
            
    # Compute the wave at each time step
    for current_time in range(ntime - 1):
    # Compute the next time step using matrix multiplication
        next_state = np.dot(A, a[:, current_time])
    # Update the array with the new state
        a[:, current_time + 1] = next_state

    return a, x, t



# Import the plotting code from wave_plot.py
def plot_wave(a, x, t):
    plotskip = 50
    fig, ax = plt.subplots()
    yoffset = a[:, 0].max() - a[:, 0].min()
    for i in np.arange(len(t)-1, -1, -plotskip):
        ax.plot(x, a[:, i] + yoffset * i / plotskip, label=f't ={t[i]:.3f}')
    ax.legend()
    ax.set_xlabel('X position')
    ax.set_ylabel('a(x,t) [offset]')
    plt.title('Wave Propagation Visualization')
    plt.savefig('NairMalavika_Lab11_Fig1.png')
    plt.show()


# Parameters
L = 5
c = 1
nspace = 300
ntime = 500
tau_rel = 1.0

# call function with Lax method
a_lax, x_lax, t_lax = advection1d('lax', nspace, ntime, tau_rel, [L, c])


# plot the wave propagation
plot_wave(a_lax, x_lax, t_lax)