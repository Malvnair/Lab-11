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
    
    # Check stability for FTCS
    if method == 'ftcs':
        spectral_radius_A = spectral_radius(A)
        if spectral_radius_A > 1:
            print("FTCS method is unstable.")


    
    # Construct the matrix A
    if method == 'ftcs':
        alpha = c * tau / (2 * h)
        A = make_tridiagonal(nspace, b=alpha, d=1.0, a=-alpha)
    elif method == 'lax':
        alpha = c * tau / (2 * h)
        A = make_tridiagonal(nspace, b=0.5 + alpha, d=0.0, a=0.5 - alpha)
    else:
        raise ValueError("Invalid method. Choose 'ftcs' or 'lax'.")


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