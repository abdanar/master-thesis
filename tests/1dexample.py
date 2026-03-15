import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import pyvista as pv

def decompose(sdomain: tuple, nx: int, overlap: int, nsubdomains: int):
    
    """
    Divide np.linspace(a, b, nx) into nsubdomains such that consecutive subdomains share exactly `overlap` + 1 points.
    """
    
    x = np.linspace(sdomain[0], sdomain[1], nx)
    
    # Base size of each non-overlapping part
    base = nx//nsubdomains  # number of intervals
    remainder = nx%nsubdomains
    
    # Determine number of local points for non-overlapping part
    sizes = []
    for i in range(nsubdomains):
        sz = base + (1 if i < remainder else 0)
        sizes.append(sz)
 
    subdomains = []
    start = 0
    for j, subsize in enumerate(sizes):
        if j == 0:
            subdomains.append(x[0: subsize + overlap])
        elif j == nsubdomains - 1:
            subdomains.append(x[-(subsize + overlap):])
        else:
            subdomains.append(x[start - overlap: start + subsize + overlap])
        
        start += subsize
    
    return subdomains

def poisson(lcond, rcond, icond, func, tdomain: tuple, sdomain: tuple, nx: int, nt: int):
    
    """
    Solve the one-dimensional heat equation

        u_t = u_xx + f(x, t),       x in (a, b),  t in (0, T)
        u(a, t) = l(t),             t in (0, T)
        u(b, t) = r(t),             t in (0, T)
        u(x, 0) = i(x),             x in (a, b)

    The method uses centered finite differences in space and the Backward Euler
    scheme in time.

    Parameters
    ----------
    lcond : callable
        Left boundary condition l(t).
    rcond : callable
        Right boundary condition r(t).
    icond : callable
        Initial condition i(x).
    func : callable
        Source term f(x, t).
    tdomain : tuple
        Time interval (0, T).
    sdomain : tuple
        Spatial interval (a, b).
    nx : int
        Number of spatial nodes.
    nt : int
        Number of time steps.

    Returns
    -------
    S : ndarray of shape (nx, nt)
        Numerical solution. Here S[i, j] approximates u(x_i, t_j).
    """

    # Grid
    xnodes = np.linspace(sdomain[0], sdomain[1], nx)
    tnodes = np.linspace(tdomain[0], tdomain[1], nt)
    deltax = (sdomain[1] - sdomain[0])/(nx - 1)
    deltat = (tdomain[1] - tdomain[0])/(nt - 1)
    ratio = deltat/deltax**2

    A = sc.sparse.diags_array([1, -2, 1], offsets=[-1, 0, 1], shape=(nx - 2, nx - 2))
    R = sc.sparse.eye_array(nx - 2) - ratio*A

    # Solution matrix S[i,j] = u(x_i, t_j)
    S = np.zeros((nx, nt))

    # Initial condition
    S[:, 0] = icond(xnodes)

    # Boundary values for all time steps

    if not callable(lcond):
        lvals = lcond
    else:
        lvals = lcond(tnodes)

    if not callable(rcond):
        rvals = rcond
    else:
        rvals = rcond(tnodes)

    S[0, :] = lvals
    S[-1, :] = rvals
    
    # Time stepping
    for step, tval in enumerate(tnodes[1:]):
        L = S[1:-1, step] + deltat*func(xnodes[1:-1], tval)  
        L[0] += ratio * S[0, step + 1]
        L[-1] += ratio * S[-1, step + 1]
        S[1:-1, step + 1] = sc.sparse.linalg.spsolve(R, L)
    return S

def domaindecomposition(lcond, rcond, icond, func, tdomain: tuple, sdomain: tuple, nx: int, nt: int, overlap: int, nsubdomains: int, maxiter: int = 200, tol: float = 1e-3):

    # Numerical solution on the whole domain
    # S_whole = poisson(lcond, rcond, icond, func, tdomain, sdomain, nx, nt)

    # Overlapping Schwarz Waveform Relaxation method
    solution = {}
    subdomains = decompose(sdomain, nx, overlap, nsubdomains)
    for i, subdomain in enumerate(subdomains):
        solution[i] = np.zeros((len(subdomain), nt))
    print(f'The Overlapping Schwarz Waveform Relaxation method for delta = {overlap*(sdomain[1] - sdomain[0])/(nx - 1)}.')
    for iter in range(maxiter):
        new_solution = {}
        for i, subdomain in enumerate(subdomains):
            if i == 0:
                new_solution[i] = poisson(lcond = lcond, rcond = solution[1][overlap, :], icond = icond, func = func, tdomain = tdomain, sdomain = (subdomain[0], subdomain[-1]), nx = len(subdomain), nt = nt)
            elif i == nsubdomains - 1:
                new_solution[i] = poisson(lcond = solution[nsubdomains-2][-overlap-1, :], rcond = rcond, icond = icond, func = func, tdomain = tdomain, sdomain = (subdomain[0], subdomain[-1]), nx = len(subdomain), nt = nt)
            else:
                new_solution[i] = poisson(lcond = solution[i-1][-overlap-1, :], rcond = solution[i+1][overlap, :], icond = icond, func = func, tdomain = tdomain, sdomain = (subdomain[0], subdomain[-1]), nx = len(subdomain), nt = nt)

        error = np.linalg.norm(solution[1] - new_solution[1], 2)

        print(f'The L2 error in the iteration {iter + 1}: {error:.6f}.')
        if error <= tol:
            break
        else:
            solution = new_solution
    
    # Form the global solution by assembling the local solutions.
    xindex = 0
    S_wave = np.zeros((nx, nt))
    for i, subdomain in enumerate(subdomains):
        if i == 0:
            S_wave[:len(subdomain), :] = solution[i]
            xindex += len(subdomain) - overlap - 1
        elif i == nsubdomains - 1:
            S_wave[-len(subdomain):, :] = solution[i]
        else:
            S_wave[xindex: xindex + len(subdomain), :] = solution[i]
            xindex += len(subdomain)
    return S_wave

def visualize(U, tdomain: tuple, sdomain: tuple, nx: int, nt: int):

    x = np.linspace(sdomain[0], sdomain[1], nx)
    t = np.linspace(tdomain[0], tdomain[1], nt)
    X, T = np.meshgrid(x, t, indexing='ij')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, T, U, cmap='viridis')

    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u(x,t)')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    plt.show()

def export(tdomain: tuple, sdomain: tuple, nx: int, nt: int, U_dd, filename, show3D = False):

    # Create structured grid
    x = np.linspace(sdomain[0], sdomain[1], nx)
    t = np.linspace(tdomain[0], tdomain[1], nt)
    X, T = np.meshgrid(x, t, indexing="ij")  
    U_exact = exact(X, T)
    error = U_exact - U_dd

    # Use height for 3D representation if desired
    Z = np.zeros_like(X) if not show3D else U_dd

    # Create PyVista structured grid
    grid = pv.StructuredGrid(X, T, Z)
    grid["U_dd"] = U_dd.flatten(order="F")
    grid["U_exact"] = U_exact.flatten(order="F")
    grid["Error"] = error.flatten(order="F")
    grid.save(filename)

    print(f"Solution exported to: {filename}")
    print(f"L∞ error: {np.max(np.abs(error)):.6e}")
    print(f"L2 error:  {np.sqrt(np.mean(error**2)):.6e}")
    print(f"L2 for the solution:  {np.sqrt(np.mean(U_numerics)):.6e}")
    print(f"L2 norm of the error: {np.linalg.norm(error, ord=2):.6e}")

# One dimensional example (see page 788)

tdomain = (0, 3)
sdomain = (0, 1)

nx = 101
nt = 301

def exact(x, t):
    return 1 + x + t*np.sin(np.pi*x)

# Source function
def func(x, t):
    return (1 + np.pi**2 * t) * np.sin(np.pi*x)

# Boundary conditions
def lcond(t):
    return exact(sdomain[0], t)

def rcond(t):
    return exact(sdomain[1], t)

# Initial condition
def icond(x):
    return exact(x, tdomain[0])
    
# Numerical solution on the whole domain
U_numerics = poisson(lcond, rcond, icond, func, tdomain, sdomain, nx, nt)
visualize(U_numerics, tdomain, sdomain, nx, nt)

# Solve the problem using Overlapping Schwarz Waveform Relaxation method
U_dd = domaindecomposition(lcond, rcond, icond, func, tdomain, sdomain, nx, nt, overlap = 1, nsubdomains = 2, maxiter = 100, tol = 1e-3)
visualize(U_dd, tdomain, sdomain, nx, nt)

export(tdomain, sdomain, nx, nt, U_dd, filename = "ddpoisson.vts")