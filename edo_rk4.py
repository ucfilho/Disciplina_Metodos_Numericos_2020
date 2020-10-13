import numpy as np

def rk4( f, x0, t ):
    """Fourth-order Runge-Kutta method to solve x' = f(x,t) with x(t[0]) = x0.
    USAGE:
        x = rk4(f, x0, t)
    INPUT:
        f     - function of x and t equal to dx/dt.  x may be multivalued,
                in which case it should a list or a NumPy array.  In this
                case f must return a NumPy array with the same dimension
                as x.
        x0    - the initial condition(s).  Specifies the value of x when
                t = t[0].  Can be either a scalar or a list or NumPy array
                if a system of equations is being solved.
        t     - list or NumPy array of t values to compute solution at.
                t[0] is the the initial condition point, and the difference
                h=t[i+1]-t[i] determines the step size h.
    OUTPUT:
        x     - NumPy array containing solution values corresponding to each
                entry in t array.  If a system is being solved, x will be
                an array of arrays.
    """
    nx = len(x0)
    n = len( t )
    x = np.array( [ x0 ] * n )
    k1 = np.zeros(nx, dtype = float) 
    k2 = np.zeros(nx, dtype = float) 
    k3 = np.zeros(nx, dtype = float) 
    k4 = np.zeros(nx, dtype = float) 
    
  
    
    h = t[1] - t[0]
    
    for i in range( n - 1 ):
        for j in range(nx):
            k1[j] = h * f( x[i,:], t[i] )[j]
            k2[j] = h * f( x[i,:] + 0.5 * k1[j], t[i] + 0.5 * h )[j]
            k3[j] = h * f( x[i,:] + 0.5 * k2[j], t[i] + 0.5 * h )[j]
            k4[j] = h * f( x[i,:] + k3[j], t[i+1] )[j]
            print('======');print(k1)
            x[i+1,j] = x[i,j] + ( k1[j] + 2.0 * ( k2[j]  + k3[j]  ) + k4[j]  ) / 6.0

    return x
