import numpy as np

def rk4( f, x0, t ):

    nx = len(x0)
    n = len( t )
    x = np.array( [ x0 ] * n )
    k1 = np.zeros(nx, dtype = float) 
    k2 = np.zeros(nx, dtype = float) 
    k3 = np.zeros(nx, dtype = float) 
    k4 = np.zeros(nx, dtype = float) 
    
    h = t[1] - t[0]
    
    for i in range( n - 1 ): 
        
        k1 = h * f( x[i,:], t[i] )
        k2 = h * f( x[i,:] + 0.5 * k1, t[i] + 0.5 * h )
        k3 = h * f( x[i,:] + 0.5 * k2, t[i] + 0.5 * h )
        k4 = h * f( x[i,:] + k3, t[i+1] )
        
        for j in range(nx):
            x[i+1,j] = x[i,j] + ( k1[j] + 2.0 * ( k2[j]  + k3[j]  ) + k4[j]  ) / 6.0

    return x
