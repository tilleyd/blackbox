import numpy as np

def modified_schwefel(x):
    nx = len(x)
    sm = 0.0
    for i in range(0, nx):
        z = x[i] + 420.9687462275036
        if z < -500:
            zm = (abs(z) % 500) - 500
            t = z + 500
            t = t*t
            sm += zm * np.sin(np.sqrt(abs(zm))) - t / (10000*nx)
        elif z > 500:
            zm = 500 - (z % 500)
            t = z - 500
            t = t*t
            sm += zm * np.sin(np.sqrt(abs(zm))) - t / (10000*nx)
        else:
            sm += z * np.sin(np.sqrt(abs(z)))

    return 418.9829*nx - sm

from blackbox.testbed import Context

bb_contexts = [
    Context(modified_schwefel, 2, -100, 100)
]
