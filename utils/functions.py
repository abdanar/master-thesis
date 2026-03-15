from typing import Callable
import numpy as np

def vectorize(f: Callable) -> Callable:
    def wrapper(x: np.ndarray, t = None):
        x = np.asarray(x)
        if x.ndim not in [1, 2]:
            raise ValueError("Input for space variable must be 1D or 2D array.")
        
        if t is None:
            if x.shape[1] == 1:
                return f(x[0, :], t)
            elif x.shape[1] == 2:
                return f(x[0, :], x[1, :], t)





