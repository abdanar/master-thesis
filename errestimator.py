import numpy as np

class ErrorEstimator:
    def __init__(self, femspace, u_h, u_exact=None):
        self.femspace = femspace
        self.u_h = u_h
        self.u_exact = u_exact