import numpy as np
from scipy.optimize import curve_fit

def gaussian(x,a,b,c):
    return a*p.exp(-(x-b)**2/(2*c**2))

def fit_gaussian(counts):
    x = np.arange(len(counts))
    return curve_fit(gaussian,x,counts)



