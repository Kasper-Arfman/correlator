import numpy as np
import pandas as pd 
from scipy.spatial.distance import cdist
from math import acos, degrees

def cos_law_angle(a, b, c):
    """
    Given a triangle with sides a, b, c, find the angle of corner C opposite to c

    a2 + b2 = c2 + 2ab*cos(C)      (Law of Cosines)
    """
    cos_C = (a*a + b*b - c*c) / (2*a*b)
    return acos(cos_C)  # [rad]

def find_Lcurve_corner(residualsNorm,contributionsNorm):
    """
    Use the triangle method to find the corner of the L-curve

    If you use this function please cite the original manuscript from which I took the idea: 
                    Castellanos, J. Longina, Susana Gómez, and Valia Guerra. 
                    "The triangle method for finding the corner of the L-curve." 
                    Applied Numerical Mathematics 43.4 (2002): 359-373.

    Input - the norm vector of the residuals and the norm vector of the contributions
            i.e., the norm of the fidelity term and the norm of the penalty term

    Returns the position of the corner of the curve log(contributionsNorm) vs log(residualsNorm) 
    """

    # Convert to log
    x = np.log(residualsNorm)
    y = np.log(contributionsNorm)

    # Normalise to avoid floating point errors - This doesn't change the shape of the curve
    x = (x - np.min(x)) / (np.max(x)-np.min(x)) * 100
    y = (y - np.min(y)) / (np.max(y)-np.min(y)) * 100

    n = len(x)
    angles = []
    indices = []
    A = (x[n-1], y[n-1]) # Point A of a triangle
    for i in range(n-4):
        B = (x[i], y[i])  # Point B of a triangle
        c = cdist([A], [B])[0]

        for j in range(i+2, n-2):
            C  = (x[j], y[j])  # Point C of a triangle
            a = cdist([B], [C])[0]
            b = cdist([A], [C])[0]

            area = (B[0] - C[0])*(C[1]-A[1]) - (C[0] - A[0])*(B[1]-C[1])
            
            angle = cos_law_angle(a, b, c)
            
            if degrees(angle) < 160 and 0 < area:
                indices.append(j)
                angles.append(angle)
       
    return indices[np.argmin(angles)] 

def parse_csv(df: pd.DataFrame, i=0):
    """Extract x, y, and y_names as arrays
    Expects x to be the first column

    optionally: exclude the first <i> timepoints
    """
    y = df.sort_values(by=df.columns[0], ascending=True)
    x = y.pop(y.columns[0])
    x, y, y_names = map(np.array, (x, y, y.columns))
    return x[i:], y[i:], y_names