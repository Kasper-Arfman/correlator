import numpy as np

def diffusion3D(r, a):
    return 1 / ((1+r)*np.sqrt(1+r/a**2))

def diffusion2D(r):
    return 1 / (1 + r)

def triplet_decay(r, F):
    return 1 + np.exp(-r)* F/(1-F)

def triplet_diffusion(t, T1, F_trip, T_trip, a, n=1):
    return triplet_decay(t/T_trip, F_trip) * diffusion3D(t/T1, a) / n

def flow_decay(r):
    return np.exp(-r)**2


# def dist_fit()
#     return sum(w * )


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Simulate data
    t = np.geomspace(1e-6, 1e2, 400)
    acf = triplet_diffusion(
        t, 
        T1=0.050, 
        F_trip=0.150, 
        T_trip=0.003, 
        a=5.5,
    ) * np.random.normal(1, 0.03, t.size)


    plt.plot(t, acf)
    plt.xscale('log')
    plt.show()