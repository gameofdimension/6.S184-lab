import torch
from matplotlib import pyplot as plt

# from lab1.brownian_motion import device
from lab1.lab_one import (BrownianMotion, EulerMaruyamaSimulator, plot_trajectories_1d)


def show1():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sigma = 1.0
    brownian_motion = BrownianMotion(sigma)
    simulator = EulerMaruyamaSimulator(sde=brownian_motion)
    x0 = torch.zeros(5, 1).to(device)  # Initial values - let's start at zero
    ts = torch.linspace(0.0, 5.0, 500).to(device)  # simulation timesteps

    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.set_title(r'Trajectories of Brownian Motion with $\sigma=$' + str(sigma), fontsize=18)
    ax.set_xlabel(r'Time ($t$)', fontsize=18)
    ax.set_ylabel(r'$X_t$', fontsize=18)
    plot_trajectories_1d(x0, simulator, ts, ax)
    plt.show()


if __name__ == '__main__':
    show1()
