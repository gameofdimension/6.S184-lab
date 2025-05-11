
import seaborn as sns
import torch
from celluloid import Camera
from IPython.display import HTML
from matplotlib import pyplot as plt

# from lab1.brownian_motion import device
from lab_one import (SDE, Density, Sampleable, Simulator,
                     imshow_density)

# """### Question 3.1: Implementing Langevin Dynamics

# In this section, we'll simulate the (overdamped) Langevin dynamics $$dX_t = \frac{1}{2} \sigma^2\nabla \log p(X_t) dt + \sigma dW_t.$$

# **Your job**: Fill in the `drift_coefficient` and `diffusion_coefficient` methods of the class `LangevinSDE` below.

# **Hint**: Use `Density.score` to access the score.
# """


class LangevinSDE(SDE):
    def __init__(self, sigma: float, density: Density):
        self.sigma = sigma
        self.density = density

    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Returns the drift coefficient of the ODE.
        Args:
            - xt: state at time t, shape (bs, dim)
            - t: time, shape ()
        Returns:
            - drift: shape (bs, dim)
        """
        # raise NotImplementedError("Fill me in for Question 3.1!")
        return 1 / 2 * self.sigma**2 * self.density.score(xt)

    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Returns the diffusion coefficient of the ODE.
        Args:
            - xt: state at time t, shape (bs, dim)
            - t: time, shape ()
        Returns:
            - diffusion: shape (bs, dim)
        """
        # raise NotImplementedError("Fill me in for Question 3.1!")
        return self.sigma


"""Now, let's graph the results!"""

# First, let's define two utility functions...


def every_nth_index(num_timesteps: int, n: int) -> torch.Tensor:
    """
    Compute the indices to record in the trajectory
    """
    if n == 1:
        return torch.arange(num_timesteps)
    return torch.cat(
        [
            torch.arange(0, num_timesteps - 1, n),
            torch.tensor([num_timesteps - 1]),
        ]
    )


def graph_dynamics(
    num_samples: int,
    source_distribution: Sampleable,
    simulator: Simulator,
    density: Density,
    timesteps: torch.Tensor,
    plot_every: int,
    bins: int,
    scale: float
):
    """
    Plot the evolution of samples from source under the simulation scheme given by simulator (itself a discretization of an ODE or SDE).
    Args:
        - num_samples: the number of samples to simulate
        - source_distribution: distribution from which we draw initial samples at t=0
        - simulator: the discertized simulation scheme used to simulate the dynamics
        - density: the target density
        - timesteps: the timesteps used by the simulator
        - plot_every: number of timesteps between consecutive plots
        - bins: number of bins for imshow
        - scale: scale for imshow
    """
    # Simulate
    x0 = source_distribution.sample(num_samples)
    xts = simulator.simulate_with_trajectory(x0, timesteps)
    indices_to_plot = every_nth_index(len(timesteps), plot_every)
    plot_timesteps = timesteps[indices_to_plot]
    plot_xts = xts[:, indices_to_plot]

    # Graph
    fig, axes = plt.subplots(2, len(plot_timesteps), figsize=(8 * len(plot_timesteps), 16))
    axes = axes.reshape((2, len(plot_timesteps)))
    for t_idx in range(len(plot_timesteps)):
        t = plot_timesteps[t_idx].item()
        xt = xts[:, t_idx]
        # Scatter axes
        scatter_ax = axes[0, t_idx]
        imshow_density(density, bins, scale, scatter_ax, vmin=-15, alpha=0.25, cmap=plt.get_cmap('Blues'))
        scatter_ax.scatter(xt[:, 0].cpu(), xt[:, 1].cpu(), marker='x', color='black', alpha=0.75, s=15)
        scatter_ax.set_title(f'Samples at t={t:.1f}', fontsize=15)
        scatter_ax.set_xticks([])
        scatter_ax.set_yticks([])

        # Kdeplot axes
        kdeplot_ax = axes[1, t_idx]
        imshow_density(density, bins, scale, kdeplot_ax, vmin=-15, alpha=0.5, cmap=plt.get_cmap('Blues'))
        sns.kdeplot(x=xt[:, 0].cpu(), y=xt[:, 1].cpu(), alpha=0.5, ax=kdeplot_ax, color='grey')
        kdeplot_ax.set_title(f'Density of Samples at t={t:.1f}', fontsize=15)
        kdeplot_ax.set_xticks([])
        kdeplot_ax.set_yticks([])
        kdeplot_ax.set_xlabel("")
        kdeplot_ax.set_ylabel("")

    plt.show()


# # Construct the simulator
# target = GaussianMixture.random_2D(nmodes=5, std=0.75, scale=15.0, seed=3.0).to(device)
# sde = LangevinSDE(sigma=10.0, density=target)
# simulator = EulerMaruyamaSimulator(sde)

# # Graph the results!
# graph_dynamics(
#     num_samples=1000,
#     source_distribution=Gaussian(mean=torch.zeros(2), cov=20 * torch.eye(2)).to(device),
#     simulator=simulator,
#     density=target,
#     timesteps=torch.linspace(0, 5.0, 1000).to(device),
#     plot_every=334,
#     bins=200,
#     scale=15
# )

# """**Your job**: Try varying the value of $\sigma$, the number and range of the simulation steps, the source distribution, and target density. What do you notice? Why?

# **Your answer**:

# Note: To run the folowing two **optional** cells, you will need to download the `ffmpeg` library. You can do so using e.g., `conda install -c conda-forge ffmpeg` (or, ideally, `mamba`). Running `pip install ffmpeg` or similar will likely **not** work.
# """

# Commented out IPython magic to ensure Python compatibility.
# %pip install ffmpeg celluloid


def animate_dynamics(
    num_samples: int,
    source_distribution: Sampleable,
    simulator: Simulator,
    density: Density,
    timesteps: torch.Tensor,
    animate_every: int,
    bins: int,
    scale: float,
    save_path: str = 'dynamics_animation.mp4'
):
    """
    Plot the evolution of samples from source under the simulation scheme given by simulator (itself a discretization of an ODE or SDE).
    Args:
        - num_samples: the number of samples to simulate
        - source_distribution: distribution from which we draw initial samples at t=0
        - simulator: the discertized simulation scheme used to simulate the dynamics
        - density: the target density
        - timesteps: the timesteps used by the simulator
        - animate_every: number of timesteps between consecutive frames in the resulting animation
    """
    # Simulate
    x0 = source_distribution.sample(num_samples)
    xts = simulator.simulate_with_trajectory(x0, timesteps)
    indices_to_animate = every_nth_index(len(timesteps), animate_every)
    animate_timesteps = timesteps[indices_to_animate]

    # Graph
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    camera = Camera(fig)
    for t_idx in range(len(animate_timesteps)):
        t = animate_timesteps[t_idx].item()
        xt = xts[:, t_idx]
        # Scatter axes
        scatter_ax = axes[0]
        imshow_density(density, bins, scale, scatter_ax, vmin=-15, alpha=0.25, cmap=plt.get_cmap('Blues'))
        scatter_ax.scatter(xt[:, 0].cpu(), xt[:, 1].cpu(), marker='x', color='black', alpha=0.75, s=15)
        scatter_ax.set_title('Samples')

        # Kdeplot axes
        kdeplot_ax = axes[1]
        imshow_density(density, bins, scale, kdeplot_ax, vmin=-15, alpha=0.5, cmap=plt.get_cmap('Blues'))
        sns.kdeplot(x=xt[:, 0].cpu(), y=xt[:, 1].cpu(), alpha=0.5, ax=kdeplot_ax, color='grey')
        kdeplot_ax.set_title('Density of Samples', fontsize=15)
        kdeplot_ax.set_xticks([])
        kdeplot_ax.set_yticks([])
        kdeplot_ax.set_xlabel("")
        kdeplot_ax.set_ylabel("")
        camera.snap()

    animation = camera.animate()
    animation.save(save_path)
    plt.close()
    return HTML(animation.to_html5_video())
