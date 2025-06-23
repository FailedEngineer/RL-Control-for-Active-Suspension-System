import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class RoadProfileGenerator:
    """
    A class to generate different types of road profiles for vehicle suspension simulations.

    This class encapsulates the logic for creating:
    1. A sharp, instantaneous square wave (idealized bump).
    2. A continuous, pseudo-random bumpy road profile.

    Attributes:
        dt (float): The time step for the simulation, used for calculating derivatives.
        _prev_zr (float): Internal state to store the previous road height for derivative calculation.
    """
    def __init__(self, dt=0.001):
        """
        Initializes the RoadProfileGenerator.

        Args:
            dt (float): The simulation time step (in seconds). Default is 0.001.
        """
        self.dt = dt
        self._prev_zr = 0.0

    def reset(self):
        """Resets the internal state of the generator."""
        self._prev_zr = 0.0

    def generate_square_wave_step(self, t, amplitude=0.02, period=3.0):
        """
        Generates a single point (height and velocity) for a square wave profile at time 't'.
        This method is useful for step-by-step simulations.

        Args:
            t (float): The current simulation time.
            amplitude (float): The height of the square wave bump.
            period (float): The total period of the wave (up and down).

        Returns:
            tuple[float, float]: A tuple containing the road height (zr) and road velocity (zr_dot)
                                 at the given time 't'.
        """
        cycle_time = t % period
        half_period = period / 2

        # Create an instantaneous step up/down
        zr = amplitude if cycle_time < half_period else 0.0

        # Calculate derivative numerically based on the previous state
        zr_dot = (zr - self._prev_zr) / self.dt
        self._prev_zr = zr

        return zr, zr_dot

    def generate_bumpy_road_step(self, t, base_amplitude=0.025, noise_amplitude=0.015, period=30.0):
        """
        Generates a single point (height and velocity) for a bumpy road profile at time 't'.
        The profile is a combination of a sine wave and smoothed random noise.
        This method is useful for step-by-step simulations.

        Args:
            t (float): The current simulation time.
            base_amplitude (float): The amplitude of the underlying sine wave.
            noise_amplitude (float): The amplitude of the random noise.
            period (float): The period of the underlying sine wave.

        Returns:
            tuple[float, float]: A tuple containing the road height (zr) and road velocity (zr_dot)
                                 at the given time 't'.
        """
        # Generate a base sine wave profile
        base_profile = base_amplitude * (1 + np.sin(2 * np.pi * t / period)) / 2

        # Generate repeatable, smoothed noise for consistency.
        # The noise is generated once and stored for efficiency.
        if not hasattr(self, '_smooth_noise'):
            np.random.seed(42)  # Use a fixed seed for reproducibility
            raw_noise = (np.random.rand(10000) - 0.5) * noise_amplitude
            self._smooth_noise = pd.Series(raw_noise).rolling(window=50, min_periods=1, center=True).mean().to_numpy()

        # Get the corresponding noise value for the current time
        idx = int(t / self.dt) % len(self._smooth_noise)
        noise = self._smooth_noise[idx]

        zr = base_profile + noise

        # Calculate derivative numerically
        zr_dot = (zr - self._prev_zr) / self.dt
        self._prev_zr = zr

        return zr, zr_dot

def plot_profiles(duration=10.0, dt=0.001):
    """
    Generates and plots the road profiles for demonstration purposes.

    Args:
        duration (float): The total duration of the simulation to plot.
        dt (float): The time step.
    """
    print("Generating and plotting road profiles...")
    
    # --- Setup ---
    time_vector = np.arange(0, duration, dt)
    n_steps = len(time_vector)
    
    # Initialize generators
    square_gen = RoadProfileGenerator(dt)
    bumpy_gen = RoadProfileGenerator(dt)

    # Arrays to store results
    square_zr = np.zeros(n_steps)
    square_zr_dot = np.zeros(n_steps)
    bumpy_zr = np.zeros(n_steps)
    bumpy_zr_dot = np.zeros(n_steps)

    # --- Simulation Loop ---
    for i, t in enumerate(time_vector):
        square_zr[i], square_zr_dot[i] = square_gen.generate_square_wave_step(t)
        bumpy_zr[i], bumpy_zr_dot[i] = bumpy_gen.generate_bumpy_road_step(t)

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('Generated Road Profiles', fontsize=16, fontweight='bold')

    # Plot Square Wave Profile
    axs[0].plot(time_vector, square_zr, label='Road Height ($z_r$)', color='crimson')
    ax0_twin = axs[0].twinx()
    ax0_twin.plot(time_vector, square_zr_dot, label='Road Velocity ($\dot{z}_r$)', color='darkorange', linestyle='--', alpha=0.7)
    axs[0].set_title('Square Wave Road Profile')
    axs[0].set_ylabel('Height (m)')
    ax0_twin.set_ylabel('Velocity (m/s)')
    axs[0].legend(loc='upper left')
    ax0_twin.legend(loc='upper right')
    axs[0].grid(True)

    # Plot Bumpy Road Profile
    axs[1].plot(time_vector, bumpy_zr, label='Road Height ($z_r$)', color='royalblue')
    ax1_twin = axs[1].twinx()
    ax1_twin.plot(time_vector, bumpy_zr_dot, label='Road Velocity ($\dot{z}_r$)', color='mediumseagreen', linestyle='--', alpha=0.7)
    axs[1].set_title('Bumpy Road Profile')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Height (m)')
    ax1_twin.set_ylabel('Velocity (m/s)')
    axs[1].legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    axs[1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == '__main__':
    # Run the demonstration
    plot_profiles(duration=10.0)
