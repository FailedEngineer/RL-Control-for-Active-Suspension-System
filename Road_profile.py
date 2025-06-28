import numpy as np

class RoadProfileGenerator:
    """Base class for road profile generators."""
    def get_profile(self, t):
        """
        Returns the road displacement at a given time.

        Args:
            t (float or np.ndarray): The current simulation time(s).

        Returns:
            float or np.ndarray: The road displacement x_g (m).
        """
        raise NotImplementedError

class BumpProfile(RoadProfileGenerator):
    """Generates a single bump, modeled as a square wave or half-sine."""
    def __init__(self, start_time=1.0, duration=0.5, height=0.05):
        self.start_time = start_time
        self.duration = duration
        self.height = height

    def get_profile(self, t):
        # A single 5cm bump modeled as a half-sine wave for smoothness
        condition = (t >= self.start_time) & (t < self.start_time + self.duration)
        profile = np.zeros_like(t, dtype=float)
        profile[condition] = self.height * np.sin(np.pi * (t[condition] - self.start_time) / self.duration)
        return profile

class SquareWaveProfile(RoadProfileGenerator):
    """Generates a repeating square wave to simulate periodic bumps."""
    def __init__(self, period=2.0, amplitude=0.02):
        self.period = period
        self.amplitude = amplitude

    def get_profile(self, t):
        # Generates a square wave: +amplitude for the first half, -amplitude for the second.
        # This is a challenging profile for suspension systems.
        return self.amplitude * np.sign(np.sin(2 * np.pi * t / self.period))


class ISO8608Profile(RoadProfileGenerator):
    """
    Generates a random road profile based on ISO 8608 Power Spectral Density (PSD).
    This creates a more realistic continuous random road surface.
    """
    def __init__(self, road_class='A', vehicle_speed_kmh=80, length=500, dt=0.01):
        """
        Args:
            road_class (str): Road class from 'A' (very good) to 'H' (very poor).
            vehicle_speed_kmh (float): Vehicle speed in km/h.
            length (int): Length of the road profile in meters.
            dt (float): Time step for the generated profile.
        """
        self.road_class = road_class
        self.vehicle_speed_ms = vehicle_speed_kmh / 3.6
        self.length = length
        self.dt = dt
        self.profile = self._generate()
        self.time_vector = np.arange(0, len(self.profile) * self.dt, self.dt)

    def _get_psd(self, spatial_freq):
        """Calculates the PSD based on ISO 8608."""
        n0 = 0.1  # Reference spatial frequency (cycles/m)
        psd_ref_map = {
            'A': 1e-6, 'B': 4e-6, 'C': 16e-6, 'D': 64e-6,
            'E': 256e-6, 'F': 1024e-6, 'G': 4096e-6, 'H': 16384e-6
        }
        psd_ref = psd_ref_map.get(self.road_class.upper(), 1e-6)
        return psd_ref * (spatial_freq / n0)**(-2)

    def _generate(self):
        """Generates the road profile using Inverse Fourier Transform."""
        N = int(self.length / (self.vehicle_speed_ms * self.dt))
        df = self.vehicle_speed_ms / self.length
        
        freq = np.fft.fftfreq(N, self.dt)[1:N//2]
        spatial_freq = freq / self.vehicle_speed_ms
        
        psd_values = self._get_psd(spatial_freq)
        
        # Create random Fourier coefficients
        random_phases = np.exp(1j * 2 * np.pi * np.random.rand(len(freq)))
        fourier_coeffs = np.sqrt(psd_values * N * df) * random_phases
        
        full_coeffs = np.zeros(N, dtype=np.complex128)
        full_coeffs[1:N//2] = fourier_coeffs
        full_coeffs[-(N//2)+1:] = np.conj(fourier_coeffs[::-1])

        # Inverse FFT to get the time-domain signal
        profile = np.fft.ifft(full_coeffs).real
        return profile

    def get_profile(self, t):
        """Interpolates the generated profile to get displacement at time t."""
        return np.interp(t, self.time_vector, self.profile, left=0, right=0)
