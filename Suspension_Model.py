import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class QuarterCarModel:
    def __init__(self, params=None):
        """
        Initialize the quarter-car active suspension model.
        
        Parameters:
        -----------
        params : dict, optional
            Dictionary of model parameters. If None, default values from the paper are used.
        """
        # Default model parameters from the paper (Table 1, page 4)
        self.params = {
            'ms': 2.45,      # Sprung mass (kg)
            'bs': 7.5,       # Damping of car body (Ns/m)
            'ks': 900,       # Stiffness of car body (N/m)
            'mus': 1,        # Unsprung mass (kg)
            'bus': 5,        # Damping of tire (Ns/m)
            'kus': 2500,     # Stiffness of tire (N/m)
        }
        
        # Update parameters if provided
        if params is not None:
            self.params.update(params)
        
        # Extract parameters for easier access
        ms = self.params['ms']
        bs = self.params['bs']
        ks = self.params['ks']
        mus = self.params['mus']
        bus = self.params['bus']
        kus = self.params['kus']
        
        # State-space matrices from paper (page 3)
        self.A = np.array([
            [0, 1, 0, 0],
            [-ks/ms, -bs/ms, ks/ms, bs/ms],
            [0, 0, 0, 1],
            [ks/mus, bs/mus, -(ks+kus)/mus, -(bs+bus)/mus]
        ])
        
        self.B = np.array([
            [0],
            [1/ms],
            [0],
            [-1/mus]
        ])
        
        self.C = np.array([
            [1, 0, -1, 0],                # Suspension travel
            [-ks/ms, -bs/ms, ks/ms, bs/ms]  # Body acceleration
        ])
        
        self.D = np.array([
            [0],
            [1/ms]
        ])
        
        # State vector: [xs, x_dot_s, xus, x_dot_us]
        self.state = np.zeros(4)
        
        # Initialize road profile variables
        self.zr = 0      # Road profile height
        self.zr_dot = 0  # Rate of change of road profile
    
    def get_disturbance(self, zr, zr_dot):
        """
        Calculate the disturbance vector d based on road profile.
        
        Parameters:
        -----------
        zr : float
            Road profile height (m)
        zr_dot : float
            Rate of change of road profile (m/s)
            
        Returns:
        --------
        d : ndarray
            Disturbance vector
        """
        bus = self.params['bus']
        mus = self.params['mus']
        kus = self.params['kus']
        
        return np.array([
            0,
            0,
            0,
            (bus/mus) * zr_dot + (kus/mus) * zr
        ])
    
    def update(self, force, zr, zr_dot, dt=0.001):
        """
        Update the system state based on the current force and road profile.
        
        Parameters:
        -----------
        force : float
            Control force (N) 
        zr : float
            Road profile height (m)
        zr_dot : float
            Rate of change of road profile (m/s)
        dt : float
            Time step (s)
            
        Returns:
        --------
        state : ndarray
            Updated state vector [xs, x_dot_s, xus, x_dot_us]
        """
        # Get disturbance from road profile
        d = self.get_disturbance(zr, zr_dot)
        
        # Compute state derivative: x_dot = Ax + Bu + d
        state_dot = (self.A @ self.state.reshape(-1, 1) + 
                     self.B * force + 
                     d.reshape(-1, 1)).flatten()
        
        # Euler integration
        self.state = self.state + state_dot * dt
        
        return self.state
    
    def get_output(self, force):
        """
        Calculate the output vector y = Cx + Du.
        
        Parameters:
        -----------
        force : float
            Control force (N)
            
        Returns:
        --------
        y : ndarray
            Output vector [suspension_travel, body_acceleration]
        """
        return (self.C @ self.state.reshape(-1, 1) + self.D * force).flatten()
    
    def reset(self):
        """Reset the model to initial conditions."""
        self.state = np.zeros(4)