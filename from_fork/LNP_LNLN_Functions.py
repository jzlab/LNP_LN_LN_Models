# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 14:21:02 2025

@author: Nilou Ghazavi
"""


# import libraries 
 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
import matplotlib.patches as patches
from tqdm import tqdm
import cv2
import os
import random
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython.display import Image, display
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from matplotlib.patches import Rectangle
import matplotlib as mpl
import gc 
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from dataclasses import dataclass
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython.display import Image, display
from matplotlib import patches
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import time 
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import gc
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import matplotlib.colors as mcolors
import re, glob
from typing import Dict, Tuple, Optional, Sequence
import psutil
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from scipy.ndimage import uniform_filter


#%% RGC parameters 


#RF diameter for each RGC type
ON_Parasol_um=300 # um
OFF_Parasol_um=250 # um
ON_Midget_um=125# um #125
OFF_Midget_um=100 # um
checker_um = 30  # μm/pixel (800X600-->100X75 (3.8 um/pixel))


RGC_PARAMS = {
    'ON_Parasol': {
        'center_size': ON_Parasol_um/(checker_um),
        'surround_size': (2*ON_Parasol_um)/checker_um,
        'center_strength': 1.0, 'surround_strength': -0.5,
        'peak1_ms': 35, 'peak2_ms': 55, 'width1_ms': 21, 'width2_ms': 38,'amp1':0.6, 'amp2':0.3,
        'alpha':1, 'beta': 2, 'gamma': 0,
        'alpha_sub': 3, 'beta_sub':10, 'gamma_sub': -0.3,
        'gamma_lnln':5
    },
    'OFF_Parasol': {
        'center_size': OFF_Parasol_um/(checker_um),
        'surround_size': (2*OFF_Parasol_um)/checker_um,
        'center_strength': 1.0, 'surround_strength': -0.5,
        'peak1_ms': 20, 'peak2_ms': 70, 'width1_ms': 25, 'width2_ms': 50,'amp1':0.5, 'amp2':0.15,
        'alpha': 1, 'beta': 2, 'gamma': 0,
        'alpha_sub': 1, 'beta_sub': 10, 'gamma_sub':0,
        'gamma_lnln':2
    },
    'ON_Midget': {
        'center_size': ON_Midget_um/(checker_um),
        'surround_size': (2*ON_Midget_um)/checker_um,
        'center_strength': 1.0, 'surround_strength': -0.5,
        'peak1_ms': 30, 'peak2_ms': 70, 'width1_ms': 20, 'width2_ms': 50,'amp1':0.5, 'amp2':0.15,
        'alpha': 1, 'beta':1, 'gamma':0,
        'alpha_sub': 10, 'beta_sub': 20, 'gamma_sub': 1,
        'gamma_lnln': 5
    },
    'OFF_Midget': {
        'center_size': OFF_Midget_um/(checker_um),
        'surround_size': (2*OFF_Midget_um)/checker_um,
        'center_strength': 1.0, 'surround_strength': -0.5,
        'peak1_ms': 30, 'peak2_ms': 100, 'width1_ms': 25, 'width2_ms': 50,'amp1':0.5, 'amp2':0.05,
        'alpha': 1, 'beta': 1, 'gamma':0,
       'alpha_sub': 1, 'beta_sub': 10, 'gamma_sub': 0,
        'gamma_lnln': 2
    }
}




# get mosaic parameters 
def get_biological_mosaic_parameters():
    """
    Get biologically accurate mosaic parameters for different RGC types
    """
    return {
        'ON_Parasol': {
            'rf_spacing': ON_Parasol_um/(checker_um),  # pixels between RF centers
            'rf_diameter': ON_Parasol_um/(checker_um),  # from your surround_size parameter
            'coverage_factor': 1,  
            'color': 'red'
        },
        'OFF_Parasol': {
            'rf_spacing':  OFF_Parasol_um/(checker_um),  # same as ON Parasol
            'rf_diameter': OFF_Parasol_um/(checker_um),  # from your surround_size parameter
            'coverage_factor': 1,
            'color': 'orange'
        },
        'ON_Midget': {
            'rf_spacing': ON_Midget_um/(checker_um),  # much denser packing
            'rf_diameter': ON_Midget_um/(checker_um),  # from your surround_size parameter
            'coverage_factor': 1,
            'color': 'green'
        },
        'OFF_Midget': {
            'rf_spacing': OFF_Midget_um/(checker_um),  # same as ON Midget
            'rf_diameter': OFF_Midget_um/(checker_um),  # from your surround_size parameter
            'coverage_factor': 1,
            'color': 'blue'
        }
    }



# input stimulus size
# height 
x=35
# width
y=35


# temporal filter

cell_memory_ms=250
cell_memory_frame=30


# default color 
DEFAULT_COLORS = {
    'ON_Parasol': 'red',
    'OFF_Parasol': 'orange',
    'ON_Midget': 'green',
    'OFF_Midget': 'blue'
}



#%%  1: load the natural movie and normalize it


def load_natural_movie_debug(
    video_path: str,
    frames: int = None,             
    height: int = 35,
    width: int = 35,
    start_frame: int = 0,           
    end_frame: int = None,           
    normalize: str = "zero_mean_unit",  
    verbose: bool = True
):
    """
    Load a segment of a video as grayscale frames resized to (height, width).

    Priority for range selection:
      1) If `frames` is not None: load from `start_frame` for `frames` frames.
      2) Else if `end_frame` is not None: load from `start_frame` to `end_frame` (exclusive).
      3) Else: load from `start_frame` to end-of-file.

    normalize:
      - "none"            : returns [0,1] grayscale
      - "zero_mean"       : subtract global mean across loaded frames
      - "zero_mean_unit"  : subtract mean, then divide by max(abs) to map to ~[-1,1]
    """
    if verbose:
        print(f"Attempting to load: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        if verbose:
            print("ERROR: Could not open video file")
        return np.array([])

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if verbose:
        print(f"  Total frames: {total_frames}")
        print(f"  Original size: {original_width}x{original_height}")

    if total_frames <= 0:
        if verbose:
            print("ERROR: Video has 0 frames")
        cap.release()
        return np.array([])

    # Clip start_frame
    start_frame = max(0, int(start_frame))
    if start_frame >= total_frames:
        if verbose:
            print(f"ERROR: start_frame {start_frame} >= total_frames {total_frames}")
        cap.release()
        return np.array([])

    # Determine how many frames to load
    if frames is not None:
        frames_to_load = int(frames)
        end_frame_eff = min(total_frames, start_frame + frames_to_load)
    else:
        if end_frame is None:
            end_frame_eff = total_frames
        else:
            end_frame_eff = max(start_frame, min(total_frames, int(end_frame)))
        frames_to_load = max(0, end_frame_eff - start_frame)

    if frames_to_load == 0:
        if verbose:
            print("Nothing to load (frames_to_load == 0).")
        cap.release()
        return np.array([])

    # Seek to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    movie_frames = []
    if verbose:
        print(f"Loading frames [{start_frame}:{start_frame + frames_to_load}) → {frames_to_load} frames")

    for i in range(frames_to_load):
        ret, frame = cap.read()
        if not ret or frame is None:
            if verbose:
                print(f"Stopped early at local index {i} (global {start_frame+i}).")
            break

        # BGR → Gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize
        gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)

        # [0,1]
        gray = gray.astype(np.float32) / 255.0
        movie_frames.append(gray)

        if verbose and (i % 25 == 0):
            print(f"  Loaded frame {start_frame + i}")

    cap.release()

    if len(movie_frames) == 0:
        if verbose:
            print("ERROR: No frames were loaded")
        return np.array([])

    movie = np.stack(movie_frames, axis=0)  # (N, H, W)

    # Normalization options
    if normalize == "zero_mean":
        movie = movie - movie.mean()
    elif normalize == "zero_mean_unit":
        movie = movie - movie.mean()
        max_abs = np.max(np.abs(movie))
        if max_abs > 1e-8:
            movie = movie / max_abs
    elif normalize == "none":
        pass
    else:
        if verbose:
            print(f"Unknown normalize='{normalize}', using 'zero_mean_unit'")
        movie = movie - movie.mean()
        max_abs = np.max(np.abs(movie))
        if max_abs > 1e-8:
            movie /= max_abs

    if verbose:
        print(f"Loaded movie shape: {movie.shape}  (dtype={movie.dtype})")

    return movie




def get_natural_movie_stimulus(movie_array, frame_index):
    """Get a specific frame from the natural movie"""
    return movie_array[frame_index, :, :]





#%% 2.  Spatial Filter (Gaussian Filter)

def create_2d_gaussian(width, height, sigma, center_x=None, center_y=None):
    """
    Create 2D Gaussian with rectangular dimensions
    
    Parameters:
    -----------
    width : int
        Width of the output array
    height : int
        Height of the output array
    sigma : float
        Standard deviation of the Gaussian
    center_x, center_y : float, optional
        Center position. If None, uses center of array
        
    Returns:
    --------
    gaussian : np.ndarray
        Shape (height, width)
    """
    if center_x is None:
        center_x = width / 2
    if center_y is None:
        center_y = height / 2
    
    # Create coordinate grids
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)
    
    # Calculate Gaussian
    gaussian = np.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * sigma**2))
    
    return gaussian



# Difference of Gaussian 

def create_spatial_filter(center_size, surround_size, center_strength, surround_strength, 
                                target_height=x, target_width=y):
    """
    Create spatial filter that matches movie size (height×width)
    
    Parameters:
    -----------
    center_size, surround_size : float
        Sigma values for center and surround
    center_strength, surround_strength : float
        Strength values for center and surround
    target_height : int
        Height of the target (75 for your movie)
    target_width : int
        Width of the target (100 for your movie)
        
    Returns:
    --------
    spatial_filter : np.ndarray
        Shape (target_height, target_width) = (75, 100)
    """
    
    # Create 2D Gaussians with correct dimensions
    center_gaussian = create_2d_gaussian(target_width, target_height, center_size)
    surround_gaussian = create_2d_gaussian(target_width, target_height, surround_size)
    
    # Normalize and combine
    center_gaussian = center_gaussian / np.sum(center_gaussian)
    surround_gaussian = surround_gaussian / np.sum(surround_gaussian)
    spatial_filter = center_strength * center_gaussian + surround_strength * surround_gaussian
    
    return spatial_filter


#%% 3. Temporal Filter : difference of two low pass filters

def create_temporal_filter(peak1_ms=40, peak2_ms=120, width1_ms=30, width2_ms=60,
                          amp1=1.0, amp2=0.5, frame_rate=120, memory_ms=250, 
                          smooth_onset=True, cell_type='ON'):
    """
    Create biphasic temporal filter without zero-integral constraint
    
    Parameters:
    -----------
    peak1_ms : float
        Time of first lobe peak (ms)
    peak2_ms : float
        Time of second lobe peak (ms)
    width1_ms : float
        Width of first lobe (ms)
    width2_ms : float
        Width of second lobe (ms)
    amp1 : float
        Amplitude of first lobe (default: 1.0)
    amp2 : float
        Amplitude of second lobe (default: 0.5)
    frame_rate : float
        Frame rate (Hz)
    memory_ms : float
        Total filter duration (ms)
    smooth_onset : bool
        Apply smooth onset window
    cell_type : str
        'ON' or 'OFF'
        
    Returns:
    --------
    temporal_filter : np.ndarray
        Shape: (memory_frames,)
    """
    memory_frames = int(memory_ms * frame_rate / 1000)
    dt_ms = 1000.0 / frame_rate
    t_ms = np.arange(memory_frames) * dt_ms
    
    # First lobe: Gaussian with custom amplitude
    lobe1 = amp1 * np.exp(-((t_ms - peak1_ms)**2) / (2 * width1_ms**2))
    
    # Second lobe: Gaussian with custom amplitude
    lobe2 = amp2 * np.exp(-((t_ms - peak2_ms)**2) / (2 * width2_ms**2))
    
    # Apply smooth onset 
    if smooth_onset:
        onset_tau_ms = 15
        onset_window = 1 - np.exp(-t_ms / onset_tau_ms)
        lobe1 = lobe1 * onset_window
        lobe2 = lobe2 * onset_window
    
    # Create biphasic filter based on cell type (NO area balancing)
    if cell_type.startswith('OFF'):
        # OFF cells: negative first, positive second
        temporal_filter = -lobe1 + lobe2
    else:
        # ON cells: positive first, negative second  
        temporal_filter = lobe1 - lobe2
    
    return temporal_filter



#%% 4. Spatiotemporal filter

def create_spatiotemporal_filter(spatial_filter, temporal_filter):
    """
    Create spatiotemporal filter w from spatial and temporal components
    
    w(tau, x, y) = spatial_filter(x, y) * temporal_filter(tau)
    
    Parameters:
    -----------
    spatial_filter : np.ndarray
        2D spatial filter (height, width)
    temporal_filter : np.ndarray
        1D temporal filter (temporal_length,)
        
    Returns:
    --------
    np.ndarray
        3D spatiotemporal filter (temporal_length, height, width)
    """
    
    temporal_length = len(temporal_filter)
    height, width = spatial_filter.shape
    
    # Create 3D spatiotemporal filter
    spatiotemporal_filter = np.zeros((temporal_length, height, width))
    # w(T, X, Y) = spatial (x,y) x temporal (t)
    for tau in range(temporal_length):
        spatiotemporal_filter[tau, :, :] = spatial_filter * temporal_filter[tau]
    
    
    # Normalization of the whole 3D kernel 
    euclidean_norm = np.linalg.norm(spatiotemporal_filter)
    spatiotemporal_filter = spatiotemporal_filter / euclidean_norm

    
    return spatiotemporal_filter



#%% 5. Nonlinearity 

def apply_nonlinearity(generator_signal, alpha, beta, gamma, nonlinearity_type='soft_rectifier'):
    if nonlinearity_type == 'soft_rectifier':
        # Soft rectifier for smooth retinal-like responses
        firing_rate = alpha * np.log(1 + np.exp(beta * (generator_signal - gamma)))
        
    elif nonlinearity_type == 'sigmoid':
        # Sigmoid for bounded responses with saturation
        firing_rate = alpha / (1 + np.exp(-beta * (generator_signal - gamma)))
        
    elif nonlinearity_type == 'relu':
    # Rectified Linear Unit (ReLU) - simple threshold with linear response
        firing_rate = alpha * np.maximum(0, beta * (generator_signal - gamma))
        
    
    return np.maximum(0, firing_rate)



#%% 6. Poisson Distribution (spike times)

# spike count 
def generate_poisson_spikes(firing_rate, dt=8.3):
    """Generate Poisson spikes"""
    
    lambda_param = firing_rate * (dt / 1000.0)
    spikes = np.random.poisson(lambda_param)
    return spikes.astype(int)


# spike time (uniform distribution)
def poisson_times_from_rate(firing_rate, dt=8.3):
    """
    firing_rate : array-like of length N, Hz (one rate per time bin)
    dt          : ms
    Returns     : sorted 1D np.ndarray of spike times (ms) over the full duration
    """
    rates = np.asarray(firing_rate, dtype=float)
    N = rates.shape[0]
    spike_times = []

    lam_per_bin = rates * (dt / 1000.0)  # expected spikes each bin
    k = np.random.poisson(lam_per_bin)   # (N,) integer counts

    for i, ki in enumerate(k):
        if ki > 0:
            bin_start = i * dt
            bin_end   = (i + 1) * dt
            # Use uniform distribution to getnumber of spikes
            spike_times_in_bin = np.random.uniform(bin_start, bin_end, int(ki))
            spike_times.extend(spike_times_in_bin)

    return np.sort(np.array(spike_times, dtype=float))


# poisson spikes with time
def generate_poisson_spikes_with_times(firing_rate, num_frames=144, dt=8.3, seed=None):
    """
    Generate Poisson-distributed spike times for a given firing rate.

    Parameters
    ----------
    firing_rate : float
        Mean firing rate in Hz (spikes per second).
    num_frames : int, optional
        Number of time bins or frames (default: 144).
    dt : float, optional
        Duration of each time bin in milliseconds (default: 8.3 ms).
    seed : int, optional
        Random seed for reproducibility (default: None).

    Returns
    -------
    np.ndarray
        Array of spike times in milliseconds.
    """

    if seed is not None:
        np.random.seed(seed)

    total_duration = num_frames * dt
    spike_times = []

    for bin_index in range(num_frames):
        # Expected number of spikes in this bin
        lambda_param = firing_rate * (dt / 1000.0)  # convert ms to s
        num_spikes = np.random.poisson(lambda_param)

        # Generate random spike times within the bin
        if num_spikes > 0:
            bin_start = bin_index * dt
            bin_end = (bin_index + 1) * dt
            spike_times_in_bin = np.random.uniform(bin_start, bin_end, num_spikes)
            spike_times.extend(spike_times_in_bin)

    return np.sort(np.array(spike_times))



#%% 7. LNP Input : rolling window

def rolling_window(array, window, time_axis=0):
    """
    Make an ndarray with a rolling window of the last dimension
    
    Parameters
    ----------
    array : array_like
        Array to add rolling window to
    window : int
        Size of rolling window
    time_axis : int, optional
        The axis of the temporal dimension, either 0 or -1 (Default: 0)
 
    Returns
    -------
    Array that is a view of the original array with a added dimension
    of size w.
    
    Examples
    --------
    >>> x=np.arange(10).reshape((2,5))
    >>> rolling_window(x, 3)
    array([[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
           [[5, 6, 7], [6, 7, 8], [7, 8, 9]]])
    
    Calculate rolling mean of last dimension:
    >>> np.mean(rolling_window(x, 3), -1)
    array([[ 1.,  2.,  3.],
           [ 6.,  7.,  8.]])
    """
    if window > 0:
        if time_axis == 0:
            array = array.T
    
        elif time_axis == -1:
            pass
    
        else:
            raise ValueError('Time axis must be 0 (first dimension) or -1 (last)')
    
        assert window < array.shape[-1], "`window` is too long."
    
        # with strides
        shape = array.shape[:-1] + (array.shape[-1] - window + 1, window)
        strides = array.strides + (array.strides[-1],)
        arr = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)
    
        if time_axis == 0:
            return np.rollaxis(arr.T, 1, 0)
        else:
            return arr
    else:
        return array




# LNP input (stimulus-windowed)
def compute_stim_windowed(movie, temporal_filter):
    """
    Create windowed stimulus once
    
    Parameters:
    -----------
    movie : np.ndarray (time, height, width)
    temporal_filter : np.ndarray (temporal_length,)
    
    Returns:
    --------
    stim_windowed : np.ndarray (n_windows, temporal_length, height, width)
    """
    time_steps, height, width = movie.shape
    temporal_length = len(temporal_filter)
    
    # Use rolling window to create overlapping temporal chunks
    stim_windowed = rolling_window(movie, temporal_length, time_axis=0)
    # Shape: (time_steps - temporal_length + 1, temporal_length, height, width)
    
    return stim_windowed



#%% 8. Use LNP Model to predict spike rate and spike time 

def compute_lnp(movie, spatial_filter, temporal_filter, alpha, beta, gamma, dt=8.3, already_windowed=True):
    """
    LNP forward pass.

    If already_windowed=False:
        movie: (T, H, W)  -> will be windowed internally to (N, L, H, W)
    If already_windowed=True:
        movie: (N, L, H, W)  (windowed stimulus)

    Returns
    -------
    generator_signal : (N,)
    firing_rate      : (N,)
    spikes           : (N,)  (Poisson counts per bin)
    """
    # Build spatiotemporal filter and flatten
    w = create_spatiotemporal_filter(spatial_filter, temporal_filter).astype(np.float32)  # (L,H,W)
    w_flat = w.ravel()  # (L*H*W,)
    L = len(temporal_filter)

    # Prepare windowed stimulus
    if already_windowed:
        stim_windowed = movie
        N, L_win, H, W = stim_windowed.shape
        assert L_win == L, f"Window length {L_win} != temporal_filter length {L}"
    else:
        # movie is (T,H,W) -> window to (N,L,H,W)
        stim_windowed = compute_stim_windowed(movie, temporal_filter)  # (N,L,H,W)
        N, L_win, H, W = stim_windowed.shape
        assert L_win == L, f"Window length {L_win} != temporal_filter length {L}"

    # Vectorized linear stage: flatten each window and do a single matmul
    S = stim_windowed.reshape(N, -1).astype(np.float32)   # (N, L*H*W)
    generator_signal = S @ w_flat                          # (N,)

    # Nonlinearity and spikes
    max_firing_rate=700
    firing_rate = apply_nonlinearity(generator_signal, alpha, beta, gamma)  # (N,)
    firing_rate = np.minimum(firing_rate, max_firing_rate)
    spikes = generate_poisson_spikes(firing_rate, dt)
    spike_times=poisson_times_from_rate (firing_rate, dt)
    return generator_signal, firing_rate, spikes,  spike_times





#%% 9. Plot all the filters 

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_rgc_components(
    RGC_PARAMS,
    create_spatial_filter,
    create_temporal_filter,
    apply_nonlinearity,
    *,
    rgc_types=('ON_Parasol','OFF_Parasol','ON_Midget','OFF_Midget'),
    
    display_names=('ON Parasol','OFF Parasol','ON Midget','OFF Midget'),
    
    fps=120,                     # frame rate for time axis
    xlim_ms=None,                # optional time limit
    gen_range=(-3, 3),           # x-range for nonlinearity
    inset_size=("35%", "35%"),   # inset (width, height)
    inset_loc=(0.55, 0.05),      # inset (x0, y0) in axis fraction
    font_size=22,                # <--- all axis labels & ticks
):
    fig, axes = plt.subplots(3, 4, figsize=(32, 20), constrained_layout=True)
    nonlin_axes = {}
    for col, (rgc_type, display_name) in enumerate(zip(rgc_types, display_names)):
        params = RGC_PARAMS[rgc_type]
        color  = DEFAULT_COLORS.get(rgc_type, 'k')
        
        # --- Row 1: Spatial RF ---
        ax_s = axes[0, col]
        spatial_filter = create_spatial_filter(
            params['center_size'], params['surround_size'],
            params['center_strength'], params['surround_strength']
        )
        vmax = float(np.max(np.abs(spatial_filter))) or 1.0
        ax_s.imshow(spatial_filter, cmap='gray', vmin=-vmax, vmax=vmax)
        ax_s.set_title(f'{display_name}\nSpatial Receptive Field',
                       color='black', fontweight='bold', fontsize=font_size)
        ax_s.axis('off')
        
        # --- Row 2: Temporal RF ---
        ax_t = axes[1, col]
        temporal_filter = create_temporal_filter(
            peak1_ms=params['peak1_ms'], 
            peak2_ms=params['peak2_ms'],
            width1_ms=params['width1_ms'], 
            width2_ms=params['width2_ms'],
            amp1=params['amp1'] ,    # <--- ADD THIS LINE
            amp2=params['amp2'] ,     # <--- ADD THIS LINE
            smooth_onset=True, 
            cell_type=rgc_type
        )
        # Create time axis going backwards (0 on right, past on left)
        time_ms = np.arange(len(temporal_filter)) * (1000.0 / fps)
        time_ms_backwards = -time_ms[::-1]  # Reverse and negate
        
        # Plot with reversed filter
        ax_t.plot(time_ms_backwards, temporal_filter[::-1], color=color, linewidth=6)
        ax_t.axhline(0, color='0.4', linestyle='-', alpha=0.4, linewidth=1.2)
        ax_t.set_xlabel('Time to spike (ms)', fontsize=28)  # Updated label
        ax_t.set_ylabel('Filter Amplitude', fontsize=28)
        ax_t.set_title(f'{display_name}\nTemporal Receptive Field',
                       color='black', fontweight='bold', fontsize=font_size)
        
        # Set x-limits with 0 on the right
        if xlim_ms is not None:
            ax_t.set_xlim(-xlim_ms, 0)
        else:
            ax_t.set_xlim(-time_ms[-1] if len(time_ms) else 0, 0)
        
        ax_t.tick_params(axis='both', labelsize=28)
        # ax_t.grid(True, alpha=0.3)  # <--- ADD GRID HERE
        # --- Inset: Nonlinearity inside temporal RF ---
        inset_ax = inset_axes(ax_t, width=inset_size[0], height=inset_size[1],
                              bbox_to_anchor=(inset_loc[0], inset_loc[1], 1, 1),
                              bbox_transform=ax_t.transAxes, loc='lower left', borderpad=0)
        gs = np.linspace(gen_range[0], gen_range[1], 600)
        fr = apply_nonlinearity(gs, params['alpha'], params['beta'], params['gamma'])
        inset_ax.plot(gs, fr, color=color, linewidth=6)
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        for spine in inset_ax.spines.values():
            spine.set_color('0.3')
            spine.set_linewidth(0.8)
        
        # --- Row 3: Full Nonlinearity ---
        ax_n = axes[2, col]
        ax_n.plot(gs, fr, color=color, linewidth=3)
        ax_n.set_xlabel('Generator Signal', fontsize=font_size)
        ax_n.set_ylabel('Firing Rate (Hz)', fontsize=font_size)
        ax_n.set_title(f'{display_name}\nNonlinearity',
                       color='black', fontweight='bold', fontsize=font_size)
        ax_n.tick_params(axis='both', labelsize=font_size)
        nonlin_axes[rgc_type] = ax_n
        
    return fig, nonlin_axes


#%% 10. create Hexogonal Mosaic

# Hexagonal mosaic 

def create_hexagonal_mosaic(visual_field_size, rf_spacing, coverage_factor=1.0, 
                           max_cells=None, selection_method='center_first',
                           rf_diameter=None, keep_boundary_cells=False):
    """
    Create hexagonal mosaic of receptive field positions
    
    Parameters:
    -----------
    visual_field_size : tuple (height, width)
    rf_spacing : float
        Spacing between RF centers
    coverage_factor : float
        Adjust spacing (1.0 = normal, >1.0 = closer, <1.0 = farther)
    max_cells : int or None
        Maximum number of cells to return. If None, returns all cells
    selection_method : str
        How to select cells when max_cells is specified:
        - 'center_first': Start from center and work outward
        - 'random': Random selection from all possible positions
        - 'edge_first': Start from edges and work inward
        - 'grid_order': Take cells in the order they're generated
    rf_diameter : float or None
        If provided, ensures RF centers are far enough from edges that
        entire RF stays within visual field
    keep_boundary_cells : bool
        If True, allows cells near boundaries (RFs may extend beyond field)
        If False, only keeps cells where entire RF is within field
    """
    height, width = visual_field_size
    
    # Adjust spacing for coverage
    effective_spacing = rf_spacing / coverage_factor
    
    # Calculate number of rows and columns - FIXED for complete coverage
    row_spacing = effective_spacing * np.sqrt(3) / 2
    n_rows = int(np.ceil(height / row_spacing)) + 2  # Use ceil for better coverage
    n_cols = int(np.ceil(width / effective_spacing)) + 2
    
    # Determine boundary constraints
    if rf_diameter is not None and not keep_boundary_cells:
        # Keep only cells where entire RF stays within visual field
        rf_radius = rf_diameter / 2
        margin = rf_radius
    else:
        # Allow cells near edges for better coverage
        margin = effective_spacing * 0.5
    
    # Generate all possible positions first
    all_positions = []
    for row in range(n_rows):
        for col in range(n_cols):
            # Hexagonal offset for odd rows
            x_offset = (effective_spacing / 2) if row % 2 == 1 else 0
            
            x = col * effective_spacing + x_offset
            y = row * row_spacing
            
            # Check if position is within bounds (considering RF size)
            if (margin <= x <= width - margin and 
                margin <= y <= height - margin):
                all_positions.append((x, y, row, col))  # Store row, col for sorting
    
    # If no limit specified, return all positions
    if max_cells is None:
        return np.array([(x, y) for x, y, _, _ in all_positions])
    
    # Apply selection method to get exactly max_cells
    if len(all_positions) <= max_cells:
        # If we have fewer positions than requested, return all
        selected_positions = [(x, y) for x, y, _, _ in all_positions]
    else:
        # Select based on method
        if selection_method == 'center_first':
            # Calculate distance from center for each position
            center_x, center_y = width / 2, height / 2
            positions_with_dist = []
            for x, y, row, col in all_positions:
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                positions_with_dist.append((x, y, dist))
            
            # Sort by distance from center and take first max_cells
            positions_with_dist.sort(key=lambda p: p[2])
            selected_positions = [(x, y) for x, y, _ in positions_with_dist[:max_cells]]
        
        elif selection_method == 'edge_first':
            # Calculate distance from center and take farthest ones
            center_x, center_y = width / 2, height / 2
            positions_with_dist = []
            for x, y, row, col in all_positions:
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                positions_with_dist.append((x, y, dist))
            
            # Sort by distance from center (descending) and take first max_cells
            positions_with_dist.sort(key=lambda p: p[2], reverse=True)
            selected_positions = [(x, y) for x, y, _ in positions_with_dist[:max_cells]]
        
        elif selection_method == 'random':
            # Random selection
            np.random.seed(42)  # For reproducibility
            selected_indices = np.random.choice(len(all_positions), max_cells, replace=False)
            selected_positions = [(all_positions[i][0], all_positions[i][1]) for i in selected_indices]
        
        elif selection_method == 'grid_order':
            # Take first max_cells in the order they were generated
            selected_positions = [(x, y) for x, y, _, _ in all_positions[:max_cells]]
        
        else:
            raise ValueError(f"Unknown selection_method: {selection_method}")
    
    return np.array(selected_positions)




def create_all_mosaics(visual_field_size=(x, y), max_cells=None, 
                      keep_boundary_cells=False):
    """
    Create mosaics for all 4 RGC types
    
    Parameters:
    -----------
    visual_field_size : tuple
        (height, width) of visual field
    max_cells : int or None
        If None, creates complete tiling with all necessary cells
        If int, limits to that number of cells per type
    keep_boundary_cells : bool
        If False (default), only keeps cells where entire RF is within field
        If True, allows cells near boundaries (RFs may extend beyond field)
    """
    mosaic_params = get_biological_mosaic_parameters()
    mosaics = {}
    
    for rgc_type, params in mosaic_params.items():
        positions = create_hexagonal_mosaic(
            visual_field_size,
            params['rf_spacing'],
            params['coverage_factor'],
            max_cells=max_cells,
            rf_diameter=params['rf_diameter'],
            keep_boundary_cells=keep_boundary_cells
        )
        
        mosaics[rgc_type] = {
            'positions': positions,
            'rf_diameter': params['rf_diameter'],
            'rf_spacing': params['rf_spacing'],
            'color': params['color'],
            'n_cells': len(positions)
        }
        
        print(f"{rgc_type}: {len(positions)} cells, spacing: {params['rf_spacing']} px")
    
    return mosaics




def create_flexible_overlaid_mosaic(visual_field_size, overlay_config,
                                   keep_boundary_cells=False):
    """
    Flexible overlaid mosaic creator
    
    overlay_config example:
    [
        {'cell_type': 'ON_Parasol', 'n_cells': None, 'base_spacing': 6.0, 'offset': (0, 0)},
        {'cell_type': 'OFF_Parasol', 'n_cells': None, 'base_spacing': 6.0, 'offset': (3, 0)},
        {'cell_type': 'ON_Midget', 'n_cells': None, 'base_spacing': 3.0, 'offset': (0, 0)},
        {'cell_type': 'OFF_Midget', 'n_cells': None, 'base_spacing': 3.0, 'offset': (1.5, 0)}
    ]
    
    Parameters:
    -----------
    keep_boundary_cells : bool
        If False (default), only keeps cells where entire RF is within field
        If True, allows cells near boundaries
    
    Note: n_cells=None gives complete tiling
    """
    rgc_params = get_biological_mosaic_parameters()
    overlaid_mosaic = {}
    
    for config in overlay_config:
        cell_type = config['cell_type']
        n_cells = config.get('n_cells', None)
        base_spacing = config.get('base_spacing', 6.0)
        offset = config.get('offset', (0, 0))
        coverage_factor = config.get('coverage_factor', 1.0)
        
        if cell_type not in rgc_params:
            continue
        
        
        
        if 'base_spacing' in config:
            base_spacing = config['base_spacing']
        else:
            # Use the biological spacing from rgc_params
            # This ensures proper tiling regardless of field size
            rf_diameter = rgc_params[cell_type]['rf_diameter']
            base_spacing = rf_diameter / coverage_factor    
        # Create base positions
        base_positions = create_hexagonal_mosaic(
            visual_field_size, 
            base_spacing, 
            coverage_factor, 
            n_cells, 
            'center_first',
            rf_diameter=rgc_params[cell_type]['rf_diameter'],
            keep_boundary_cells=keep_boundary_cells
        )
        
        # Apply offset
        offset_x, offset_y = offset
        final_positions = base_positions + np.array([offset_x, offset_y])
        
        overlaid_mosaic[cell_type] = {
            'positions': final_positions,
            'rf_diameter': rgc_params[cell_type]['rf_diameter'],
            'color': rgc_params[cell_type]['color'],
            'n_cells': len(final_positions)
        }
    
    return overlaid_mosaic



def plot_rgc_mosaics(mosaics, visual_field_size=(x,y), show_circles=True):
    """
    Plot the 4 RGC mosaics exactly like in your reference image
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    rgc_types = ['ON_Parasol', 'OFF_Parasol', 'ON_Midget', 'OFF_Midget']
    display_names = ['ON Parasol', 'OFF Parasol', 'ON Midget', 'OFF Midget']
    for idx, rgc_type in enumerate((rgc_types)):
        display_name=display_names[idx]
        ax = axes[idx]
        mosaic_data = mosaics[rgc_type]
        
        positions = mosaic_data['positions']
        rf_diameter = mosaic_data['rf_diameter']
        color = mosaic_data['color']
        
        # Plot visual field boundary
        rect = patches.Rectangle((0, 0), visual_field_size[1], visual_field_size[0], 
                               linewidth=2, edgecolor='black', facecolor='white')
        ax.add_patch(rect)
        
        # Plot receptive fields
        for x, y in positions:
            # Only show RFs that overlap with the visual field
            if (x >= -rf_diameter/2 and x <= visual_field_size[1] + rf_diameter/2 and
                y >= -rf_diameter/2 and y <= visual_field_size[0] + rf_diameter/2):
                
                if show_circles:
                    # Draw RF as circle
                    circle = patches.Circle((x, y), rf_diameter/2, 
                                          linewidth=1.5, edgecolor='black', 
                                          facecolor='none', alpha=0.8)
                    ax.add_patch(circle)
                
                # Mark RF center
                ax.plot(x, y, 'ko', markersize=2, alpha=0.7)
        
        # Formatting to match reference image
        ax.set_xlim(-10, visual_field_size[1] + 10)
        ax.set_ylim(-10, visual_field_size[0] + 10)
        ax.set_aspect('equal')
        ax.set_title(f'{display_name}', fontsize=14, fontweight='bold')
        
        # Remove axis ticks but keep the frame
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add cell count
        n_cells_in_field = np.sum([
            (0 <= pos[0] <= visual_field_size[1] and 0 <= pos[1] <= visual_field_size[0])
            for pos in positions
        ])
        
    
    # Add scale bar
    scale_length = 20  # pixels
    ax = axes[-1]
    ax.plot([visual_field_size[1] - 30, visual_field_size[1] - 30 + scale_length], 
            [20, 20], 'k-', linewidth=4)
    ax.text(visual_field_size[1] - 30 + scale_length/2, 30, f'{scale_length} pixels', 
            ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    return fig


def plot_simple_mosaic(mosaics, stimulus_frame=None, 
                       title="RGC Mosaic",
                       # --- stimulus display ---
                       show_stimulus: bool = True,
                       # --- frame/border ---
                       show_frame: bool = True,
                       frame_color: str = 'black',
                       frame_lw: float = 1.5,
                       # --- scale bar ---
                       scale_bar_length_um: float = None,
                       px_per_um: float = None,
                       scale_bar_length_pixels: float = None,
                       scale_bar_loc: str = 'lower right',
                       scale_bar_color: str = 'black',
                       scale_bar_height_frac: float = 0.02,
                       scale_bar_fontsize: int = 9,
                       scale_bar_pad_frac: float = 0.02,
                       # --- figure ---
                       figsize: tuple = (10, 8),
                       savepath: str = None):
    """
    SIMPLIFIED PLOT - Just colored RF circles with optional frame and scale bar
    
    Parameters
    ----------
    mosaics : dict
        Dictionary of mosaic data per cell type
    stimulus_frame : np.ndarray
        Stimulus image (used for dimensions, optionally displayed)
    title : str, optional
        Plot title (None for no title)
    show_stimulus : bool
        If True, overlay mosaic on stimulus. If False, use white background.
    show_frame : bool
        Whether to show border around plot
    scale_bar_length_um : float, optional
        Scale bar length in microns
    px_per_um : float, optional
        Pixels per micron conversion factor
    scale_bar_length_pixels : float, optional
        Scale bar length in pixels (alternative to um)
    savepath : str, optional
        Path to save figure
        
    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    if stimulus_frame is None:
        raise ValueError("stimulus_frame is required to determine visual field size")
    
    # Get dimensions from stimulus frame
    height, width = stimulus_frame.shape[:2]
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Show stimulus or white background
    if show_stimulus:
        ax.imshow(stimulus_frame, cmap='gray', extent=[0, width, height, 0])
    else:
        ax.set_facecolor('white')
    
    # Set consistent limits
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
        
    legend_handles = []
    # Plot each cell type - RF CIRCLES
    for cell_type, mosaic_data in mosaics.items():
        if mosaic_data['n_cells'] == 0:
            continue
            
        positions = mosaic_data['positions']
        rf_diameter = mosaic_data['rf_diameter']
        color = mosaic_data['color']
        
        # Create legend handle for this cell type
        legend_handle = Line2D([0], [0], color=color, linewidth=3, 
                              label=cell_type.replace('_', ' '))
        legend_handles.append(legend_handle)
        
        # Draw RF circles
        for i, (x, y) in enumerate(positions):
            circle = patches.Circle((x, y), rf_diameter/2, 
                                  edgecolor=color, facecolor='none', 
                                  linewidth=2)
            ax.add_patch(circle)
            
    ax.set_aspect('equal')
    
    if title:
        ax.set_title(title, fontsize=14)
    
    # --- Frame/border ---
    if show_frame:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(frame_lw)
            spine.set_color(frame_color)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.axis('off')
    
    # --- Scale bar ---
    if scale_bar_length_um is not None and px_per_um is not None:
        length_px = scale_bar_length_um * px_per_um
        label = f'{int(scale_bar_length_um)} µm'
    elif scale_bar_length_pixels is not None:
        length_px = scale_bar_length_pixels
        label = f'{int(scale_bar_length_pixels)} px'
    else:
        length_px = None
    
    if length_px is not None:
        scale_bar_frac = length_px / width
        scale_bar_frac = np.clip(scale_bar_frac, 0.005, 0.8)
        
        pad = scale_bar_pad_frac
        bar_h = scale_bar_height_frac
        
        # Position based on location
        loc = scale_bar_loc.lower()
        y0 = pad if 'lower' in loc else (1.0 - pad - bar_h)
        x0 = pad if 'left' in loc else (1.0 - pad - scale_bar_frac)
        
        # Draw bar
        rect = patches.Rectangle((x0, y0), scale_bar_frac, bar_h,
                                  transform=ax.transAxes, facecolor=scale_bar_color,
                                  edgecolor='none', clip_on=False, zorder=20)
        ax.add_patch(rect)
        
        # Label
        tx = x0 + scale_bar_frac / 2.0
        ty = y0 + bar_h + 0.5 * pad
        ax.text(tx, ty, label, transform=ax.transAxes, ha='center', va='bottom',
                fontsize=scale_bar_fontsize, color=scale_bar_color, zorder=21)
    
    plt.tight_layout()
    
    if savepath:
        fig.savefig(savepath, bbox_inches='tight')
    
    return fig, ax


def plot_mosaic_grid(
    mosaics: Dict[str, Dict],
    visual_field_size: Tuple[int, int],
    show_types: Optional[Sequence[str]] = None,
    cols: int = 2,
    stimulus_frame: Optional[np.ndarray] = None,
    show_rf: bool = True,
    show_centers: bool = True,
    annotate_ids: bool = False,
    figsize: Tuple[float, float] = (12, 8),
    savepath: Optional[str] = None,
    # --- scale bar ---
    scale_bar_length_pixels: Optional[float] = None,
    scale_bar_length_um: Optional[float] = None,
    px_per_um: Optional[float] = None,
    scale_bar_loc: str = 'lower right',
    scale_bar_color: str = 'black',
    scale_bar_height_frac: float = 0.02,
    scale_bar_fontsize: int = 9,
    scale_bar_pad_frac: float = 0.01,
    scale_bar_units: str = 'um',
    # --- highlight (single/per-type) ---
    highlight_type: Optional[str] = None,
    highlight_index: Optional[int] = None,
    highlight_indices: Optional[Dict[str, int]] = None,
    highlight_fill: bool = True,
    highlight_facecolor: str = 'yellow',
    highlight_edgecolor: str = 'black',
    highlight_alpha: float = 0.35,
    highlight_lw: float = 2.5,
    # --- NEW: highlight all RFs ---
    highlight_all: bool = False,
    highlight_all_fill: bool = False,
    highlight_all_facecolor: str = 'auto',
    highlight_all_edgecolor: str = 'auto',
    highlight_all_alpha: float = 0.9,
    highlight_all_lw: float = 4.0,
    rf_edge_lw: float = 1.8
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot RGC mosaics for all types with optional scale bar and highlights.
    Supports:
    - scale bar in µm or mm (via scale_bar_length_um and px_per_um)
    - single or per-type highlighted cell
    - optional highlight for all receptive fields
    """

    types = list(mosaics.keys()) if show_types is None else [t for t in show_types if t in mosaics]
    if not types:
        raise ValueError("No mosaic types to plot.")

    # normalize highlight selection
    sel_map = {}
    if highlight_indices is not None:
        sel_map.update(highlight_indices)
    if highlight_type is not None and highlight_index is not None:
        sel_map[highlight_type] = highlight_index

    rows = int(np.ceil(len(types) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(rows, cols)
    idx = 0
    H, W = visual_field_size

    default_frac = 0.10  # fallback scale bar length fraction

    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            if idx >= len(types):
                ax.axis('off')
                idx += 1
                continue

            typ = types[idx]
            data = mosaics[typ]
            positions = np.asarray(data.get('positions', np.zeros((0, 2))))
            rf_diam = float(data.get('rf_diameter', 0.0))
            color =DEFAULT_COLORS.get(typ, data.get('color', 'k'))

            # background
            if stimulus_frame is not None:
                ax.imshow(stimulus_frame, cmap='gray', origin='upper', extent=[0, W, H, 0])
            else:
                ax.set_xlim(0, W)
                ax.set_ylim(H, 0)

            # RFs
            if show_rf and rf_diam > 0 and positions.size > 0:
                rad = rf_diam / 2.0
                for (x, y) in positions:
                    if (x < -rf_diam or x > W + rf_diam) or (y < -rf_diam or y > H + rf_diam):
                        continue

                    # base RF outline
                    ax.add_patch(
                        patches.Circle((x, y), rad, edgecolor=color, facecolor='none',
                                       alpha=0.7, lw=rf_edge_lw, zorder=3)
                    )

                    # highlight all
                    if highlight_all:
                        face_all = (
                            color if (highlight_all_facecolor == 'auto' and highlight_all_fill)
                            else ('none' if not highlight_all_fill else highlight_all_facecolor)
                        )
                        edge_all = color if (highlight_all_edgecolor == 'auto') else highlight_all_edgecolor
                        ax.add_patch(
                            patches.Circle(
                                (x, y), rad,
                                facecolor=face_all,
                                edgecolor=edge_all,
                                alpha=highlight_all_alpha if highlight_all_fill else max(0.8, highlight_all_alpha),
                                lw=highlight_all_lw,
                                zorder=9
                            )
                        )

                # single highlight
                if typ in sel_map:
                    i_sel = sel_map[typ]
                    if 0 <= i_sel < len(positions):
                        hx, hy = positions[i_sel]
                        face_col = color if (highlight_facecolor == 'auto' and highlight_fill) else (
                            'none' if not highlight_fill else highlight_facecolor
                        )
                        edge_col = color if (highlight_edgecolor == 'auto') else highlight_edgecolor
                        ax.add_patch(
                            patches.Circle(
                                (hx, hy), rad,
                                facecolor=face_col,
                                edgecolor=edge_col,
                                alpha=highlight_alpha if highlight_fill else 1.0,
                                lw=highlight_lw,
                                zorder=10
                            )
                        )

            # annotate IDs
            if annotate_ids and positions.size > 0:
                for i, (x, y) in enumerate(positions):
                    ax.text(x + 1.5, y + 1.5, str(i + 1), color=color, fontsize=7, zorder=15)

            ax.set_title(f"{typ.replace('_',' ')}", color=color, fontsize=11, fontweight='bold')
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_aspect('equal')

            # --- scale bar ---
            length_px = None
            if scale_bar_length_pixels is not None:
                length_px = float(scale_bar_length_pixels)
                length_um_val = (length_px / px_per_um) if (px_per_um is not None) else None
            elif (scale_bar_length_um is not None) and (px_per_um is not None):
                length_px = float(scale_bar_length_um) * float(px_per_um)
                length_um_val = float(scale_bar_length_um)
            else:
                length_frac = default_frac
                length_px = length_frac * float(W)
                length_um_val = (length_px / px_per_um) if (px_per_um is not None) else None

            length_frac = float(length_px) / float(W) if W > 0 else default_frac
            length_frac = np.clip(length_frac, 0.005, 0.8)

            # label
            label = (
                f"{int(round(length_um_val))} µm"
                if (scale_bar_units == 'um' and length_um_val is not None)
                else f"{int(round(length_px))} px"
            )

            loc = scale_bar_loc.lower()
            pad = scale_bar_pad_frac
            bar_h = scale_bar_height_frac
            y0 = pad if ('lower' in loc) else (1.0 - pad - bar_h)
            x0 = pad if ('left' in loc) else (1.0 - pad - length_frac)

            rect = patches.Rectangle((x0, y0), length_frac, bar_h, transform=ax.transAxes,
                                     facecolor=scale_bar_color, edgecolor='none', clip_on=False, zorder=20)
            ax.add_patch(rect)
            tx = x0 + length_frac / 2.0
            ty = y0 + bar_h + 0.5 * pad
            ax.text(tx, ty, label, transform=ax.transAxes, ha='center', va='bottom',
                    fontsize=scale_bar_fontsize, color=scale_bar_color, zorder=21)

            idx += 1

    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
    return fig, axes



#%% rectangular mosaic 


@dataclass
class RectLattice:
    """Rectangular lattice parameters"""
    H: int              # field height
    W: int              # field width
    dx: float           # horizontal spacing (anchor)
    dy: float           # vertical spacing (anchor)
    xmin: float         # left margin
    ymin: float         # top margin
    n_rows: int         # number of rows
    n_cols: int         # number of columns


def build_rectangular_lattice(H, W, dx_anchor, dy_anchor=None, keep_inbounds_radius=0.0):
    """
    Build a rectangular (grid) lattice that tiles the visual field.
    
    Parameters:
    -----------
    H, W : int
        Visual field dimensions
    dx_anchor : float
        Horizontal spacing for the master lattice
    dy_anchor : float, optional
        Vertical spacing (if None, uses dx_anchor for square grid)
    keep_inbounds_radius : float
        Margin from edges to keep RFs fully within bounds
        
    Returns:
    --------
    L : RectLattice
        Lattice parameters
    XY : ndarray (N, 2)
        All lattice point coordinates
    rows, cols : ndarray (N,)
        Row and column indices for each point
    """
    r = float(keep_inbounds_radius)
    dx = float(dx_anchor)
    dy = float(dy_anchor) if dy_anchor is not None else dx
    
    # Define the region where RF centers can be placed
    xmin, xmax = r, W - r
    ymin, ymax = r, H - r
    
    # Calculate number of rows and columns
    n_cols = int(np.floor((xmax - xmin) / dx)) + 1
    n_rows = int(np.floor((ymax - ymin) / dy)) + 1
    
    # Generate grid coordinates
    xs = xmin + dx * np.arange(n_cols)
    ys = ymin + dy * np.arange(n_rows)
    
    # Create meshgrid
    X, Y = np.meshgrid(xs, ys)
    XY = np.column_stack([X.ravel(), Y.ravel()])
    
    # Create row and column indices
    rows_grid, cols_grid = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing='ij')
    rows = rows_grid.ravel()
    cols = cols_grid.ravel()
    
    L = RectLattice(
        H=H, W=W, dx=dx, dy=dy,
        xmin=xmin, ymin=ymin,
        n_rows=n_rows, n_cols=n_cols
    )
    
    return L, XY, rows, cols


def subsample_rectangular_lattice(
    L, XY, rows, cols,
    rf_diameter, 
    stride_x=1, 
    stride_y=None,
    offset_x=0.0, 
    offset_y=0.0
):
    """
    Subsample from rectangular master lattice with adjustable spacing and offsets.
    
    Parameters:
    -----------
    L : RectLattice
        Master lattice parameters
    XY : ndarray (N, 2)
        Master lattice coordinates
    rows, cols : ndarray (N,)
        Row and column indices
    rf_diameter : float
        Receptive field diameter (for boundary checking)
    stride_x : int
        Subsampling stride in x (columns)
    stride_y : int, optional
        Subsampling stride in y (rows). If None, uses stride_x
    offset_x : float
        Horizontal offset in units of master lattice spacing (can be fractional)
    offset_y : float
        Vertical offset in units of master lattice spacing (can be fractional)
        
    Returns:
    --------
    XY_sub : ndarray (M, 2)
        Subsampled coordinates, all within bounds
    """
    if stride_y is None:
        stride_y = stride_x
    
    stride_x = max(1, int(stride_x))
    stride_y = max(1, int(stride_y))
    
    # Select every stride_x-th column and stride_y-th row
    keep_col = (cols % stride_x == 0)
    keep_row = (rows % stride_y == 0)
    mask = keep_col & keep_row
    
    if not np.any(mask):
        return np.empty((0, 2), dtype=float)
    
    XY_sub = XY[mask].copy()
    
    # Apply offsets (in units of master lattice spacing)
    XY_sub[:, 0] += offset_x * L.dx
    XY_sub[:, 1] += offset_y * L.dy
    
    # Keep only points where entire RF is within field bounds
    r = rf_diameter / 2.0
    eps = 1e-9
    xmin, xmax = r, L.W - r
    ymin, ymax = r, L.H - r
    
    inbounds = (
        (XY_sub[:, 0] >= xmin - eps) & (XY_sub[:, 0] <= xmax + eps) &
        (XY_sub[:, 1] >= ymin - eps) & (XY_sub[:, 1] <= ymax + eps)
    )
    
    return XY_sub[inbounds]


def _select_subset(XY, n_cells, method, field_size):
    """
    Select a subset of n_cells from XY using specified method.
    
    Parameters:
    -----------
    XY : ndarray (N, 2)
        All available positions
    n_cells : int or None
        Number of cells to select (None = use all)
    method : str
        Selection method: 'center_first', 'edge_first', 'random', 'grid_order'
    field_size : tuple (H, W)
        Visual field dimensions
        
    Returns:
    --------
    XY_selected : ndarray (n_cells, 2)
        Selected positions
    """
    if n_cells is None or len(XY) <= n_cells:
        return XY
    
    H, W = field_size
    cx, cy = W / 2.0, H / 2.0
    
    if method == 'grid_order':
        return XY[:n_cells]
    
    # Calculate distance from center
    dist = np.sqrt((XY[:, 0] - cx)**2 + (XY[:, 1] - cy)**2)
    
    if method == 'edge_first':
        order = np.argsort(-dist)  # Farthest first
    elif method == 'random':
        order = np.random.permutation(len(XY))
    else:  # 'center_first' (default)
        order = np.argsort(dist)   # Closest first
    
    return XY[order[:n_cells]]


# =============== High-level API ===============

def build_rectangular_mosaics(
    visual_field_size,
    per_type_config,
    anchor_spacing=None
):
    """
    Build multiple rectangular mosaics with adjustable spacing via coverage factor.
    
    Parameters:
    -----------
    visual_field_size : tuple (H, W)
        Visual field dimensions in pixels
    per_type_config : dict
        Configuration for each cell type:
        {
            "ON_Parasol": {
                "rf_diameter": 10.0,      # RF diameter in pixels
                "coverage_factor": 1.0,   # spacing = rf_diameter / coverage_factor
                "color": "red",           # plotting color
                "offset_x": 0.0,          # horizontal offset (in master lattice units)
                "offset_y": 0.0,          # vertical offset (in master lattice units)
                "n_cells": None,          # optional: limit number of cells
                "selection_method": "center_first"  # how to select subset
            },
            ...
        }
    anchor_spacing : float, optional
        Master lattice spacing. If None, uses GCD of all spacings.
        
    Returns:
    --------
    mosaics : dict
        Dictionary mapping cell type names to mosaic data:
        {
            "positions": ndarray (N, 2),
            "rf_diameter": float,
            "color": str,
            "n_cells": int,
            "spacing": float  # actual spacing used
        }
    """
    H, W = visual_field_size
    
    # Calculate effective spacing for each type: spacing = rf_diameter / coverage_factor
    spacings = []
    for cfg in per_type_config.values():
        rf_diam = float(cfg["rf_diameter"])
        cov_factor = float(cfg.get("coverage_factor", 1.0))
        effective_spacing = rf_diam / max(cov_factor, 1e-9)
        spacings.append(effective_spacing)
    
    # Determine anchor spacing (master lattice spacing)
    if anchor_spacing is None:
        # Use GCD-like approach: find spacing that divides all requested spacings
        anchor_spacing = _find_anchor_spacing(spacings, quantum=0.1)
    
    dx_anchor = float(anchor_spacing)
    
    # Build master rectangular lattice
    # FIXED: Use minimal margin for master lattice - each type applies its own boundary
    # This allows smaller RFs to reach edges while larger RFs are still bounded
    L, XY_master, rows, cols = build_rectangular_lattice(
        H, W, dx_anchor, dy_anchor=dx_anchor,
        keep_inbounds_radius=0.0  # No margin - let each type handle its own boundaries
    )
    
    if L is None or len(XY_master) == 0:
        return {}
    
    # Build mosaic for each cell type
    mosaics = {}
    for cell_type, cfg in per_type_config.items():
        rf_diam = float(cfg["rf_diameter"])
        cov_factor = float(cfg.get("coverage_factor", 1.0))
        effective_spacing = rf_diam / max(cov_factor, 1e-9)
        
        # Calculate stride: how many master lattice steps per RF spacing
        stride = int(np.round(effective_spacing / dx_anchor))
        stride = max(1, stride)  # Ensure at least stride of 1
        
        # Get offsets
        offset_x = float(cfg.get("offset_x", 0.0))
        offset_y = float(cfg.get("offset_y", 0.0))
        
        # Subsample from master lattice
        XY_full = subsample_rectangular_lattice(
            L, XY_master, rows, cols,
            rf_diameter=rf_diam,
            stride_x=stride,
            stride_y=stride,
            offset_x=offset_x,
            offset_y=offset_y
        )
        
        # Optionally select subset
        n_cells = cfg.get("n_cells", None)
        method = cfg.get("selection_method", "center_first")
        XY_final = _select_subset(XY_full, n_cells, method, (H, W))
        
        # Store mosaic data
        mosaics[cell_type] = {
            "positions": XY_final,
            "rf_diameter": rf_diam,
            "color": cfg.get("color", "black"),
            "n_cells": len(XY_final),
            "spacing": stride * dx_anchor,       # ADD THIS LINE
            "coverage_factor": cov_factor        # ADD THIS LINE
        }
            
    return mosaics


def _find_anchor_spacing(spacings, quantum=0.1):
    """
    Find a good anchor spacing that approximately divides all requested spacings.
    Uses a GCD-like approach with quantization.
    """
    # Quantize spacings
    quantized = np.round(np.array(spacings) / quantum).astype(int)
    gcd_val = np.gcd.reduce(quantized)
    return max(1, gcd_val) * quantum


def make_cfg_from_bio_params(bio_params):
    """
    Convert biological parameters to configuration dictionary.
    
    Parameters:
    -----------
    bio_params : dict
        {
            "ON_Parasol": {
                "rf_diameter": 10.0,
                "coverage_factor": 1.0,
                "color": "red",
                "offset_x": 0.0,
                "offset_y": 0.0,
                "n_cells": None,
                "selection_method": "center_first"
            },
            ...
        }
    
    Returns:
    --------
    cfg : dict
        Configuration ready for build_rectangular_mosaics()
    """
    cfg = {}
    for cell_type, params in bio_params.items():
        cfg[cell_type] = {
            "rf_diameter": float(params["rf_diameter"]),
            "coverage_factor": float(params.get("coverage_factor", 1.0)),
            "color": params.get("color", "black"),
            "offset_x": float(params.get("offset_x", 0.0)),
            "offset_y": float(params.get("offset_y", 0.0)),
            "n_cells": params.get("n_cells", None),
            "selection_method": params.get("selection_method", "center_first")
        }
    return cfg


# =============== Visualization ===============

def plot_rectangular_mosaics(mosaics, visual_field_size, title="Rectangular RGC Mosaic"):
    """
    Plot overlaid rectangular mosaics.
    
    Parameters:
    -----------
    mosaics : dict
        Output from build_rectangular_mosaics()
    visual_field_size : tuple (H, W)
        Visual field dimensions
    title : str
        Plot title
    """
    H, W = visual_field_size
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)  # Flip y-axis
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Draw field boundary
    ax.plot([0, W, W, 0, 0], [0, 0, H, H, 0], 'k-', lw=2, label='Field boundary')
    
    # Plot each mosaic
    from matplotlib.lines import Line2D
    handles = []
    
    for cell_type, data in mosaics.items():
        positions = data["positions"]
        rf_diam = data["rf_diameter"]
        color = data["color"]
        n_cells = data["n_cells"]
        spacing = data["spacing"]
        cov = data["coverage_factor"]
        
        # Draw RFs as circles
        for x, y in positions:
            circle = patches.Circle(
                (x, y), rf_diam / 2.0,
                edgecolor=color, facecolor='none', 
                linewidth=1.5, alpha=0.7
            )
            ax.add_patch(circle)
        
        # Create legend entry
        label = f"{cell_type.replace('_', ' ')} (n={n_cells}, d={rf_diam:.1f}, s={spacing:.1f}, c={cov:.2f})"
        handles.append(Line2D([0], [0], color=color, lw=3, label=label))
    
    ax.legend(handles=handles, loc="upper right", fontsize=9)
    ax.set_xlabel("X (pixels)", fontsize=12)
    ax.set_ylabel("Y (pixels)", fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig


def analyze_mosaic_spacing(mosaics):
    """
    Analyze and print spacing statistics for each mosaic.
    """
    from scipy.spatial.distance import cdist
    
    print("\n" + "="*70)
    print("MOSAIC SPACING ANALYSIS")
    print("="*70)
    
    for cell_type, data in mosaics.items():
        positions = data["positions"]
        rf_diam = data["rf_diameter"]
        expected_spacing = data["spacing"]
        cov_factor = data["coverage_factor"]
        
        if len(positions) < 2:
            print(f"\n{cell_type}: Not enough cells for analysis")
            continue
        
        # Calculate nearest neighbor distances
        dists = cdist(positions, positions)
        np.fill_diagonal(dists, np.inf)
        min_dists = dists.min(axis=1)
        
        print(f"\n{cell_type}:")
        print(f"  RF Diameter:        {rf_diam:.2f} px")
        print(f"  Coverage Factor:    {cov_factor:.2f}")
        print(f"  Expected Spacing:   {expected_spacing:.2f} px")
        print(f"  Actual Mean Spacing: {min_dists.mean():.2f} ± {min_dists.std():.2f} px")
        print(f"  Min Spacing:        {min_dists.min():.2f} px")
        print(f"  Max Spacing:        {min_dists.max():.2f} px")
        
        # Check coverage type
        if cov_factor > 1.0:
            overlap = rf_diam - expected_spacing
            print(f"  → OVERLAP:          {overlap:.2f} px ({overlap/rf_diam*100:.1f}% of diameter)")
        elif cov_factor < 1.0:
            gap = expected_spacing - rf_diam
            print(f"  → GAP:              {gap:.2f} px ({gap/expected_spacing*100:.1f}% of spacing)")
        else:
            print(f"  → PERFECT TILING    (coverage = 1.0)")
    
    print("="*70 + "\n")


#%% 11. Spatial Filter for a mosaic 


def create_positioned_spatial_filter(center_size, surround_size, center_strength, surround_strength, 
                                   visual_field_shape, cell_center):
    """
    Create spatial filter at specific position using YOUR existing functions
    
    Parameters:
    -----------
    center_size, surround_size : float
        Sigma values for center and surround
    center_strength, surround_strength : float  
        Strength values for center and surround
    visual_field_shape : tuple
        (height, width) of the visual field
    cell_center : tuple
        (x, y) position where to center the filter
    """
    height, width = visual_field_shape
    center_x, center_y = cell_center
    
    # Use existing create_2d_gaussian with center positioning
    center_gaussian = create_2d_gaussian(width, height, center_size, center_x, center_y)
    surround_gaussian = create_2d_gaussian(width, height,  surround_size, center_x, center_y)
    
    # Crop to visual field height if needed
    if center_gaussian.shape[0] > height:
        center_gaussian = center_gaussian[:height, :]
        surround_gaussian = surround_gaussian[:height, :]
    
    # Use existing normalization and combination logic
    center_gaussian = center_gaussian / np.sum(center_gaussian)
    surround_gaussian = surround_gaussian / np.sum(surround_gaussian) 
    spatial_filter = center_strength * center_gaussian + surround_strength * surround_gaussian
    
    return spatial_filter


def create_overlaid_spatial_filters_simple(overlaid_mosaic, movie_shape=(x,y)):
    """
    Create overlaid spatial filters using YOUR existing functions
    """
    movie_height, movie_width = movie_shape
    
    combined_filter = np.zeros((movie_height, movie_width))
    individual_filters = {}
    
    # print("Creating positioned spatial filters using your existing functions...")
    
    for rgc_type, mosaic_data in overlaid_mosaic.items():
        if mosaic_data['n_cells'] == 0:
            continue
            
        positions = mosaic_data['positions']
        params = RGC_PARAMS[rgc_type]
        individual_filters[rgc_type] = []
        
        # print(f"\n{rgc_type} ({len(positions)} cells):")
        
        for i, position in enumerate(positions):
            center_x, center_y = position[0], position[1]
            # print(f"  Cell {i+1} at ({center_x:.1f}, {center_y:.1f})")
            
            # Create positioned spatial filter 
            cell_filter = create_positioned_spatial_filter(
                params['center_size'],
                params['surround_size'], 
                params['center_strength'],
                params['surround_strength'],
                (movie_height, movie_width),
                (center_x, center_y)
            )
            
            combined_filter += cell_filter
            
            individual_filters[rgc_type].append({
                'filter': cell_filter,
                'center': (center_x, center_y),
                'cell_id': i + 1
            })
    
    return combined_filter, individual_filters



#%% 12. Get responses of a mosaic 


def compute_lnp_for_mosaic_final(stim_windowed, overlaid_mosaic, dt=8.3):
    """
    Compute LNP responses using YOUR existing functions
    """
    responses = {}
    
    # Create positioned spatial filters 
    # print("Creating positioned spatial filters...")
    combined_filter, individual_filters = create_overlaid_spatial_filters_simple(overlaid_mosaic, stim_windowed.shape[2:])
    
    total_cells = sum(len(filters) for filters in individual_filters.values())
    # print(f"\nComputing LNP responses for {total_cells} cells...")
    
    with tqdm(total=total_cells, desc="Processing cells") as pbar:
        for rgc_type, cell_filters in individual_filters.items():
            if len(cell_filters) == 0:
                responses[rgc_type] = []
                continue
                
            responses[rgc_type] = []
            params = RGC_PARAMS[rgc_type]
            
            # Create temporal filter 
            temporal_filter = create_temporal_filter(
                peak1_ms=params['peak1_ms'], 
                peak2_ms=params['peak2_ms'],
                width1_ms=params['width1_ms'], 
                width2_ms=params['width2_ms'],
                amp1=params['amp1'] ,    # <--- ADD THIS LINE
                amp2=params['amp2'] ,     # <--- ADD THIS LINE
                smooth_onset=True, 
                cell_type=rgc_type
            )
                                        
            for cell_data in cell_filters:
                spatial_filter = cell_data['filter']
                cell_center = cell_data['center']
                cell_id = cell_data['cell_id']
                
                # Use YOUR existing compute_lnp function
                generator_signal, firing_rate, spikes, spike_times = compute_lnp(
                    stim_windowed, spatial_filter, temporal_filter, 
                    params['alpha'], params['beta'], params['gamma'], dt
                )
                
                response = {
                    'generator_signal': generator_signal,
                    'firing_rate': firing_rate,
                    'spikes': spikes,
                    'spike_times':spike_times,
                    'spatial_filter': spatial_filter,
                    'cell_center': cell_center,
                    'rgc_type': rgc_type,
                    'cell_id': cell_id
                }
                
                responses[rgc_type].append(response)
                pbar.update(1)
    
    return responses




#%%15. Plot generator signals, firing rate and spikes 

def plot_final_results(overlaid_mosaic, responses, movie_shape=(75, 100), movie=None):
    """
    Enhanced plot function showing all 4 RGC types and responses from random cells across all types
    """
    
    if movie is not None:
        samples, n_frames, movie_height, movie_width = movie.shape
        movie_shape = (movie_height, movie_width)
    else:
        movie_height, movie_width = movie_shape
        n_frames = 72  # default fallback
    
    # Color mapping
    type_colors = {
        'ON_Parasol': 'red', 
        'OFF_Parasol': 'orange', 
        'ON_Midget': 'green', 
        'OFF_Midget': 'blue'
    }
    
    # Create the spatial filters
    combined_filter, individual_filters = create_overlaid_spatial_filters_simple(overlaid_mosaic, movie_shape)
    
    # Plot 1: Spatial filters for all 4 RGC types
    fig1, axes1 = plt.subplots(1, 5, figsize=(25, 5))
    
    # Combined filters (all types)
    ax = axes1[0]
    vmax = np.max(np.abs(combined_filter))
    im1 = ax.imshow(combined_filter, cmap='gray', vmin=-vmax, vmax=vmax,
                    extent=[0, movie_shape[1], movie_shape[0], 0])
    ax.set_title('All Spatial Filters Combined\n(75×100 pixels)', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=ax, shrink=0.8)
    
    # Individual RGC types
    rgc_types = ['ON_Parasol', 'OFF_Parasol', 'ON_Midget', 'OFF_Midget']
    display_names = ['ON Parasol', 'OFF Parasol', 'ON Midget', 'OFF Midget']
    colors = ['red', 'orange', 'green', 'blue']
    
    for i, (rgc_type, color, display_name) in enumerate(zip(rgc_types, colors, display_names)):
        ax = axes1[i+1]
        
        type_filter = np.zeros(movie_shape)
        for cell_data in individual_filters.get(rgc_type, []):
            type_filter += cell_data['filter']
        
        if np.max(np.abs(type_filter)) > 0:
            vmax_type = np.max(np.abs(type_filter))
            im = ax.imshow(type_filter, cmap='gray', vmin=-vmax_type, vmax=vmax_type,
                          extent=[0, movie_shape[1], movie_shape[0], 0])
            ax.set_title(f'{rgc_type}', fontsize=12, color=color, fontweight='bold')
            plt.colorbar(im, ax=ax, shrink=0.8)
        else:
            ax.text(0.5, 0.5, f'No {rgc_type}\ncells', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12, color=color)
            ax.set_xlim(0, movie_shape[1])
            ax.set_ylim(movie_shape[0], 0)
    
    plt.tight_layout()
    plt.show()
    
    # Plot 2: LNP Responses
    fig2, axes2 = plt.subplots(3, 8, figsize=(20, 8))
    
    # Select 2 cells from each type
    selected_cells = []
    
    for rgc_type in rgc_types:
        if rgc_type in responses and len(responses[rgc_type]) > 0:
            type_cells = responses[rgc_type]
            if len(type_cells) >= 2:
                selected_from_type = random.sample(type_cells, 2)
            else:
                selected_from_type = type_cells
            selected_cells.extend(selected_from_type)
    
    # Organize by type
    organized_cells = []
    for rgc_type in rgc_types:
        type_cells = [cell for cell in selected_cells if cell['rgc_type'] == rgc_type]
        organized_cells.extend(type_cells)
    
    selected_cells = organized_cells[:8]
    
    # Get actual response length from first cell
    if len(selected_cells) > 0:
        actual_n_frames = len(selected_cells[0]['generator_signal'])
        time_axis = np.arange(actual_n_frames)
    else:
        actual_n_frames = n_frames
        time_axis = np.arange(n_frames)
    
    for cell_idx, response in enumerate(selected_cells):
        if cell_idx >= 8:
            break
            
        rgc_type = response['rgc_type']
        color = type_colors[rgc_type]
        
        generator = response['generator_signal']
        firing_rate = response['firing_rate']
        spikes = response['spikes']
        
        # Use actual length of data
        response_time_axis = np.arange(len(generator))
        
        # Generator signal
        axes2[0, cell_idx].plot(response_time_axis, generator, color=color, linewidth=2)
        axes2[0, cell_idx].set_title(f'{rgc_type}\nCell {response["cell_id"]}', 
                                    fontsize=10, color=color, fontweight='bold')
        if cell_idx == 0:
            axes2[0, cell_idx].set_ylabel('Generator Signal', fontweight='bold')
        
        # Firing rate
        axes2[1, cell_idx].plot(response_time_axis, firing_rate, color=color, linewidth=2)
        if cell_idx == 0:
            axes2[1, cell_idx].set_ylabel('Firing Rate (Hz)', fontweight='bold')
        
        # Spikes
        spike_times = response_time_axis[spikes == 1]
        if len(spike_times) > 0:
            axes2[2, cell_idx].vlines(spike_times, 0, 1, colors=color, linewidth=2, alpha=0.8)
        if cell_idx == 0:
            axes2[2, cell_idx].set_ylabel('Spikes', fontweight='bold')
        axes2[2, cell_idx].set_xlabel('Time (frames)')
        axes2[2, cell_idx].set_ylim(-0.1, 1.1)
        axes2[2, cell_idx].set_yticks([0, 1])
        
        # Add cell position info
        center_x, center_y = response['cell_center']
        axes2[2, cell_idx].text(0.5, -0.3, f'({center_x:.1f}, {center_y:.1f})', 
                               transform=axes2[2, cell_idx].transAxes, 
                               ha='center', fontsize=8, color=color)
    
    # Fill empty subplots
    for cell_idx in range(len(selected_cells), 8):
        for row in range(3):
            axes2[row, cell_idx].text(0.5, 0.5, 'No cell',
                                      transform=axes2[row, cell_idx].transAxes,
                                      ha='center', va='center', fontsize=12, alpha=0.5)
            axes2[row, cell_idx].set_xlim(0, actual_n_frames-1)
            axes2[row, cell_idx].set_xlabel('Time (frames)')
    
    plt.suptitle('LNP Responses from Random Cells Across All RGC Types', fontsize=14, fontweight='bold')
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], color=color, lw=3, label=rgc_type) 
                      for rgc_type, color in type_colors.items()]
    fig2.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.95))
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    # print("\n" + "="*60)
    # print("MOSAIC SUMMARY")
    # print("="*60)
    
    for rgc_type in rgc_types:
        if rgc_type in overlaid_mosaic:
            n_cells = overlaid_mosaic[rgc_type]['n_cells']
            color = type_colors[rgc_type]
            print(f"{rgc_type:12} ({color:6}): {n_cells:3d} cells")
    
    return fig1, fig2




#%% 16. generator vs firig rate 



def overlay_fr_vs_gen_on_nonlinearity(nonlin_axes, responses, RGC_PARAMS, apply_nonlinearity,
                                      s=8, alpha=0.35, replot_sigmoid=True):
    """
    nonlin_axes: dict { 'ON_Parasol': Axes, ... } returned by plot_rgc_components
    responses:   dict { typ: [ { 'generator_signal':..., 'firing_rate':... }, ... ] }
    """
    for rgc_type, ax in nonlin_axes.items():
        if rgc_type not in responses or len(responses[rgc_type]) == 0:
            continue

        # 2a) scatter all cells (blue)
        for cell in responses[rgc_type]:
            g = np.asarray(cell['generator_signal']).ravel()
            fr = np.asarray(cell['firing_rate']).ravel()
            if g.size == 0 or g.size != fr.size:
                continue
            ax.scatter(g, fr, s=s, alpha=alpha, color='blue', zorder=2)

        # 2b) (optional) redraw analytic sigmoid on top in black so it’s visible
        if replot_sigmoid:
            params = RGC_PARAMS[rgc_type]
            # pick a generator range that covers both the axis and the data
            xlo, xhi = ax.get_xlim()
            all_g = np.hstack([np.asarray(c['generator_signal']).ravel()
                               for c in responses[rgc_type] if c['generator_signal'] is not None])
            if all_g.size:
                xlo = min(xlo, np.nanmin(all_g))
                xhi = max(xhi, np.nanmax(all_g))
            gx = np.linspace(xlo, xhi, 500)
            gy = apply_nonlinearity(gx, params['alpha'], params['beta'], params['gamma'])
            ax.plot(gx, gy, color='black', lw=2.5, zorder=3)
            ax.legend(frameon=False, loc='best')

        ax.figure.canvas.draw_idle()





#%% 17. Raster plots 

#Generate multiple independent Poisson spike trains for each cell
def compute_lnp_with_multiple_trials(movie, overlaid_mosaic, n_trials=10, dt=8.3):
    """
    Compute LNP with multiple Poisson spike realizations
    """
    responses_all_trials = []
    
    for trial in range(n_trials):
        print(f"Computing trial {trial + 1}/{n_trials}...")
        # Each call generates different Poisson spikes
        responses_trial = compute_lnp_for_mosaic_final(movie, overlaid_mosaic, dt=dt)
        responses_all_trials.append(responses_trial)
    
    return responses_all_trials




# PSTH 
def compute_psth_from_multiple_trials(responses_all_trials, bin_width_ms=50, 
                                     total_duration_ms=cell_memory_ms, dt=8.3):
    """
    Compute PSTH averaged across trials AND cells
    with spike times generated from Poisson process.
    """
    psths = {}
    rgc_types = ['ON_Parasol', 'OFF_Parasol', 'ON_Midget', 'OFF_Midget']
    
    for rgc_type in rgc_types:
        all_spike_trains = []
        n_cells = 0
        n_trials = len(responses_all_trials)
        
        # Get number of unique cells from first trial
        if n_trials > 0 and rgc_type in responses_all_trials[0]:
            n_cells = len(responses_all_trials[0][rgc_type])
        
        # Collect spike times from all trials and all cells
        for trial_responses in responses_all_trials:
            if rgc_type not in trial_responses:
                continue
            for cell_response in trial_responses[rgc_type]:
                spike_times = cell_response['spike_times']  # Use spike times here
                all_spike_trains.append(spike_times)
        
        if len(all_spike_trains) == 0:
            continue
        
        # Create histogram
        bins = np.arange(0, total_duration_ms + bin_width_ms, bin_width_ms)
        bin_centers = bins[:-1] + bin_width_ms / 2
        
        histograms = []
        for spike_times in all_spike_trains:
            counts, _ = np.histogram(spike_times, bins=bins)
            firing_rate = counts / (bin_width_ms / 1000.0)  # Convert to spikes per second
            histograms.append(firing_rate)
        
        histograms = np.array(histograms)
        psths[rgc_type] = {
            'time_bins': bin_centers,
            'firing_rate': np.mean(histograms, axis=0),
            'sem': np.std(histograms, axis=0) / np.sqrt(len(histograms)),
            'n_cells': n_cells,  # Actual number of cells
            'n_trials': n_trials,  # Number of trials
            'n_samples': len(all_spike_trains)  # Total samples (cells × trials)
        }
    
    return psths



# Raster Plot
def create_raster_and_psth_unified(
    responses_all_trials,
    rgc_type='ON_Parasol',
    *,
    cell_index=0,               # <— pick the cell you want (0-based)
    dt=8.3,                     # ms/frame
    axis_units='ms',            # 'ms' or 'frames'
    bin_width=50,               # if 'ms': ms; if 'frames': frames
    expand_counts_for_raster=True,
    jitter_within_frame=True
):
    """
    Create raster + PSTH for ONE RGC type across trials, for a SELECTED cell index.
    """
    all_trains = []
    n_frames_inferred = None
    cell_id_for_title = None

    for trial_responses in responses_all_trials:
        if rgc_type not in trial_responses:
            continue
        cells = trial_responses[rgc_type]
        if not cells or cell_index >= len(cells):
            continue

        cell_resp = cells[cell_index]
        cell_id_for_title = cell_resp.get('cell_id', cell_index)

        # Exact spike times (ms) path
        if 'spike_times' in cell_resp and cell_resp['spike_times'] is not None:
            spike_times_ms = np.asarray(cell_resp['spike_times'], dtype=float)
            if axis_units == 'ms':
                all_trains.append(np.sort(spike_times_ms))
            else:  # frames
                all_trains.append((spike_times_ms / dt).astype(int))

        # Counts-per-frame path
        elif 'spikes' in cell_resp and cell_resp['spikes'] is not None:
            counts = np.asarray(cell_resp['spikes'], dtype=int)
            n_frames_inferred = max(n_frames_inferred or 0, counts.shape[0])

            if axis_units == 'ms':
                if expand_counts_for_raster:
                    times = []
                    for f, k in enumerate(counts):
                        if k > 0:
                            start = f * dt; end = (f + 1) * dt
                            if jitter_within_frame:
                                times.extend(np.random.uniform(start, end, int(k)))
                            else:
                                times.extend([start + 0.5 * dt] * int(k))
                    all_trains.append(np.sort(np.array(times, dtype=float)))
                else:
                    frames = np.where(counts > 0)[0]
                    all_trains.append((frames * dt).astype(float))
            else:  # frames axis
                if expand_counts_for_raster:
                    frames = []
                    for f, k in enumerate(counts):
                        if k > 0:
                            frames.extend([f] * int(k))
                    all_trains.append(np.array(frames, dtype=int))
                else:
                    all_trains.append(np.where(counts > 0)[0].astype(int))
        else:
            continue

    if len(all_trains) == 0:
        print(f"No data for {rgc_type} at cell_index={cell_index}")
        return

    # ---- Axis range & bins ----
    if axis_units == 'ms':
        if n_frames_inferred is not None:
            xlim_max = n_frames_inferred * dt
        else:
            xlim_max = max((np.max(st) if len(st) else 0) for st in all_trains)
        bins = np.arange(0, xlim_max + bin_width, bin_width)
        bin_centers = bins[:-1] + bin_width / 2.0
        bin_width_s = bin_width / 1000.0
    else:
        if n_frames_inferred is None:
            max_frame = max((np.max(st) if len(st) else 0) for st in all_trains)
            n_frames = max_frame + 1
        else:
            n_frames = n_frames_inferred
        xlim_max = n_frames
        bins = np.arange(0, n_frames + bin_width, bin_width, dtype=int)
        bin_centers = bins[:-1] + bin_width / 2.0
        bin_width_s = (bin_width * dt) / 1000.0

    # ---- Figure ----
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.05)

    # Raster
    ax_r = fig.add_subplot(gs[0])
    for idx, train in enumerate(all_trains):
        if axis_units == 'ms':
            ax_r.scatter(train, [idx] * len(train), c='black', marker='|', s=50, linewidths=0.5)
        else:
            ax_r.scatter(train + 0.5, [idx] * len(train), c='black', marker='|', s=50, linewidths=0.5)

    title_id = f"cell {cell_id_for_title}" if cell_id_for_title is not None else f"index {cell_index}"
    ax_r.set_title(f'{rgc_type} Raster — {title_id}', fontsize=22, fontweight='bold')
    ax_r.set_ylabel('Trial', fontsize=24, fontweight='bold')
    ax_r.set_xlim(0, xlim_max)
    ax_r.set_ylim(-0.5, len(all_trains) - 0.5)
    ax_r.tick_params(axis='both', labelsize=24)
    ax_r.spines['top'].set_visible(False); ax_r.spines['right'].set_visible(False)
    ax_r.tick_params(labelbottom=False)

    # PSTH
    ax_p = fig.add_subplot(gs[1])
    histograms = []
    for train in all_trains:
        counts, _ = np.histogram(train, bins=bins)
        histograms.append(counts / bin_width_s)  # spikes/s
    histograms = np.asarray(histograms, dtype=float)
    mean_fr = np.mean(histograms, axis=0)

    ax_p.plot(bin_centers, mean_fr, 'k-', linewidth=2)
    ax_p.set_xlabel(f"Time ({'ms' if axis_units=='ms' else 'frames'})", fontsize=22, fontweight='bold')
    ax_p.set_ylabel('Rate (sp/s)', fontsize=24, fontweight='bold')
    ax_p.set_xlim(0, xlim_max); ax_p.set_ylim(0, None)
    ax_p.tick_params(axis='both', labelsize=24)
    ax_p.spines['top'].set_visible(False); ax_p.spines['right'].set_visible(False)

    fig.tight_layout()
    return fig



#%% 18. Compute_Linear : Linear Model Foward pass 


def compute_linear(movie, spatial_filter, temporal_filter, already_windowed=True):
    """
    Linear model forward pass (no nonlinearity, no spikes).
    
    Returns
    -------
    generator_signal : (N,)  — linear filter output
    """
    # Build spatiotemporal filter and flatten
    w = create_spatiotemporal_filter(spatial_filter, temporal_filter).astype(np.float32)
    w_flat = w.ravel()
    L = len(temporal_filter)
    
    # Prepare windowed stimulus
    if already_windowed:
        stim_windowed = movie
        N, L_win, H, W = stim_windowed.shape
        assert L_win == L, f"Window length {L_win} != temporal_filter length {L}"
    else:
        stim_windowed = compute_stim_windowed(movie, temporal_filter)
        N, L_win, H, W = stim_windowed.shape
        assert L_win == L, f"Window length {L_win} != temporal_filter length {L}"
    
    # Linear stage only: r = w · s
    S = stim_windowed.reshape(N, -1).astype(np.float32)
    generator_signal = S @ w_flat
    
    return generator_signal


# -----------------------------------------------------------------------------
# 2. compute_linear_for_mosaic - Compute linear responses for all cells
# -----------------------------------------------------------------------------
def compute_linear_for_mosaic(stim_windowed, overlaid_mosaic):
    """
    Compute LINEAR responses for all cells in the mosaic.
    No nonlinearity, no spikes - just generator signal.
    
    Returns
    -------
    responses : dict
        responses[rgc_type] = list of dicts with 'generator_signal', etc.
    """
    responses = {}
    
    # Create positioned spatial filters
    combined_filter, individual_filters = create_overlaid_spatial_filters_simple(
        overlaid_mosaic, stim_windowed.shape[2:]
    )
    
    total_cells = sum(len(filters) for filters in individual_filters.values())
    
    with tqdm(total=total_cells, desc="Computing linear responses") as pbar:
        for rgc_type, cell_filters in individual_filters.items():
            if len(cell_filters) == 0:
                responses[rgc_type] = []
                continue
                
            responses[rgc_type] = []
            params = RGC_PARAMS[rgc_type]
            
            # Create temporal filter
            temporal_filter = create_temporal_filter(
                peak1_ms=params['peak1_ms'], 
                peak2_ms=params['peak2_ms'],
                width1_ms=params['width1_ms'], 
                width2_ms=params['width2_ms'],
                amp1=params['amp1'],
                amp2=params['amp2'],
                smooth_onset=True, 
                cell_type=rgc_type
            )
                                        
            for cell_data in cell_filters:
                spatial_filter = cell_data['filter']
                cell_center = cell_data['center']
                cell_id = cell_data['cell_id']
                
                # LINEAR model - just the generator signal
                generator_signal = compute_linear(
                    stim_windowed, spatial_filter, temporal_filter
                )
                
                response = {
                    'generator_signal': generator_signal,
                    'firing_rate': generator_signal.copy(),  # For plotting compatibility, use generator signal
                    'spikes': None,
                    'spike_times': None,
                    'spatial_filter': spatial_filter,
                    'cell_center': cell_center,
                    'rgc_type': rgc_type,
                    'cell_id': cell_id
                }
                
                responses[rgc_type].append(response)
                pbar.update(1)
    
    return responses


#%% 19  =========================================================================
#   LN-LN CASCADE (SUBUNIT MODEL) FUNCTIONS
#   Subunit nonlinearity: softplus applied per pixel (params: alpha_sub, beta_sub, gamma_sub)
#   Output nonlinearity: same as standard LNP (params: alpha, beta, gamma)
# ===========================================================================



def compute_lnln(stim_windowed, spatial_filter, temporal_filter,
                 alpha_sub, beta_sub, gamma_sub,
                 alpha, beta, gamma, dt=8.3,
                 nonlinearity_type='soft_rectifier',
                 subunit_size=1, pixel_size_um=32):
    """
    LN-LN cascade with local spatial pooling before subunit nonlinearity.
    
    subunit_size : int
        Pooling window size in pixels (default=2, i.e. 2x2 = 64 µm at 32 µm/pixel).
        Set to 1 for original per-pixel behavior.
    pixel_size_um : float
        Size of each pixel in micrometers (for documentation).
    """
    N, L, H, W = stim_windowed.shape

    # Step 1: Per-pixel temporal filtering
    u = np.tensordot(stim_windowed, temporal_filter, axes=([1], [0]))  # (N, H, W)

    # Step 2a: Local spatial pooling (~64 µm subunit at 32 µm/pixel with size=2)
    if subunit_size > 1:
        u_pooled = uniform_filter(u, size=(1, subunit_size, subunit_size), mode='constant')
        u_pooled *= subunit_size**2  # mean -> sum
    else:
        u_pooled = u

    # Step 2b: Subunit nonlinearity on pooled signal
    h = apply_nonlinearity(u_pooled, alpha_sub, beta_sub, gamma_sub, nonlinearity_type)

    # Step 3: Spatial summation
    generator_signal = np.sum(h * spatial_filter[np.newaxis, :, :], axis=(1, 2))

    # Step 4: Output nonlinearity
    max_firing_rate = 700
    firing_rate = apply_nonlinearity(generator_signal, alpha, beta, gamma,
                                     nonlinearity_type)
    firing_rate = np.minimum(firing_rate, max_firing_rate)

    # Step 5: Poisson spikes
    spikes = generate_poisson_spikes(firing_rate, dt)
    spike_times = poisson_times_from_rate(firing_rate, dt)

    return generator_signal, firing_rate, spikes, spike_times, u, h



def compute_lnln_for_mosaic(stim_windowed, overlaid_mosaic, dt=8.3,
                            nonlinearity_type='soft_rectifier'):
    """
    Compute LN-LN cascade responses for all cells in the mosaic.

    Returns
    -------
    responses : dict
        responses[rgc_type] = list of dicts, each with keys:
        'generator_signal', 'firing_rate', 'spikes', 'spike_times',
        'u', 'h', 'spatial_filter', 'cell_center', 'rgc_type', 'cell_id'
    """
    responses = {}

    combined_filter, individual_filters = create_overlaid_spatial_filters_simple(
        overlaid_mosaic, stim_windowed.shape[2:]
    )

    total_cells = sum(len(filters) for filters in individual_filters.values())

    with tqdm(total=total_cells, desc="Computing LN-LN responses") as pbar:
        for rgc_type, cell_filters in individual_filters.items():
            if len(cell_filters) == 0:
                responses[rgc_type] = []
                continue

            responses[rgc_type] = []
            params = RGC_PARAMS[rgc_type]

            temporal_filter = create_temporal_filter(
                peak1_ms=params['peak1_ms'],
                peak2_ms=params['peak2_ms'],
                width1_ms=params['width1_ms'],
                width2_ms=params['width2_ms'],
                amp1=params['amp1'],
                amp2=params['amp2'],
                smooth_onset=True,
                cell_type=rgc_type
            )

            for cell_data in cell_filters:
                spatial_filter = cell_data['filter']
                cell_center = cell_data['center']
                cell_id = cell_data['cell_id']

                gamma_out = params.get('gamma_lnln', params['gamma'])
                generator_signal, firing_rate, spikes, spike_times, u, h = compute_lnln(
                    stim_windowed, spatial_filter, temporal_filter,
                    params['alpha_sub'], params['beta_sub'], params['gamma_sub'],
                    params['alpha'], params['beta'], gamma_out,
                    dt, nonlinearity_type
                )

                response = {
                    'generator_signal': generator_signal,
                    'firing_rate': firing_rate,
                    'spikes': spikes,
                    'spike_times': spike_times,
                    'u': u,
                    'h': h,
                    'spatial_filter': spatial_filter,
                    'cell_center': cell_center,
                    'rgc_type': rgc_type,
                    'cell_id': cell_id
                }

                responses[rgc_type].append(response)
                pbar.update(1)

    return responses