# -*- coding: utf-8 -*-

# This script performs seismic data processing and beamforming for analyzing 
# ambient noise cross-correlations (ANCs) in dense array seismic environments.
# It is designed for rapid phase velocity extraction and surface wave mode analysis 
# at a specified period beneath the array.
#
# The script applies narrow bandpass filtering, selects relevant data, 
# performs phase shifts in the frequency domain, stacks and computes envelope maxima 
# for beamforming, and visualizes the results with a heatmap of the slowness grid.

# Author: Yuan Yusong, China University of Geosciences, Wuhan
# Date: 2025-11-15

import numpy as np
from obspy import Stream
import matplotlib.pyplot as plt
import scipy.signal as ss
from scipy.fftpack import ifft
import scipy.ndimage

# Global Plotting Settings
plt.rcParams.update({
    'font.size': 26,          # Global font size for all text
    'axes.titlesize': 26,     # Font size for plot titles
    'axes.labelsize': 26,     # Font size for axis labels
    'xtick.labelsize': 22,    # Font size for X-axis tick labels
    'ytick.labelsize': 22,    # Font size for Y-axis tick labels
    'legend.fontsize': 20,    # Font size for legend
    'figure.dpi': 300,        # Output resolution for figures
    'savefig.bbox': 'tight'  # Automatically crop white borders when saving figures
})  

# Bandpass Filter Function
def bandpass(tr, per, alpha=10):
    """
    Applies narrow bandpass filtering using a Gaussian window for a single center period.

    Args:
    tr (obspy.Trace): Input seismic trace object.
    per (float): Target center period for the bandpass filter.
    alpha (float): Width of the bandpass filter (default is 10).

    Returns:
    np.ndarray: Filtered seismic waveform (real part of the inverse FFT).
    """
    # Perform Fourier Transform of the signal
    sf = np.fft.fft(tr.data)  
    delta = tr.stats.delta  # Sampling interval
    ns = len(sf)  # Length of the signal
    dom = 2 * np.pi / (ns * delta)  # Dominant frequency resolution
    om_k = 2 * np.pi / per  # Angular frequency corresponding to the center period

    # Create a Gaussian window in the frequency domain
    b = np.exp(-((dom * np.arange(ns) - om_k) / om_k) ** 2 * alpha)
    fils = b * sf  # Apply the Gaussian filter

    # Zero out high frequencies and correct the DC component
    for m in range(ns // 2 + 1, ns):
        fils[m] = 0.0
    fils[0] /= 2.0  # Correct the DC component
    fils[ns // 2] = np.real(fils[ns // 2])  # Handle Nyquist frequency

    # Perform the inverse FFT to get the filtered signal
    tmp = ifft(fils)
    return np.real(tmp[0:ns])  # Return only the real part of the inverse FFT

# Data Selection Function
def select_data(src_cen, rece_cen, arr_half_length, per):
    """
    Selects relevant seismic data from a given file based on source and receiver positions 
    and applies filtering to the selected traces.

    Args:
    src_cen (int): Central source position.
    rece_cen (int): Central receiver position.
    arr_half_length (int): Half-length of the source/receiver array.
    per (float): Target center period for bandpass filtering.

    Returns:
    Stream: Stream with selected and processed seismic data.
    obspy.Trace: The central trace for the given source and receiver.
    """
    # Create arrays for source and receiver positions
    src_arr = np.linspace(src_cen - arr_half_length, src_cen + arr_half_length, num=2*arr_half_length + 1)
    rece_arr = np.linspace(rece_cen - arr_half_length, rece_cen + arr_half_length, num=2*arr_half_length + 1)

    # Initialize the output Stream object to store selected traces
    st_out = Stream()

    # Load the seismic data from a pickle file
    fnm = f'Test_ANCs.pickle'
    st = np.load(fnm, allow_pickle=True)

    # Define the time mask to crop data between -5s and 15s
    tr = st[0]  # Use the first trace to determine time parameters
    npts = tr.stats.npts  # Number of data points in the trace
    t0 = tr.stats.b  # Start time of the trace
    delta = tr.stats.delta  # Sampling interval
    taxis = np.linspace(t0, t0 + npts * delta - delta, npts)  # Time axis for the trace
    mask = (taxis >= -5) & (taxis <= 15)  # Time mask between -5s and 15s

    # Loop through the traces and select relevant ones based on source and receiver positions
    for tr in st:
        src = int(tr.stats.kevnm)  # Source position from trace metadata
        rec = int(tr.stats.kstnm)  # Receiver position from trace metadata
        if src in src_arr and rec in rece_arr:
            tr_cp = tr.copy()  # Copy the trace
            tr.data += tr.data[::-1]  # Reverse the data and add it (this might be a form of data augmentation)
            tr_cp.data = tr.data[mask]  # Apply the time mask
            tr_cp.stats.b = -5.0  # Set the start time for the trace
            tr_cp.taper(max_percentage=0.05)  # Apply tapering to reduce edge effects
            tr_cp.data = np.squeeze(bandpass(tr_cp, per))  # Apply the bandpass filter
            st_out.append(tr_cp)  # Append the processed trace to the output Stream

            # Store the central trace for later use
            if int(tr_cp.stats.kevnm) == src_cen and int(tr_cp.stats.kstnm) == rece_cen:
                tr_cen = tr_cp.copy()

    return st_out, tr_cen

# Beamforming Stack Function
def beamforming_stack(st, tr_cen, src_slown, rece_slown):
    """
    Applies beamforming to the selected seismic data and computes the envelope maximum.

    Args:
    st (Stream): Stream containing the seismic traces.
    tr_cen (obspy.Trace): Central trace for comparison.
    src_slown (float): Source slowness (s/km).
    rece_slown (float): Receiver slowness (s/km).

    Returns:
    float: Maximum value of the envelope (beamformed signal).
    """
    stack = np.zeros_like(tr_cen.data)  # Initialize the stack with zeros
    sampling_rate = 1. / tr_cen.stats.delta  # Sampling rate

    # Loop through each trace in the Stream
    for tr in st:
        data = tr.data  # Seismic data
        src_diff = int(tr.stats.kevnm) - int(tr_cen.stats.kevnm)  # Difference in source positions
        rece_diff = int(tr.stats.kstnm) - int(tr_cen.stats.kstnm)  # Difference in receiver positions
        time_shift = src_diff * 0.02 * src_slown - rece_diff * 0.02 * rece_slown  # Calculate time shift based on slowness
        shift_samples = time_shift * sampling_rate  # Convert time shift to samples

        if abs(shift_samples) < len(data):
            shifted_data = precise_shift(data, shift_samples)  # Shift the data with sub-sample precision
        else:
            shifted_data = np.zeros_like(data)  # If shift is too large, create a zero array

        stack += shifted_data  # Add the shifted data to the stack

    stack /= len(st)  # Normalize the stack by the number of traces
    analy_signal = ss.hilbert(stack)  # Compute the analytic signal
    env = np.abs(analy_signal)  # Compute the envelope of the signal
    env_max = np.max(env)  # Get the maximum envelope value

    return env_max

# Precise Shift Function
def precise_shift(data, shift_samples):
    """
    Shifts the data in the frequency domain with sub-sample precision.

    Args:
    data (np.ndarray): Input seismic data.
    shift_samples (float): Number of samples to shift the data.

    Returns:
    np.ndarray: Shifted seismic data.
    """
    n = len(data)  # Number of data points in the trace
    freq = np.fft.fftfreq(n)  # Frequency components of the FFT
    fft_data = np.fft.fft(data)  # Perform FFT on the data
    phase_shift = np.exp(-2j * np.pi * freq * shift_samples)  # Phase shift factor
    shifted_fft = fft_data * phase_shift  # Apply phase shift to the FFT data
    return np.real(np.fft.ifft(shifted_fft))  # Inverse FFT to obtain the shifted data

# Grid Search for Optimal Slowness
def grid_search(st, tr_cen):
    """
    Performs a grid search over a range of slowness values to find the optimal source and receiver slownesses.

    Args:
    st (Stream): Stream with seismic traces.
    tr_cen (obspy.Trace): Central trace for comparison.

    Returns:
    np.ndarray: Grid with computed envelope maxima for each slowness combination.
    np.ndarray: Array of slowness values used for the grid search.
    """
    slowns = np.arange(0., 6.1, 0.1)  # Define the slowness range
    res_grid = np.zeros((len(slowns), len(slowns)))  # Initialize the result grid

    # Perform the grid search
    for i, src_slown in enumerate(slowns):
        for j, rece_slown in enumerate(slowns):
            env_max = beamforming_stack(st, tr_cen, src_slown, rece_slown)  # Compute envelope maximum for the current slowness combination
            res_grid[i, j] = env_max  # Store the result in the grid

    res_grid /= np.max(res_grid)  # Normalize the grid by the maximum value
    return res_grid, slowns

# Plotting Function
def plot_slowness_grid(res_grid, slowns, src_cen, rece_cen):
    """
    Plots the slowness grid as a heatmap with overlayed contours and local maxima.

    Args:
    res_grid (np.ndarray): Grid with computed envelope maxima.
    slowns (np.ndarray): Array of slowness values.
    src_cen (int): Central source position.
    rece_cen (int): Central receiver position.
    """
    res_grid /= np.max(res_grid)  # Normalize the grid

    # Create the figure and plot the heatmap
    plt.figure(figsize=(10, 8))
    im = plt.imshow(res_grid.T, extent=[slowns[0], slowns[-1], slowns[0], slowns[-1]], 
                    origin='lower', aspect='equal', cmap='viridis', vmin=0, vmax=1.0)

    # Add contour lines at the 0.5 level
    X, Y = np.meshgrid(slowns, slowns)
    cs = plt.contour(X, Y, res_grid.T, levels=[0.5], colors='white', linewidths=3.)
    plt.clabel(cs, inline=True, fontsize=22, fmt='%.1f')

    # Find the global maximum and highlight it
    global_max_pos = np.unravel_index(np.argmax(res_grid), res_grid.shape)
    global_max_value = res_grid[global_max_pos]
    plt.scatter(slowns[global_max_pos[0]], slowns[global_max_pos[1]], c='red', marker='x', s=300, label='Global Maximum')

    # Identify and plot local maxima
    neighborhood_size = 3
    local_max = scipy.ndimage.maximum_filter(res_grid, size=neighborhood_size) == res_grid
    local_max_coords = np.argwhere(local_max)

    significant_local_max_coords = [
        coord for coord in local_max_coords if res_grid[tuple(coord)] > 0.5 * global_max_value and not np.array_equal(coord, global_max_pos)
    ]

    if significant_local_max_coords:
        largest_local_max_coord = max(significant_local_max_coords, key=lambda coord: res_grid[tuple(coord)])
        plt.scatter(slowns[largest_local_max_coord[0]], slowns[largest_local_max_coord[1]], c='lime', marker='s', s=200, edgecolors='black', label='Local Max')

    # Add colorbar and labels
    cbar = plt.colorbar(im)
    cbar.set_label('Envelope Maximum')

    plt.xticks(np.arange(slowns[0], slowns[-1], 1))
    plt.yticks(np.arange(slowns[0], slowns[-1], 1))

    plt.xlabel('Source Slowness (s/km)')
    plt.ylabel('Receiver Slowness (s/km)')
    plt.title(f'Source {src_cen} Receiver {rece_cen}')

    plt.legend(loc='upper left')
    plt.savefig(f"src{src_cen}-rece{rece_cen}.png")
    plt.close()
    
# Main Function to Run the Analysis
def main():
    # Constants for source, receiver, and period
    src_cen = 260
    rece_cen = 320
    per = 0.5  # Center period for bandpass filtering
    arr_half_length = 3

    # Select the relevant data and perform grid search
    st, tr_cen = select_data(src_cen, rece_cen, arr_half_length, per)
    res_grid, slowns = grid_search(st, tr_cen)

    # Plot the results
    plot_slowness_grid(res_grid, slowns, src_cen, rece_cen)

# Run the main function
main()
