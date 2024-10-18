import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import convolve

# Download and load the record and annotation from PhysioNet's MIT-BIH Arrhythmia Database
record_name = '100'  # Example record from MIT-BIH dataset
record = wfdb.rdrecord(record_name, sampfrom=0, sampto=1000, pn_dir='mitdb')  # Adjust sampto for more data

# Extract the EKG signal (channel 0 contains the EKG signal)
ekg_signal = record.p_signal[:, 0]
fs = record.fs  # Sampling frequency
time = np.arange(len(ekg_signal)) / fs  # Time array using the sampling frequency

# Apply Gaussian filter for noise smoothing
sigma = 2  # Standard deviation for Gaussian filter
smoothed_signal = gaussian_filter1d(ekg_signal, sigma=sigma)

# Calculate the derivative for edge detection (helps to identify sharp changes like QRS complex)
edge_signal = np.gradient(smoothed_signal)

# Calculate noise reduction by finding the difference between original and smoothed signals
noise_reduction = ekg_signal - smoothed_signal

# Create subplots for better visualization of each step
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Original EKG signal
axs[0, 0].plot(time, ekg_signal, color='blue', label='Original EKG Signal')
axs[0, 0].set_title('Original EKG Signal (Raw)')
axs[0, 0].set_xlabel('Time (seconds)')
axs[0, 0].set_ylabel('Amplitude (mV)')
axs[0, 0].grid(True)
axs[0, 0].legend()

# Gaussian-filtered EKG signal (noise reduction)
axs[0, 1].plot(time, smoothed_signal, color='orange', label='Smoothed EKG (Noise Reduced)', linewidth=2)
axs[0, 1].set_title('Smoothed EKG Signal (Gaussian Filter)')
axs[0, 1].set_xlabel('Time (seconds)')
axs[0, 1].set_ylabel('Amplitude (mV)')
axs[0, 1].grid(True)
axs[0, 1].legend()

# Edge detection signal (highlights sharp transitions like heartbeats)
axs[1, 0].plot(time, edge_signal, color='green', label='Edge Detection (Gradient of Smoothed Signal)', linewidth=2)
axs[1, 0].set_title('Edge Detection (Gradient Method)')
axs[1, 0].set_xlabel('Time (seconds)')
axs[1, 0].set_ylabel('Amplitude Change Rate')
axs[1, 0].grid(True)
axs[1, 0].legend()

# Noise reduction signal (difference between original and smoothed)
axs[1, 1].plot(time, noise_reduction, color='purple', label='Noise (Original - Smoothed)', linewidth=2)
axs[1, 1].set_title('Noise Reduction (Difference Signal)')
axs[1, 1].set_xlabel('Time (seconds)')
axs[1, 1].set_ylabel('Amplitude Difference (mV)')
axs[1, 1].grid(True)
axs[1, 1].legend()

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()

# Define a Sobel kernel for edge detection (emphasizes sharp changes in signal)
sobel_kernel = np.array([-1, 0, 1])  # 1D Sobel kernel for edge detection

# Apply convolution with the Sobel kernel to detect sharp edges
edge_signal_sobel = convolve(ekg_signal, sobel_kernel, mode='reflect')

# Create new plots for the Sobel edge detection
plt.figure(figsize=(10, 5))

# Plot original EKG signal again for comparison
plt.subplot(1, 2, 1)
plt.plot(time, ekg_signal, color='blue', label='Original EKG Signal')
plt.title('Original EKG Signal (Raw)')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude (mV)')
plt.grid(True)
plt.legend()

# Plot the result of edge detection using Sobel kernel
plt.subplot(1, 2, 2)
plt.plot(time, edge_signal_sobel, color='red', label='Edge Detection (Sobel Kernel)', linewidth=2)
plt.title('Edge Detection using Sobel Kernel (Sharp Changes)')
plt.xlabel('Time (seconds)')
plt.ylabel('Edge Strength (Amplitude Change Rate)')
plt.grid(True)
plt.legend()

# Adjust layout and show the plots
plt.tight_layout()
plt.show()

