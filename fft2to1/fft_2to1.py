#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

import cv2

def load_image(image_path):
    # load an image by the scaling of RGB[A] to Gray:Y←0.299⋅R+0.587⋅G+0.114⋅B
    # Ref: https://docs.opencv.org/4.2.0/de/d25/imgproc_color_conversions.html#color_convert_rgb_gray
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image

def calc_radial_profile(image):
    # Perform 2D FFT
    fft_image = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft_image)
    magnitude_spectrum = np.abs(fft_shifted)

    # Get image dimensions
    rows, cols = image.shape
    mid_row, mid_col = rows // 2, cols // 2

    # Compute max radius and frequency, and initialize radial profile and count
    max_r = int(np.ceil(np.sqrt(mid_row**2 + mid_col**2)))
    max_x = int(np.ceil(mid_row))
    max_y = int(np.ceil(mid_col))
    freq_r = np.arange(max_r)
    freq_x = np.arange(max_x)
    freq_y = np.arange(max_y)
    rprof = np.zeros(max_r)
    xprof = np.zeros(max_x)
    yprof = np.zeros(max_y)
    countr = np.zeros(max_r)
    countx = np.zeros(max_x)
    county = np.zeros(max_y)

    ### Compute the radial profile ###
    # Create coordinate grid
    y, x = np.indices((rows, cols))
    r = np.round(np.sqrt((y - mid_row) ** 2 + (x - mid_col) ** 2)).astype(int)
    r = np.minimum(r, max_r - 1)  # Ensure r does not exceed max_radius - 1

    # Bin the values using numpy's efficient indexing
    np.add.at(rprof, r, magnitude_spectrum)
    np.add.at(countr, r, 1)

    # Avoid division by zero and compute the average
    countr[countr == 0] = 1
    rprof /= countr

    ### Compute kx and ky profiles ###
    kx = np.round(np.sqrt((x - mid_col) ** 2)).astype(int)
    ky = np.round(np.sqrt((y - mid_row) ** 2)).astype(int)
    kx = np.minimum(kx, max_x - 1)  # Ensure kx does not exceed max_x - 1
    ky = np.minimum(ky, max_y - 1)  # Ensure ky does not exceed max_y - 1
    np.add.at(xprof, kx, magnitude_spectrum)
    np.add.at(countx, kx, 1)
    np.add.at(yprof, ky, magnitude_spectrum)
    np.add.at(county, ky, 1)
    countx[countx == 0] = 1
    county[county == 0] = 1
    xprof /= countx
    yprof /= county

    return (freq_r, rprof), (freq_x, xprof), (freq_y, yprof)


def binning_data(x, y, bin_size=5, is_log=True):
    if is_log:
        x = np.log10(x)
        y = np.log10(y)
    binned_x = []
    binned_y = []
    binned_err = []
    for i in range(0, len(x), bin_size):
        binned_x.append(np.mean(x[i:i+bin_size]))
        binned_y.append(np.mean(y[i:i+bin_size]))
        binned_err.append(np.std(y[i:i+bin_size]) / np.sqrt(bin_size))
    return np.array(binned_x), np.array(binned_y), np.array(binned_err)

def power_law(params, x):
    a, b = params
    return a * x + b

def residuals(params, x, y, yerr):
    return (y - power_law(params, x)) / yerr

def fit_power_law(x, y, yerr, xmin=1, xmax=-400):
    """
    Fit a power law to the binned data using least squares optimization.
    """
    # Initial guess for the parameters and bounds
    initial_params = [1.0, 1.0]
    bounds = ([-np.inf, -np.inf], [np.inf, np.inf])

    # Perform least squares fitting
    result = least_squares(
        residuals, initial_params,
        args=(x[xmin:xmax], y[xmin:xmax], yerr[xmin:xmax]),
        bounds=bounds
    )
    params = result.x
    J = result.jac
    cov = np.linalg.inv(J.T @ J)
    errors = np.sqrt(np.diag(cov))
    chi2 = np.sum(result.fun ** 2)
    dof = len(x) - len(initial_params)

    return params, errors, chi2, dof, (xmin, xmax)

def plot_pix_scale(x, y, yerr, params, errors, chi2, dof, fmin, fmax, ax=None):
    """
    Plot the pixel scale on the given axis.
    """
    if ax is None:
        ax = plt.gca()
    ax.errorbar(10**x, 10**y, yerr=10**yerr, fmt='o', markersize=3, color='black', label='data')
    ax.axvspan(10**x[fmin], 10**x[fmax], color='gray', alpha=0.03)
    ax.plot(10**x, 10**power_law(params, x), 'r-', lw=2, label=f"fit result:\n10^"+ \
        fr"[{params[0]:.2f} ($\pm${errors[0]:.2f}) f + {params[1]:.2f} ($\pm${errors[1]:.2f})]"+'\n'+fr"$\chi^2$ / DOF = {chi2:.2f} / {dof:.0f}")
    ax.grid(lw=0.2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Frequency[/pix]')
    ax.set_ylabel('Magnitude')
    ax.legend()
    return ax

def main():
    # get parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'filepath',
        help='Path to the image file',
        type=str,
    )
    args = parser.parse_args()

    # load image
    image = load_image(args.filepath)

    # compute radial profile
    frequency, radial_profile = computer_radial_profile(image)

    # binning data
    fbin, rbin, err = binning_data(frequency, radial_profile, bin_size=5)

    # fit data by power law
    params, errors, chi2, dof, (fmin, fmax) = fit_power_law(fbin, rbin, err)

    # plot data
    fig, ax = plt.subplots(figsize=(8, 6))
    _=plot_pix_scale(fbin, rbin, params, errors, chi2, dof, fmin, fmax, ax=ax)
    plt.show()

if __name__ == '__main__':
    main()
# This script computes the radial profile of the FFT magnitude spectrum of an image.
# It uses numpy for efficient computation and matplotlib for visualization.
# The script takes an image file path as input and outputs the radial profile plot.
