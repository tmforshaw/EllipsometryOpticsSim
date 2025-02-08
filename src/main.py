import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Import the functions from the light_traversal.py file
from light_traversal import get_rotated_linear_polariser_matrix, get_rotated_quarter_wave_plate, get_sample_matrix
from helpers import *

# Select the matrix to use for the sample
SAMPLE_MATRIX_FUNCTION = 1 

# Calculates the expected intensity of the light, for a range of compensator angles, using the Jones' matrix ray transfer method
def get_expected_intensities(compensator_angles, sample_angle_of_incidence, n_gold, k_gold, d, x_offset = 0, y_offset = 0):
    # Get the default parameters for this traversal
    N_air, N_glass = get_default_refractive_index_param()

    # Turn the n_gold and k_gold into a complex refractive index
    N_gold = get_complex_refractive_index(n_gold, k_gold)

    original_field_strength = np.array([1, 0]) # Parallel Linearly-Polarised Light

    sample_mat = get_sample_matrix(sample_angle_of_incidence, N_air, N_gold, N_glass, d, 632.8e-9, SAMPLE_MATRIX_FUNCTION)

    # polarisation_mat = get_rotated_linear_polariser_matrix(0)
    # analyser_mat = get_rotated_linear_polariser_matrix(np.pi / 2)

    polarisation_mat = get_rotated_linear_polariser_matrix(-np.pi/4)
    analyser_mat = get_rotated_linear_polariser_matrix(np.pi / 4)

    # Multiply the Jones' matrices in reverse order to represent the light-ray traversal, then multiply by the field strength vector to apply this combined matrix to it
    final_field_strength = np.array([analyser_mat @ get_rotated_quarter_wave_plate(compensator_angle - x_offset) @ sample_mat @ polarisation_mat @ original_field_strength for compensator_angle in compensator_angles])# Use @ instead of * to allow for different sized matrices to be dot-producted together

    # Find the effective reflection and take the absolute value so it's real
    # R_effective = (R_paralell + R_perpendicular) / 2
    # intensities = np.abs(np.sum(final_field_strength, axis=2).reshape(len(final_field_strength)) / 2) * amplitude + offset 

    intensities = np.sum(np.abs(final_field_strength) ** 2, axis=2).reshape(len(final_field_strength)) + y_offset

    intensities /= max(intensities)

    return intensities

def get_guesses_and_bounds():
    # Initial guesses and bounds for parameters
    # amplitude_guess,  amplitude_bounds  = 1,                             [0.001, 1]
    n_angle_guess,    n_angle_bounds    = get_default_brewsters_angle(), [50 * np.pi/180, 60 * np.pi / 180]
    n_gold_guess,     n_gold_bounds     = 0.19404,                       [0, 5]
    k_gold_guess,     k_gold_bounds     = 3.5934,                        [-1, 5]
    d_guess,          d_bounds          = 40e-9,                         [10e-9, 100e-9]
    # wavelength_guess, wavelength_bounds = 632.8e-9,                      [600e-9, 700e-9]
    x_offset_guess,     x_offset_bounds     = 0,                             [-np.pi/8, np.pi/8]
    y_offset_guess,     y_offset_bounds     = 0,                             [0, 1]

    # Combine initial guesses into single array
    initial_guesses = [n_angle_guess, n_gold_guess, k_gold_guess, d_guess, x_offset_guess, y_offset_guess]

    # Combine bounds into single array
    bounds = np.array([n_angle_bounds, n_gold_bounds, k_gold_bounds, d_bounds, x_offset_bounds, y_offset_bounds])

    return initial_guesses, bounds

# Fits the provided data to the expected intensities, returning the optimal parameters which were found, along with their errors
def fit_data_to_expected(compensator_angles, measured_intensities, intensity_uncertainties):
    # Get the initial guesses and bounds for each parameter
    initial_guesses, bounds = get_guesses_and_bounds()

    # Fit the data using these initial parameters
    optimal_param, param_convolution = curve_fit(get_expected_intensities, compensator_angles, measured_intensities, p0=initial_guesses, bounds=(bounds[::, 0], bounds[::, 1]))

    # Calculate errors from convolution matrix
    param_err = np.sqrt(np.diag(param_convolution))

    # Print out the values, along with their errors
    print_parameters_nicely(optimal_param, param_err, names=["Angle", "N_gold", "K_gold", "d", "X-Offset", "Y-Offset"], units=["Degrees", "", "", "Metres", "Degrees", ""])

    return optimal_param, param_err

# Read the data from a file, then plot this data, alongside the expected intensities from the optimal fitting of the data
def fit_from_data(filenames):
    for filename in filenames:
        print("Data for {}:\n".format(filename))
    
        # Read the data and seperate it into x and y values
        data = read_file_to_data(filename)
        data_x, data_y = data[0], data[1]

        # TODO Normalise the data
        data_y = normalise_data(data_y)

        # Format the figure and plot
        format_plot(max(data_y))

        # A modifier to add the filenamae if there are multiple files
        label_modifier = " {}".format(filename) if len(filenames) > 1 else ""

        # Uncertainty in the y-data
        sigma_percent = 0.02 
        sigma = data_y * sigma_percent

        # Fit the data to the function
        optimal_param, param_err = fit_data_to_expected(data_x, data_y, sigma)
        plt.plot(data_x * 180 / np.pi, get_expected_intensities(data_x, *optimal_param), c='k', ls="-", label="Calculated Result{}".format(label_modifier)) # Plot the light intensity expected for the fitted parameters

        # Plot the measured data to the same figure
        # plt.errorbar(data_x * 180 / np.pi, data_y, c='r', alpha=0.2, yerr=sigma, fmt='o', label="Intensity Data")
        plt.scatter(data_x * 180 / np.pi, data_y, c='r', label="Intensity Data{}".format(label_modifier), lw=4)
        plt.fill_between(data_x * 180 / np.pi, data_y - sigma, data_y + sigma, color="r", alpha=0.2, label="Errors on Intensity Data")

    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_from_data(filenames):
    for filename in filenames:
        print("Data for {}:\n".format(filename))
    
        # Read the data and seperate it into x and y values
        data = read_file_to_data(filename)
        data_x, data_y = data[0], data[1]

        # Format the figure and plot
        format_plot(max(data_y))

        # A modifier to add the filenamae if there are multiple files
        label_modifier = " {}".format(filename) if len(filenames) > 1 else ""

        # Plot the measured data to the same figure
        plt.scatter(data_x * 180 / np.pi, data_y, c='r', label="Intensity Data{}".format(label_modifier))

    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    print_sample_matrix_type(SAMPLE_MATRIX_FUNCTION)

    # plot_default()

    # fit_from_data(["data/Gold_A_1"])
    # fit_from_data(["data/Gold_B_4_NDF"])

    # fit_from_data(["data/Gold_B_5_NDF"])
    # fit_from_data(["data/Gold_B_5_NDF", "data/Gold_B_6_NDF_Amp"])
    # fit_from_data(["data/Gold_B_6_NDF_Amp"])

    # fit_from_data(["data/Gold_C_1"])

    fit_from_data(["data/Gold_C_45_45"])

    # plot_range_of_wavelengths()
    # plot_range_of_depths()
    # plot_range_of_brewsters()

main()
