import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Import the functions from the light_traversal.py file
from light_traversal import get_rotated_linear_polariser_matrix, get_rotated_quarter_wave_plate, get_sample_matrix, get_matrix_from_psi_delta
from helpers import *

# Calculates the expected intensity of the light, for a range of compensator angles, using the Jones' matrix ray transfer method
def get_expected_intensities(compensator_angles, sample_angle_of_incidence, n_gold, k_gold, d, x_offset = 0, y_offset = 0):
    # Get the default parameters for this traversal
    N_air, N_glass = get_default_refractive_index_param()

    # Turn the n_gold and k_gold into a complex refractive index
    N_gold = get_complex_refractive_index(n_gold, k_gold)

    original_field_strength = np.array([1, 0]) # Parallel Linearly-Polarised Light

    sample_mat = get_sample_matrix(sample_angle_of_incidence, N_air, N_gold, N_glass, d, 632.8e-9)

    polarisation_mat = get_rotated_linear_polariser_matrix(-np.pi/4)
    analyser_mat = get_rotated_linear_polariser_matrix(np.pi / 4)

    # Multiply the Jones' matrices in reverse order to represent the light-ray traversal, then multiply by the field strength vector to apply this combined matrix to it
    final_field_strength = np.array([analyser_mat @ get_rotated_quarter_wave_plate(compensator_angle - x_offset) @ sample_mat @ polarisation_mat @ original_field_strength for compensator_angle in compensator_angles])# Use @ instead of * to allow for different sized matrices to be dot-producted together

    # Find the effective reflection and take the absolute value so it's real
    # R_effective = (R_paralell + R_perpendicular) / 2
    intensities = np.abs(np.sum(final_field_strength, axis=2).flatten() / 2) + y_offset

    # Normalise the data
    intensities /= max(intensities)

    return intensities

# Define the initial guesses and bounds for the fitting
def get_guesses_and_bounds():
    # |------------------------ Initial guesses and bounds for parameters ------------------------|
    n_angle_guess,      n_angle_bounds      = np.radians(70),                [np.radians(60), np.radians(80)]
    # n_angle_guess,      n_angle_bounds      = np.radians(50),                [np.radians(40), np.radians(60)]

    # --------------------------- Parameters for Gold and Glass ----------------------------
    n_gold_guess,       n_gold_bounds       = 0.18377,                       [0.18360, 0.18380]
    k_gold_guess,       k_gold_bounds       = 3.4313,                        [3.4305, 3.4320]
    # --------------------------------------------------------------------------------------

    # # ---------------------------- Parameters for Silicon Wafer ----------------------------
    # n_gold_guess,       n_gold_bounds       = 1.455,                         [1.2, 1.6]
    # k_gold_guess,       k_gold_bounds       = 0,                             [-0.05, 0.05]
    # # --------------------------------------------------------------------------------------

    d_guess,            d_bounds            = 40e-9,                         [10e-9, 150e-9]
    x_offset_guess,     x_offset_bounds     = 0,                             [-np.pi/8, np.pi/8]
    y_offset_guess,     y_offset_bounds     = 0,                             [-1, 1]

    # Combine initial guesses into single array
    initial_guesses = [n_angle_guess, n_gold_guess, k_gold_guess, d_guess, x_offset_guess, y_offset_guess]

    # Combine bounds into single array
    bounds = np.array([n_angle_bounds, n_gold_bounds, k_gold_bounds, d_bounds, x_offset_bounds, y_offset_bounds])

    return initial_guesses, bounds

# Fits the provided data to the expected intensities, returning the optimal parameters which were found, along with their errors
def fit_data_to_expected(compensator_angles, measured_intensities, intensity_uncertainties, output_in = None):
    # Get the initial guesses and bounds for each parameter
    initial_guesses, bounds = get_guesses_and_bounds()

    function_to_fit = get_expected_intensities

    # Fit the data using these initial parameters
    optimal_param, param_convolution = curve_fit(function_to_fit , compensator_angles, measured_intensities, p0=initial_guesses, bounds=(bounds[::, 0], bounds[::, 1]))

    # Calculate errors from convolution matrix
    param_err = np.sqrt(np.diag(param_convolution))

    # Print out the values, along with their errors
    output = print_parameters_nicely(optimal_param, param_err, names=["Angle", "N_gold", "K_gold", "d", "X-Offset", "Y-Offset"], units=["Degrees", "", "", "Metres", "Degrees", ""], conversions = [180/np.pi, 1, 1, 1, 180/np.pi, 1], display_filter = [False, False, False, True, False, False])

    # Add parameter output to output_in array
    if not (output_in is None):
        output_in.extend(output)

    return optimal_param, param_err

# Read the data from a file, then plot this data, alongside the expected intensities from the optimal fitting of the data
def fit_from_data(filenames, x_bounds = None):
    if len(filenames) > 1:
        combined_output = []
    
    for i, filename in enumerate(filenames):
        print("Fitting {}:".format(filename))
    
        # Read the data and seperate it into x and y values
        data = read_file_to_data(filename)
        data_x, data_y = data[0], data[1]

        # Smooth the data
        data_x, data_y = smooth_data(data_x, data_y, smoothing_width = 5)

        # Normalise the data
        data_y = normalise_data(data_y)

        # Format the figure and plot
        format_plot(max(data_y))

        # A modifier to add the filenamae if there are multiple files
        label_modifier = " {}".format(filename) if len(filenames) > 1 else ""

        # Adjust the bounds of the plot
        if not (x_bounds is None):
            new_data_indices = np.argwhere((data_x >= np.radians(x_bounds[0])) & (data_x <= np.radians(x_bounds[1]))).flatten()
            data_x = data_x[new_data_indices]
            data_y = data_y[new_data_indices]

        # Uncertainty in the y-data
        sigma_percent = 0.02 
        sigma = data_y * sigma_percent

        # Get the output for the data which was fitted
        output = []

        # Fit the data to the function
        optimal_param, param_err = fit_data_to_expected(data_x, data_y, sigma, output)

        if len(filenames) > 1:
            combined_output.append(output)

        intensity_function = get_expected_intensities

        # Plot the calculated result using the fitting parameters
        plt.plot(np.degrees(data_x), intensity_function(data_x, *optimal_param), c='k', ls="-", label="Calculated Result{}".format(label_modifier)) # Plot the light intensity expected for the fitted parameters

        expected_y = intensity_function(data_x, *optimal_param)
        chi_sqr = np.sum((data_y - expected_y) ** 2 / expected_y)
        print("\tGoodness of fit: ", chi_sqr)

        # Plot the measured data to the same figure
        # plt.errorbar(np.degrees(data_x), data_y, c='r', alpha=0.2, yerr=sigma, fmt='o', label="Intensity Data")
        plt.scatter(np.degrees(data_x), data_y, c='r', label="Intensity Data{}".format(label_modifier), lw=4)
        plt.fill_between(np.degrees(data_x), data_y - sigma, data_y + sigma, color="r", alpha=0.2, label="Errors on Intensity Data")

        # Add spacing between outputs for different files
        if i < len(filenames) - 1:
            print()

    # Print the output parameters in a list
    if len(filenames) > 1:
        import sys
        np.savetxt(sys.stdout.buffer, np.array(combined_output), fmt='%.4E', delimiter=',')

    # plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    fit_from_data(["data/Gold_C_5"])
