import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Import the functions from the light_traversal.py file
from light_traversal import get_rotated_linear_polariser_matrix, get_rotated_quarter_wave_plate, get_sample_matrix, get_matrix_from_psi_delta
from helpers import *

# Select the matrix to use for the sample
SAMPLE_MATRIX_FUNCTION = 4 

# Whether to fit using only psi and delta, or to use parameters
FIT_TO_PSI_DELTA = False

# Calculates the expected intensity of the light, for a range of compensator angles, using the Jones' matrix ray transfer method
def get_expected_intensities(compensator_angles, sample_angle_of_incidence, n_gold, k_gold, d, x_offset = 0, y_offset = 0):
    # Get the default parameters for this traversal
    N_air, N_glass = get_default_refractive_index_param()

    # Turn the n_gold and k_gold into a complex refractive index
    N_gold = get_complex_refractive_index(n_gold, k_gold)

    original_field_strength = np.array([1, 0]) # Parallel Linearly-Polarised Light

    sample_mat = get_sample_matrix(sample_angle_of_incidence, N_air, N_gold, N_glass, d, 632.8e-9, SAMPLE_MATRIX_FUNCTION)

    polarisation_mat = get_rotated_linear_polariser_matrix(-np.pi/4)
    analyser_mat = get_rotated_linear_polariser_matrix(np.pi / 4)

    # Multiply the Jones' matrices in reverse order to represent the light-ray traversal, then multiply by the field strength vector to apply this combined matrix to it
    final_field_strength = np.array([analyser_mat @ get_rotated_quarter_wave_plate(compensator_angle - x_offset) @ sample_mat @ polarisation_mat @ original_field_strength for compensator_angle in compensator_angles])# Use @ instead of * to allow for different sized matrices to be dot-producted together

    # Find the effective reflection and take the absolute value so it's real
    # R_effective = (R_paralell + R_perpendicular) / 2
    intensities = np.abs(np.sum(final_field_strength, axis=2).reshape(len(final_field_strength)) / 2) + y_offset

    # Normalise the data
    intensities /= max(intensities)

    return intensities

# Calculates the expected intensity of the light, for a range of compensator angles, using the Jones' matrix ray transfer method
def get_expected_intensities_psi_delta(compensator_angles, psi, delta, x_offset = 0, y_offset = 0):
    original_field_strength = np.array([1, 0]) # Parallel Linearly-Polarised Light

    sample_mat = get_matrix_from_psi_delta(psi, delta)

    polarisation_mat = get_rotated_linear_polariser_matrix(-np.pi/4)
    analyser_mat = get_rotated_linear_polariser_matrix(np.pi / 4)

    # Multiply the Jones' matrices in reverse order to represent the light-ray traversal, then multiply by the field strength vector to apply this combined matrix to it
    final_field_strength = np.array([analyser_mat @ get_rotated_quarter_wave_plate(compensator_angle - x_offset) @ sample_mat @ polarisation_mat @ original_field_strength for compensator_angle in compensator_angles])# Use @ instead of * to allow for different sized matrices to be dot-producted together

    # Find the effective reflection and take the absolute value so it's real
    # R_effective = (R_paralell + R_perpendicular) / 2
    intensities = np.abs(np.sum(final_field_strength, axis=2).reshape(len(final_field_strength)) / 2) + y_offset 

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
def fit_data_to_expected(compensator_angles, measured_intensities, intensity_uncertainties):
    # Get the initial guesses and bounds for each parameter
    initial_guesses, bounds = get_guesses_and_bounds()

    function_to_fit = get_expected_intensities

    # Adjust initial guesses and bounds for fitting to psi and delta, along with the function to fit
    if FIT_TO_PSI_DELTA:
        initial_guesses = [np.radians(20), np.radians(45), initial_guesses[-2], initial_guesses[-1]]
        bounds = np.array([[0, np.pi], [0, np.pi], bounds[-2], bounds[-1]])
        function_to_fit = get_expected_intensities_psi_delta

    # Fit the data using these initial parameters
    optimal_param, param_convolution = curve_fit(function_to_fit , compensator_angles, measured_intensities, p0=initial_guesses, bounds=(bounds[::, 0], bounds[::, 1]))

    # Calculate errors from convolution matrix
    param_err = np.sqrt(np.diag(param_convolution))

    # Print out the values, along with their errors, depending on if fitting was done using psi and delta or not
    if FIT_TO_PSI_DELTA:
        print_parameters_nicely(optimal_param, param_err, names=["Psi", "Delta", "X-Offset", "Y-Offset"], units=["Degrees", "Degrees", "", ""], conversions=[180/np.pi, 180/np.pi, 1, 1])
    else:
        print_parameters_nicely(optimal_param, param_err, names=["Angle", "N_gold", "K_gold", "d", "X-Offset", "Y-Offset"], units=["Degrees", "", "", "Metres", "Degrees", ""], conversions = [180/np.pi, 1, 1, 1, 180/np.pi, 1], display_filter = [True, False, False, True, False, False])

    return optimal_param, param_err

# Read the data from a file, then plot this data, alongside the expected intensities from the optimal fitting of the data
def fit_from_data(filenames, x_bounds = None):
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

        # Fit the data to the function
        optimal_param, param_err = fit_data_to_expected(data_x, data_y, sigma)

        # Choose the function to fit depending on FIT_TO_PSI_DELTA variable
        if FIT_TO_PSI_DELTA:
            intensity_function = get_expected_intensities_psi_delta
        else:
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

    # plt.legend()
    plt.tight_layout()
    plt.show()

def plot_from_data(filenames, x_bounds = None):
    for filename in filenames:
        print("Plotting {}:".format(filename))
    
        # Read the data and seperate it into x and y values
        data = read_file_to_data(filename)
        data_x, data_y = data[0], data[1]

        # Smooth the data
        data_x, data_y = smooth_data(data_x, data_y, smoothing_width = 5)

        # Normalise the data
        data_y = normalise_data(data_y)

        # Format the figure and plot
        format_plot(max(data_y))

        # Adjust the bounds of the plot
        if not (x_bounds is None):
            new_data_indices = np.argwhere((data_x >= np.radians(x_bounds[0])) & (data_x <= np.radians(x_bounds[1]))).flatten()
            data_x = data_x[new_data_indices]
            data_y = data_y[new_data_indices]

        # Plot the measured data to the same figure, With a label modifier to add the filenamae if there are multiple files
        plt.plot(np.degrees(data_x), data_y, label="Intensity Data{}".format(" {}".format(filename) if len(filenames) > 1 else ""), lw=4)

    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    print_sample_matrix_type(SAMPLE_MATRIX_FUNCTION)

    # plot_from_data(["data/Gold_Phi_192", "data/Gold_Phi_8_(194)", "data/Gold_Phi_7_(196)", "data/Gold_Phi_6_(198)", "data/Gold_Phi_1", "data/Gold_Phi_2_(204)", "data/Gold_Phi_3_(206)", "data/Gold_Phi_4_(208)", "data/Gold_Phi_5_(210)", "data/Gold_Phi_212", "data/Gold_Phi_214", "data/Gold_Phi_216"], x_bounds=[0, 180])
    # fit_from_data(["data/Gold_Phi_1", "data/Gold_Phi_2_(204)", "data/Gold_Phi_3_(206)", "data/Gold_Phi_4_(208)", "data/Gold_Phi_5_(210)", "data/Gold_Phi_6_(198)", "data/Gold_Phi_7_(196)", "data/Gold_Phi_8_(194)", "data/Gold_Phi_192", "data/Gold_Phi_212", "data/Gold_Phi_214", "data/Gold_Phi_216"])

    # fit_from_data(["data/Gold_C_45_45_3", "data/Gold_I_2"])
    # plot_from_data(["data/Gold_I_1"])

    # fit_from_data(["data/Gold_Ficc_1", "data/Gold_Ficc_2", "data/Gold_Ficc_3"])

    # fit_from_data(["data/Gold_52nm_2"])
    # fit_from_data(["data/Gold_52nm_1", "data/Gold_52nm_2",  "data/Gold_52nm_3"])
    # plot_from_data(["data/Gold_52nm_1", "data/Gold_52nm_2",  "data/Gold_65nm_1"])

    # fit_from_data(["data/Gold_52nm_3", "data/Gold_65nm_3"])
    # fit_from_data(["data/Gold_Ficc_4"])
    # fit_from_data(["data/Gold_Ficc_4", "data/Gold_52nm_3", "data/Gold_65nm_3"])
    # fit_from_data(["data/Gold_Ficc_4", "data/Gold_52nm_3", "data/Gold_65nm_3", "data/Gold_C_4"])

    # fit_from_data(["data/Glass_4"])
    # fit_from_data(["data/Silicon_5"])
    # fit_from_data(["data/Gold_65nm_5"])
    # fit_from_data(["data/Gold_52nm_3"])

    # fit_from_data(["data/Gold_C_5"])

    # plot_from_data(["data/Gold_50s_1", "data/Gold_69s_1", "data/Gold_80s_1", "data/Gold_90s_1", "data/Gold_100s_1", "data/Gold_110s_1"])
    fit_from_data(["data/Gold_40s_1", "data/Gold_50s_1", "data/Gold_69s_1", "data/Gold_71s_1", "data/Gold_73s_1", "data/Gold_80s_1", "data/Gold_85s_1", "data/Gold_90s_1", "data/Gold_100s_1", "data/Gold_110s_1"])

    # fit_from_data(["data/Gold_40s_1"])

    # plot_default()

    # plot_range_of_depths()
    # plot_range_of_brewsters()

def run_multiple_matrix_functions(filenames, matrix_function_range = range(4)):
    for i, matrix_index in enumerate(matrix_function_range):
        # Update global variable to change which sample matrix function is used
        global SAMPLE_MATRIX_FUNCTION
        SAMPLE_MATRIX_FUNCTION = matrix_index

        print_sample_matrix_type(matrix_index)
        fit_from_data(filenames)

        if i < len(matrix_function_range) - 1:
            print()

if __name__ == "__main__":
    main()

    # run_multiple_matrix_functions(["data/Silicon_4"], [1, 4])
