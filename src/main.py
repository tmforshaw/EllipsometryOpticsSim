import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Import the functions from the light_traversal.py file
from light_traversal import get_rotated_linear_polariser_matrix, get_rotated_quarter_wave_plate, get_sample_matrix, get_matrix_from_psi_delta
from helpers import *

# Select the matrix to use for the sample
SAMPLE_MATRIX_FUNCTION = 4 

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
    intensities = np.abs(np.sum(final_field_strength, axis=2).reshape(len(final_field_strength)) / 2) + y_offset 

    # intensities = np.sum(np.abs(final_field_strength) ** 2, axis=2).reshape(len(final_field_strength)) + y_offset

    intensities /= max(intensities)

    return intensities

# Calculates the expected intensity of the light, for a range of compensator angles, using the Jones' matrix ray transfer method
def get_expected_intensities_psi_delta(compensator_angles, psi, delta, x_offset = 0, y_offset = 0):
    original_field_strength = np.array([1, 0]) # Parallel Linearly-Polarised Light

    sample_mat = get_matrix_from_psi_delta(psi, delta)

    # polarisation_mat = get_rotated_linear_polariser_matrix(0)
    # analyser_mat = get_rotated_linear_polariser_matrix(np.pi / 2)

    polarisation_mat = get_rotated_linear_polariser_matrix(-np.pi/4)
    analyser_mat = get_rotated_linear_polariser_matrix(np.pi / 4)

    # Multiply the Jones' matrices in reverse order to represent the light-ray traversal, then multiply by the field strength vector to apply this combined matrix to it
    final_field_strength = np.array([analyser_mat @ get_rotated_quarter_wave_plate(compensator_angle) @ sample_mat @ polarisation_mat @ original_field_strength for compensator_angle in compensator_angles])# Use @ instead of * to allow for different sized matrices to be dot-producted together

    # Find the effective reflection and take the absolute value so it's real
    # R_effective = (R_paralell + R_perpendicular) / 2
    intensities = np.abs(np.sum(final_field_strength, axis=2).reshape(len(final_field_strength)) / 2) + y_offset 

    # intensities = np.sum(np.abs(final_field_strength) ** 2, axis=2).reshape(len(final_field_strength)) + y_offset

    intensities /= max(intensities)

    return intensities

# Define the initial guesses and bounds for the fitting
def get_guesses_and_bounds():
    # Initial guesses and bounds for parameters
    n_angle_guess,      n_angle_bounds      = get_default_brewsters_angle(), [50 * np.pi/180, 60 * np.pi/180]
    # n_gold_guess,       n_gold_bounds       = 0.18508,                       [0.1, 0.2]
    # k_gold_guess,       k_gold_bounds       = 3.4233,                        [3, 4]
    n_gold_guess,       n_gold_bounds       = 0.18508,                       [0.18505, 0.18510]
    k_gold_guess,       k_gold_bounds       = 3.4233,                        [3.4230, 3.4235]
    d_guess,            d_bounds            = 40e-9,                         [10e-9, 150e-9]
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
    print_parameters_nicely(optimal_param, param_err, names=["Angle", "N_gold", "K_gold", "d", "X-Offset", "Y-Offset"], units=["Degrees", "", "", "Metres", "Degrees", ""], conversions = [180/np.pi, 1, 1, 1, 180/np.pi, 1])

    return optimal_param, param_err

# Fits the provided data to the expected intensities, returning the optimal parameters which were found, along with their errors
def fit_data_to_expected_psi_delta(compensator_angles, measured_intensities, intensity_uncertainties):
    # Get the initial guesses and bounds for each parameter
    initial_guesses, bounds = get_guesses_and_bounds()
    
    bounds = np.array([[0, np.pi], [0, np.pi], [-np.pi/8, np.pi/8], [0, 1]])

    # Fit the data using these initial parameters
    optimal_param, param_convolution = curve_fit(get_expected_intensities_psi_delta, compensator_angles, measured_intensities, p0=[20 * np.pi/180, 45 * np.pi/180,0,0], bounds=(bounds[::, 0], bounds[::, 1]))

    # Calculate errors from convolution matrix
    param_err = np.sqrt(np.diag(param_convolution))

    # Print out the values, along with their errors
    print_parameters_nicely(optimal_param, param_err, names=["Psi", "Delta", "X-Offset", "Y-Offset"], units=["Degrees", "Degrees", "", ""], conversions=[180/np.pi, 180/np.pi, 1, 1])

    return optimal_param, param_err

# Read the data from a file, then plot this data, alongside the expected intensities from the optimal fitting of the data
def fit_from_data(filenames):
    for i, filename in enumerate(filenames):
        print("Fitting {}:".format(filename))
    
        # Read the data and seperate it into x and y values
        data = read_file_to_data(filename)
        data_x, data_y = data[0], data[1]

        # Smooth the data
        data_x, data_y = smooth_data(data_x, data_y)

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

        expected_y = get_expected_intensities(data_x, *optimal_param)
        chi_sqr = np.sum((data_y - expected_y) ** 2 / expected_y)

        print("Goodness of fit: ", chi_sqr)

        # # Fit the data to the function
        # optimal_param, param_err = fit_data_to_expected_psi_delta(data_x, data_y, sigma)
        # plt.plot(data_x * 180 / np.pi, get_expected_intensities_psi_delta(data_x, *optimal_param), c='k', ls="-", label="Calculated Result{}".format(label_modifier)) # Plot the light intensity expected for the fitted parameters

        # Plot the measured data to the same figure
        # plt.errorbar(data_x * 180 / np.pi, data_y, c='r', alpha=0.2, yerr=sigma, fmt='o', label="Intensity Data")
        plt.scatter(data_x * 180 / np.pi, data_y, c='r', label="Intensity Data{}".format(label_modifier), lw=4)
        plt.fill_between(data_x * 180 / np.pi, data_y - sigma, data_y + sigma, color="r", alpha=0.2, label="Errors on Intensity Data")

        if i < len(filenames) - 1:
            print()

    # plt.legend()
    plt.tight_layout()
    plt.show()

def plot_from_data(filenames):
    for filename in filenames:
        print("Plotting {}:".format(filename))
    
        # Read the data and seperate it into x and y values
        data = read_file_to_data(filename)
        data_x, data_y = data[0], data[1]

        # Smooth the data
        data_x, data_y = smooth_data(data_x, data_y)

        # TODO Normalise the data
        data_y = normalise_data(data_y)

        # Format the figure and plot
        format_plot(max(data_y))

        # A modifier to add the filenamae if there are multiple files
        label_modifier = " {}".format(filename) if len(filenames) > 1 else ""

        # Plot the measured data to the same figure
        plt.scatter(data_x * 180 / np.pi, data_y, label="Intensity Data{}".format(label_modifier))

    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    print_sample_matrix_type(SAMPLE_MATRIX_FUNCTION)

    plot_from_data(["data/Gold_Phi_1", "data/Gold_Phi_2_(204)", "data/Gold_Phi_3_(206)", "data/Gold_Phi_4_(208)", "data/Gold_Phi_5_(210)", "data/Gold_Phi_6_(198)", "data/Gold_Phi_7_(196)", "data/Gold_Phi_8_(194)", "data/Gold_Phi_192", "data/Gold_Phi_212", "data/Gold_Phi_214"])

    # fit_from_data(["data/Gold_Phi_1", "data/Gold_Phi_2_(204)", "data/Gold_Phi_3_(206)", "data/Gold_Phi_4_(208)", "data/Gold_Phi_5_(210)", "data/Gold_Phi_6_(198)", "data/Gold_Phi_7_(196)", "data/Gold_Phi_8_(194)"])

    # fit_from_data(["data/Gold_C_45_45_3", "data/Gold_I_2"])
    # plot_from_data(["data/Gold_I_1"])

    # fit_from_data(["data/Gold_Ficc_1", "data/Gold_Ficc_2", "data/Gold_Ficc_3"])

    # fit_from_data(["data/Gold_52nm_2"])
    # fit_from_data(["data/Gold_52nm_1", "data/Gold_52nm_2",  "data/Gold_65nm_1"])
    # plot_from_data(["data/Gold_52nm_1", "data/Gold_52nm_2",  "data/Gold_65nm_1"])

    # fit_from_data(["data/Gold_65nm_1", "data/Gold_65nm_2",])
    # fit_from_data(["data/Gold_Phi_194"])

    # fit_from_data(["data/Gold_Phi_192"])

    # fit_from_data(["data/Glass_4"])

    # plot_default()

    # plot_range_of_depths()
    # plot_range_of_brewsters()

def run_multiple_matrix_functions(filenames, function_range = range(4)):
    for i, matrix_index in enumerate(function_range):
        # Update global variable to change which sample matrix function is used
        global SAMPLE_MATRIX_FUNCTION
        SAMPLE_MATRIX_FUNCTION = matrix_index

        print_sample_matrix_type(matrix_index)
        fit_from_data(filenames)

        if i < len(function_range) - 1:
            print()

if __name__ == "__main__":
    main()

    # run_multiple_matrix_functions(["data/Gold_52nm_1"], [1, 3, 4])
