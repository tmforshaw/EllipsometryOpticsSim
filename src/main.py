import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Import the functions from the light_traversal.py file
from light_traversal import get_rotated_linear_polariser_matrix, get_rotated_quarter_wave_plate, get_sample_matrix

# The default values for known parameters
def get_default_refractive_index_param():
    n_air, k_air = 1, 0
    n_glass, k_glass = 1.5, 0
    return ((n_air, k_air), (n_glass, k_glass))

def get_default_brewsters_angle():
    (n_air, _), (n_glass, _) = get_default_refractive_index_param()
    return np.arctan(n_glass / n_air)

# Calculates the expected intensity of the light, for a range of compensator angles, using the Jones' matrix ray transfer method
def get_expected_intensities(compensator_angles, amplitude, sample_angle_of_incidence, n_gold, k_gold, d, wavelength, offset = 0):
    # Get the default parameters for this traversal
    (n_air, k_air), (n_glass, k_glass) = get_default_refractive_index_param()
    # original_field_strength = np.array([1, 1]) / np.sqrt(2) # Normalised 45 degree, linearly-polarised light
    original_field_strength = np.array([1, 0]) # Parallel Linearly-Polarised Light

    sample_mat = get_sample_matrix(sample_angle_of_incidence, n_air, n_gold, n_glass, k_air, k_gold, k_glass, d, wavelength)

    # polariser_angle = np.pi / 4 # 45 Degree Angle
    # polarisation_mat = get_rotated_linear_polariser_matrix(polariser_angle)
    # analyser_mat = get_rotated_linear_polariser_matrix(-polariser_angle)

    polarisation_mat = get_rotated_linear_polariser_matrix(0)
    analyser_mat = get_rotated_linear_polariser_matrix(np.pi / 2)

    # Multiply the Jones' matrices in reverse order to represent the light-ray traversal, then multiply by the field strength vector to apply this combined matrix to it
    final_field_strength = np.array([analyser_mat @ get_rotated_quarter_wave_plate(compensator_angle) @ sample_mat @ polarisation_mat @ original_field_strength for compensator_angle in compensator_angles]) + offset # Use @ instead of * to allow for different sized matrices to be dot-producted together

    # Find the effective reflection and take the absolute value so it's real
    # R_effective = (R_paralell + R_perpendicular) / 2
    intensities = amplitude * np.abs(np.sum(final_field_strength, axis=2).reshape(len(final_field_strength)) / 2) 

    return intensities

# Fits the provided data to the expected intensities, returning the optimal parameters which were found, along with their errors
def fit_data_to_expected(compensator_angles, measured_intensities, intensity_uncertainties):
    # Initial guesses for parameters
    amplitude_guess = 1
    n_angle_guess = get_default_brewsters_angle()
    n_gold_guess, k_gold_guess = 3.85, -0.22
    d_guess = 50e-9
    wavelength_guess = 632.8e-9
    offset_guess = 0

    # Combine initial guesses into single array
    initial_guesses = [amplitude_guess, n_angle_guess, n_gold_guess, k_gold_guess, d_guess, wavelength_guess, offset_guess]

    # Bounds for each guess
    amplitude_bounds = [0.001, 1]
    n_angle_bounds = [0, np.pi]
    n_gold_bounds = [0, 5]
    k_gold_bounds = [-1, 5]
    d_bounds = [0.1e-9, 100e-9]
    wavelength_bounds = [100e-9, 1000e-9]
    offset_bounds = [0, 1]

    # Combine bounds into single array
    bounds = np.array([amplitude_bounds, n_angle_bounds, n_gold_bounds, k_gold_bounds, d_bounds, wavelength_bounds, offset_bounds])

    # Fit the data using these initial parameters
    optimal_param, param_convolution = curve_fit(get_expected_intensities, compensator_angles, measured_intensities, p0=initial_guesses, bounds=(bounds[::, 0], bounds[::, 1]), sigma=intensity_uncertainties, method="trf")

    param_err = np.sqrt(np.diag(param_convolution))

    print("amplitude: {:.4G} +- {:.4G}\nangle: {:.4G} +- {:.4G}\nn_gold: {:.4G} +- {:.4G}\nk_gold: {:.4G} +- {:.4G}\nd: {:.4G} +- {:.4G}\nwavelength: {:.4G} +- {:.4G}\noffset: {:.4G} +- {:.4G}".format(optimal_param[0], param_err[0], optimal_param[1] * 180 / np.pi, param_err[1] * 180 / np.pi, optimal_param[2], param_err[2], optimal_param[3], param_err[3], optimal_param[4], param_err[4], optimal_param[5], param_err[5], optimal_param[6], param_err[6]))

    return optimal_param, param_err

# Read the data from a file and split it into the angle data and the intensity data
def read_file_to_data(filename):
    angles, intensities = [], []

    with open(filename) as file:
        for i, line in enumerate(file):
            line = line.strip().split('\t')

            angles.append(float(line[0]) * np.pi / 180) # Convert degrees to radians
            intensities.append(float(line[1]))

        # Ensure that both arrays of data are the same length
        assert len(angles) == len(intensities), "Data arrays are of different length:\n\tAngles: {}\tIntensities: {}".format(len(angles), len(intensities))

    return (np.array(angles), np.array(intensities))

def format_plot(y_max):
    # Set the font size for the title and axes labels
    plt.rc('axes', titlesize=20, labelsize=20)
    plt.rc('legend', fontsize=15)

    # Set the minor and major tick labels to different sizes
    plt.tick_params(axis='both', which='minor', labelsize=10)
    plt.tick_params(axis='both', which='major', labelsize=15)

    # Set the grid to be visible for multiples of 15 degrees and 45 degrees
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MultipleLocator(45)) # Set the tick sizes for major ticks
    ax.xaxis.set_minor_locator(plt.MultipleLocator(15)) # Set the tick sizes for minor ticks
    ax.xaxis.set_minor_formatter(plt.FormatStrFormatter("%.d")) # Show the minor ticks with labels

    # Show a grid with more prominent grid-lines for the major ticks
    plt.grid(visible = True, axis="x", which = "major", ls="-", c="k")
    plt.grid(visible = True, axis="x", which = "minor", ls="-.")

    # Remove the margins around the data
    plt.margins(x=0, y=y_max / 50, tight=True)

# Read the data from a file, then plot this data, alongside the expected intensities from the optimal fitting of the data
def fit_from_data(filename):
    print("Data for {}:\n".format(filename))
    
    # Read the data and seperate it into x and y values
    data = read_file_to_data(filename)
    data_x, data_y = data[0], data[1]

    # Format the figure and plot
    format_plot(max(data_y))

    # Set the title and axes labels for this plot
    plt.title(r"Light Intensity $\left(\frac{I_{Final}}{I_{Initial}}\right)$ vs Compensator Angle")
    plt.xlabel("Compensator Angle [Degrees]")
    plt.ylabel(r"Normalised Light Intensity ($I_{final} \div I_{initial}$)")

    # x = np.linspace(-np.pi/2, np.pi/2, num=500, endpoint=True)
    # plt.plot(x, get_expected_intensities([], 1, 56, 0.18, 3.4432, 50e-9, 600e-9))

    # Uncertainty in the y-data
    sigma_absolute = 0.05 * max(data_y)
    sigma = np.array([sigma_absolute for _ in data_y])

    # Fit the data to the function
    optimal_param, param_err = fit_data_to_expected(data_x, data_y, sigma)

    # Plot the light intensity expected for the fitted parameters
    x = np.linspace(min(data_x), max(data_x), num=len(data_x), endpoint=True)
    plt.plot(x * 180 / np.pi, get_expected_intensities(x, *optimal_param), c='k', ls="-", label="Calculated Result")

    # Plot the measured data to the same figure
    # plt.errorbar(data_x * 180 / np.pi, data_y, c='r', alpha=0.2, yerr=sigma, fmt='o', label="Intensity Data")
    plt.scatter(data_x * 180 / np.pi, data_y, c='r', label="Intensity Data", lw=4)
    # plt.fill_between(data_x * 180 / np.pi, data_y - sigma, data_y + sigma, color="r", alpha=0.2, label="Errors on Intensity Data")

    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_from_data(filename):
    print("Data for {}:\n".format(filename))
    
    # Read the data and seperate it into x and y values
    data = read_file_to_data(filename)
    data_x, data_y = data[0], data[1]

    # Format the figure and plot
    format_plot(max(data_y))

    # Set the title and axes labels for this plot
    plt.title(r"Light Intensity $\left(\frac{I_{Final}}{I_{Initial}}\right)$ vs Compensator Angle")
    plt.xlabel("Compensator Angle [Degrees]")
    plt.ylabel(r"Normalised Light Intensity ($I_{final} \div I_{initial}$)")

    # Plot the measured data to the same figure
    plt.scatter(data_x * 180 / np.pi, data_y, c='r', label="Intensity Data")

    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_range_of_wavelengths():
    wavelengths = np.linspace(250e-9, 900e-9, num=15)

    brewsters_angle = get_default_brewsters_angle()

    for wavelength in wavelengths:
        x = np.linspace(-np.pi/2, np.pi/2, num=300, endpoint=True)
        plt.plot(x * 180 / np.pi, get_expected_intensities(x, 1, brewsters_angle, 0.18, 3.4432, 50e-9, wavelength), ls="-", label=r"$\lambda = {:.4G}$".format(wavelength))

    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_range_of_depths():
    depths = np.linspace(0.1e-9, 100e-9, num=15)

    brewsters_angle = get_default_brewsters_angle()

    for depth in depths:
        x = np.linspace(-np.pi/2, np.pi/2, num=300, endpoint=True)
        plt.plot(x * 180 / np.pi, get_expected_intensities(x, 1, brewsters_angle, 0.18, 3.4432, depth, 632e-9), ls="-", lw=3, label="$d = {:.4G}$".format(depth))

    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_range_of_brewsters():
    N = 15
    brewsters = np.linspace(np.pi/N, np.pi, num=N, endpoint=True)

    for brewster in brewsters:
        x = np.linspace(-np.pi/2, np.pi/2, num=300, endpoint=True)
        plt.plot(x * 180 / np.pi, get_expected_intensities(x, 1, brewster, 0.18, 3.4432, 50e-9, 632e-9), ls="-", lw=3, label=r"$\theta_B = {:.4G}$".format(brewster))

    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_default():
    brewsters_angle = get_default_brewsters_angle()

    x = np.linspace(0, np.pi * 2, num=300, endpoint=True)
    plt.plot(x * 180 / np.pi, get_expected_intensities(x, 1, brewsters_angle, 3.443, -0.022, 50e-9, 632e-9), ls="-", lw=3, label="Plot")

    plt.legend()
    plt.tight_layout()
    plt.show()

# plot_default()

fit_from_data("data/Gold_B_3")
# plot_from_data("data/Gold_B_3")
# fit_from_data("data/Glass_3")


# plot_range_of_wavelengths()
# plot_range_of_depths()
# plot_range_of_brewsters()
