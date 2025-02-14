import numpy as np
import matplotlib.pyplot as plt

# The default values for known parameters
def get_default_refractive_index_param():
    N_air = get_complex_refractive_index(1, 0)
    N_glass = get_complex_refractive_index(1.457, 0)
    return (N_air, N_glass)

def get_default_brewsters_angle():
    # N_air, N_glass = get_default_refractive_index_param()
    # return np.arctan(np.real(N_glass) / np.real(N_air))
    return 55 * np.pi/180

def get_complex_refractive_index(n, k):
    return n + 1j * k

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

    # # Remove the margins around the data
    # plt.margins(x=0, y=y_max / 50, tight=True)
    plt.margins(x=0, tight=True)

    # Set the title and axes labels for this plot
    plt.title(r"Light Intensity $\left(\frac{I}{I_{max}}\right)$ vs Compensator Angle")
    plt.xlabel("Compensator Angle [Degrees]")
    plt.ylabel(r"Normalised Light Intensity ($I \div I_{max}$)")

def normalise_data(data_y):
    return data_y / max(data_y)

def print_sample_matrix_type(type):
    match type:
        case 0:
            type_string = "Snell's Law"
        case 1:
            type_string = "Fresnel"
        case 2:
            type_string = "2023"
        case 3:
            type_string = "Rihan's"
        case _:
            type_string = "Unknown"


    spacing_str = "{:-<8}".format("")
    print(f"{spacing_str}# Using {type_string} Matrix #{spacing_str}")

def print_parameters_nicely(values, errors, names, units, conversions = [180/np.pi, 1, 1, 1, 180/np.pi]):
    # Get the length of the longest name in the list
    max_name_len = len(max(names, key=len))

    # Adjust specific values to change their units from radians to degrees
    value_and_errors = list(zip(values, errors))
    for i, (value, err) in enumerate(value_and_errors):
        if i < len(conversions) and conversions[i] != 0:
                value *= conversions[i]
                err *= conversions[i]

                value_and_errors[i] = value, err

    # Create an array for the values and their associated errors
    value_err_strings = ["{:.4G} +- {:.4G}".format(value, err) for value, err in value_and_errors]
    max_value_err_len = len(max(value_err_strings, key=len))

    # Print out the (nicely spaced) names, values, errors, and units
    for i in range(len(value_and_errors)):
        # Pad the values, using escaped curly brackets "{{}}" so that the next format still works
        padded_string = "{{{{}}}}: {{:<{}}}{{{{}}}}{{:<{}}}[{{{{}}}}]".format(max_name_len - len(names[i]) + 2, max_value_err_len - len(value_err_strings[i]) + 2).format("","")

        print(padded_string.format(names[i], value_err_strings[i], units[i]))

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Plotting functions to plot default parameters, varying only one -------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# def plot_range_of_wavelengths():
#     from main import get_expected_intensities, get_guesses_and_bounds

#     wavelengths = np.linspace(250e-9, 900e-9, num=15)
#     param, _ = get_guesses_and_bounds()

#     for wavelength in wavelengths:
#         x = np.linspace(-np.pi/2, np.pi/2, num=300, endpoint=True)
#         plt.plot(x * 180 / np.pi, get_expected_intensities(x, param[0], param[1], param[2], param[3], param[4], wavelength), ls="-", label=r"$\lambda = {:.4G}$".format(wavelength))

#     plt.legend()
#     plt.tight_layout()
#     plt.show()

def plot_range_of_depths():
    from main import get_expected_intensities, get_guesses_and_bounds

    depths = np.linspace(30e-9, 80e-9, num=5, endpoint=True)
    param, _ = get_guesses_and_bounds()

    for depth in depths:
        x = np.linspace(0, np.pi * 2, num=300, endpoint=True)
        plt.plot(x * 180 / np.pi, get_expected_intensities(x, param[0], param[1], param[2], depth), ls="-", lw=3, label="$d = {:.4G}$".format(depth))

    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_range_of_brewsters():
    from main import get_expected_intensities, get_guesses_and_bounds

    brewsters = np.linspace(30 * np.pi/180, 60 * np.pi/180, num=15, endpoint=True)
    param, _ = get_guesses_and_bounds()

    for brewster in brewsters:
        x = np.linspace(0, np.pi, num=300, endpoint=True)
        plt.plot(x * 180 / np.pi, get_expected_intensities(x, brewster, param[1], param[2], param[3]), ls="-", lw=3, label=r"$\theta_B = {:.4G}$".format(brewster))

    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_default():
    from main import get_expected_intensities, get_guesses_and_bounds

    param, _ = get_guesses_and_bounds()

    x = np.linspace(0, np.pi * 2, num=300, endpoint=True)
    plt.plot(x * 180 / np.pi, get_expected_intensities(x, *param), ls="-", lw=3, label="Plot")

    plt.legend()
    plt.tight_layout()
    plt.show()
