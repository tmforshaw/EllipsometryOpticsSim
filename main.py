import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# A linear polariser which is rotated about an angle theta (w.r.t horizontal)
def get_rotated_linear_polariser_matrix(theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return np.matrix([[cos_theta ** 2, cos_theta * sin_theta], [cos_theta * sin_theta, sin_theta ** 2]])

# A quarter waveplate which is rotated about an angle theta (w.r.t horizontal)
def get_rotated_quarter_wave_plate(theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    combined_sin_cos = (1 - 1j) * sin_theta * cos_theta

    return np.exp(-1j * np.pi / 4) * np.matrix([[cos_theta ** 2 + 1j * sin_theta ** 2, combined_sin_cos], [combined_sin_cos, sin_theta ** 2 + 1j * cos_theta ** 2]])

def get_freespace_matrix(d):
    return np.matrix([[1, d], [0, 1]])

# Uses a more in-depth version of Snell's law to describe how the parallel and perpendicular components of the electric field are reflected differently
# returns an array with the parallel and perpendicular components of the reflected light [0-1]
# also returns the refracted angle, at which light has passed through the boundary
def modified_snells_law(theta_incoming, n1, n2, k1, k2):
    # Create complex variables from the components of the complex refractive indices
    N1 = np.complex64(n1, k1)
    N2 = np.complex64(n2, k2)
    
    # Snell's law to find the outgoing angle
    theta_refracted = np.asin(N1 / N2 * np.sin(theta_incoming))

    # Calculate the reflected components for both the parallel and perpendicular components
    R_parallel = (np.tan(theta_incoming - theta_refracted) ** 2) / (np.tan(theta_incoming + theta_refracted) ** 2)
    R_perpendicular = (np.sin(theta_incoming - theta_refracted) ** 2) / (np.sin(theta_incoming + theta_refracted) ** 2)

    return (np.array([R_parallel, R_perpendicular]), theta_refracted)

# Use the modified Snell's law to describe the full traversal of light through the thin gold film: air -> gold -> reflection from glass -> air
def get_thin_film_matrix(theta_incoming, n_air, n_gold, n_glass, k_air, k_gold, k_glass, d, wavelength):
    # Light enters gold via air-gold interface
    R_air_gold, theta_refracted_air_gold = modified_snells_law(theta_incoming, n_air, n_gold, k_air, k_gold)

    # Light which was transmitted through the gold is reflected from the gold-glass interface
    R_gold_glass_at_100_percent, _ = modified_snells_law(theta_refracted_air_gold, n_gold, n_glass, k_gold, k_glass)
    # The true amount of reflected light present from this reflection (as a percentage of initial light intensity) Transmited, then reflected
    R_gold_glass = (1 - R_air_gold) * R_gold_glass_at_100_percent

    # Light which is reflected from the gold-glass interfacae, and must be transmitted through the gold-air interface (The angle is the same as the incoming angle previously)
    R_gold_air_at_100_percent, _ = modified_snells_law(theta_refracted_air_gold, n_gold, n_air, k_gold, k_air)
    # The true amount of light which was reflected both times  Transmitted, Reflected, then Reflected Again
    R_gold_air = R_gold_glass * R_gold_air_at_100_percent

    # The true transmitted light after being transmitted, reflected, then transmimtted again
    T_gold_air = R_gold_glass * (1 - R_gold_air_at_100_percent)

    # # Sum the light which was initially reflected, and the light which was transmitted after all light bounces through the film
    T_total = R_air_gold + T_gold_air + R_gold_glass
    # T_total = R_gold_glass

    ratio = T_total[0] / T_total[1]
    
    # TODO get distance and wavelength involved in a better way
    # Psi is the angle at which the fast axis acts (the semi-major axis of the ellipse)
    psi = np.pi - np.atan2(np.abs(T_total[0]), np.abs(T_total[1]))

    psi = np.pi - np.atan2(np.abs(T_total[0]), np.abs(T_total[1]))

    # Delta is the difference between the phase offset given, by this traversal, to the parallel and perpendicular components
    delta = n_gold * d / wavelength * 2 * (np.pi - np.atan2(np.imag(ratio), np.real(ratio)))
    # delta = (n_gold + n_air + n_glass) * d / wavelength * 2 * (np.pi - np.atan2(np.imag(ratio), np.real(ratio)))
    # delta = (d/wavelength * 2 * np.pi) * (n_gold * np.cos(theta_refracted_air_gold) + n_air * np.cos(theta_incoming))

    # delta = (2 * np.pi / wavelength * d) * np.cos(theta_refracted_air_gold)

    # delta = 2 * np.pi * (np.modf(np.real((2 * d * n_gold) * np.cos(theta_refracted_air_gold)) / wavelength)[0] + 0.5)
    # print(delta)

    #Describe this phase offset and the magnitude in a Jones' matrix
    return (np.matrix([[np.sin(psi) * np.exp(1j * delta), 0], [0, np.cos(psi)]]))

def jonesmodelthinfilm( wavelength , angle , n1 , k1 , n2 , k2 , d ) :
    import cmath
    N1 = complex ( n1 , k1 ) #D e f i n e Gold r e f r a c t i v e i n d e x v a r i a b l e s
    N2 = complex ( n2 , k2 ) #D e f i n e G l a s s r e f r a c t i v e i n d e x v a r i a b l e s

    anglerad = angle * ( np.pi /180) #c o n v e r t B rew ste r ' s a n g l e t o ←-


    beta = 2 * np.pi * ( d / wavelength ) * cmath.sqrt(N1**2 - np.sin(anglerad)**2)

    #d e f i n e components t o b u i l d t h e c o e f f i c i e n t o f r e f l e c t i o n f o r ←-
    # s-polarised and p-polarisedformulas
    c_t_0 = np.cos(anglerad)
    c_t_1 = cmath.sqrt((1-(1/N1**2))*np.sin(anglerad)**2)
    c_t_2 = cmath.sqrt((1-(1/N2**2))*np.sin(anglerad)**2)

    r01p = (N1* c_t_0 - c_t_1 ) / (N1* c_t_0 + c_t_1 )
    r01s= ( c_t_0 - N1* c_t_1 ) / ( c_t_0 + N1* c_t_1 )
    r12p = (N2* c_t_1 - N1* c_t_2 ) / (N2* c_t_1 + N1* c_t_2 )
    r12s= (N1* c_t_1 - N2* c_t_2 ) / (N1* c_t_1 + N2* c_t_2 )
    # C o e f f o f r e f l e c t i o n f o r p- and s- p o l a r i s e d l i g h t
    Rp = ( r01p + r12p * np.exp(-1j * 2 * beta ) ) / ( 1 + r01p * r12p* np.exp(-1j * 2 * beta ) )
    Rs=(r01s+r12s*np.exp(-1j*2*beta))/(1+r01s*r12s*np.exp(-1j*2*beta))
    #calculatepsianddeltafromRsandRp
    psi=(np.pi-np.arctan2(np.abs(Rp),np.abs(Rs)))
    delta=(np.pi-np.arctan2(np.imag(Rp/Rs),np.real(Rp/Rs)))
    #BuildJonesmatrixforreflectedlightusingpsianddelta
    J_g=np.array([[np.sin(psi)*np.exp(1j*delta),0.0],[0.0,np.cos(psi)]])

    return J_g

def get_sample_matrix(sample_angle_of_incidence, n_air, n_gold, n_glass, k_air, k_gold, k_glass, d, wavelength):
    return get_thin_film_matrix(sample_angle_of_incidence, n_air, n_gold, n_glass, k_air, k_gold, k_glass, d, wavelength)
    # return jonesmodelthinfilm(wavelength, sample_angle_of_incidence, n_gold, k_gold, n_glass, k_glass, d)

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
    original_field_strength = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]) # Normalised 45 degree, linearly-polarised light

    sample_mat = get_sample_matrix(sample_angle_of_incidence, n_air, n_gold, n_glass, k_air, k_gold, k_glass, d, wavelength)

    polariser_angle = np.pi / 4 # 45 Degree Angle
    polarisation_mat = get_rotated_linear_polariser_matrix(polariser_angle)
    analyser_mat = get_rotated_linear_polariser_matrix(-polariser_angle)

    # Multiply the Jones' matrices in reverse order to represent the light-ray traversal, then multiply by the field strength vector to apply this combined matrix to it
    final_field_strength = np.array([analyser_mat @ get_rotated_quarter_wave_plate(compensator_angle) @ sample_mat @ polarisation_mat @ original_field_strength for compensator_angle in compensator_angles]) # Use @ instead of * to allow for different sized matrices to be dot-producted together

    # Normalise each field strength vector to get its intensity (taking the absolute value to ensure the answer is real)
    intensities = amplitude * np.linalg.norm(np.linalg.norm(final_field_strength,axis=1) ** 2, axis=1) + offset
    # intensities = amplitude * np.linalg.norm(np.flip(np.linalg.norm(final_field_strength,axis=1) ** 2), axis=1)

    return intensities

# Fits the provided data to the expected intensities, returning the optimal parameters which were found, along with their errors
def fit_data_to_expected(compensator_angles, measured_intensities, intensity_uncertainties):
    # Initial guesses for parameters
    amplitude_guess = 1
    n_angle_guess = get_default_brewsters_angle()
    n_gold_guess, k_gold_guess = 0.18, 3.4432
    d_guess = 50e-9
    wavelength_guess = 630e-9
    offset_guess = 0

    # Combine initial guesses into single array
    initial_guesses = [amplitude_guess, n_angle_guess, n_gold_guess, k_gold_guess, d_guess, wavelength_guess, offset_guess]

    # Bounds for each guess
    amplitude_bounds = [0.0001, 1]
    n_angle_bounds = [-np.pi/2, np.pi/2]
    n_gold_bounds = [0, 10]
    k_gold_bounds = [0, 10]
    d_bounds = [0.1e-9, 100e-9]
    wavelength_bounds = [100e-9, 1000e-9]
    offset_bounds = [0, 1]

    # Combine bounds into single array
    bounds = np.array([amplitude_bounds, n_angle_bounds, n_gold_bounds, k_gold_bounds, d_bounds, wavelength_bounds, offset_bounds])

    # Fit the data using these initial parameters
    optimal_param, param_convolution = curve_fit(get_expected_intensities, compensator_angles, measured_intensities, p0=initial_guesses, bounds=(bounds[::, 0], bounds[::, 1]), sigma=intensity_uncertainties, method="trf")

    param_err = np.sqrt(np.diag(param_convolution ))

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
def read_data_and_plot(filename):
    # Read the data and seperate it into x and y values
    data = read_file_to_data(filename)
    data_x, data_y = data[0], data[1]

    # # Clean the data up
    # avg_data_y = np.sum(data_y) / len(data_y)
    # new_data_indices = np.where(abs(data_y) < avg_data_y * 1.5)
    # data_x = data_x[new_data_indices]
    # data_y = data_y[new_data_indices]

    # brewsters_angle = get_default_brewsters_angle()
    # data = np.zeros(200)
    # data_x = np.linspace(0, np.pi, num=len(data), endpoint = True)
    # data_y = get_expected_intensities(data_x, 1, brewsters_angle, 0.18, 3.4432, 50e-9, 632e-9)

    # Format the figure and plot
    format_plot(max(data_y))

    # Set the title and axes labels for this plot
    plt.title(r"Light Intensity $\left(\frac{I_{Final}}{I_{Initial}}\right)$ vs Compensator Angle")
    plt.xlabel("Compensator Angle [Degrees]")
    plt.ylabel(r"Normalised Light Intensity ($I_{final} \div I_{initial}$)")

    # Uncertainty in the y-data
    sigma_absolute = 0.001
    sigma = np.array([sigma_absolute for _ in data_y])

    # Fit the data to the function
    optimal_param, param_err = fit_data_to_expected(data_x, data_y, sigma)

    # Plot the light intensity expected for the fitted parameters
    x = np.linspace(min(data_x), max(data_x), num=len(data_x), endpoint=True)
    plt.plot(x * 180 / np.pi, get_expected_intensities(x, *optimal_param), c='k', ls="-", label="Calculated Result")

    # Plot the measured data to the same figure
    plt.scatter(data_x * 180 / np.pi, data_y, c='r', label="Intensity Data", lw=4)
    # plt.errorbar(data_x * 180 / np.pi, data_y, c='r', yerr=sigma, fmt='o', label="Intensity Data")
    # plt.scatter(data_x * 180 / np.pi, data_y, c='r', label="Intensity Data")
    # plt.fill_between(data_x * 180 / np.pi, data_y - sigma, data_y + sigma, color="r", alpha=0.2, label="Errors on Intensity Data")

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

read_data_and_plot("data/FirstRun_Gold")

# plot_range_of_wavelengths()
# plot_range_of_depths()
# plot_range_of_brewsters()
