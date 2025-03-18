import numpy as np

def get_rotation_matrix(theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return np.matrix([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

# A linear polariser which is rotated about an angle theta (w.r.t horizontal)
def get_rotated_linear_polariser_matrix(theta):
    # Rotate horizontally (parallel) polarised light by an angle theta
    return get_rotation_matrix(theta) @ np.matrix([[1, 0], [0, 0]]) @ get_rotation_matrix(-theta)

# A quarter waveplate which is rotated about an angle theta (w.r.t horizontal) [The Compensator]
def get_rotated_quarter_wave_plate(theta):
    delta = np.pi / 2

    return get_rotation_matrix(theta) @ np.matrix([[1, 0], [0, np.exp(-1j * delta)]]) @ get_rotation_matrix(-theta)

#Describe the phase offset and fast-axis angle in a Jones' matrix
def get_matrix_from_psi_delta(psi, delta):
    return np.matrix([[np.sin(psi) * np.exp(1j * delta), 0], [0, np.cos(psi)]])

# Fresnel equations used in "Determination of refractive index and layer thickness of nm-thin films via ellipsometry" by Peter Nestler and Christiane A. Helm
def get_fresnel_thin_film_hardcoded(theta_incoming, N_air, N_gold, N_glass, d, wavelength):
    # TODO Find out whether this needs to be N_glass or N_gold
    theta_refracted = np.asin((N_air / N_glass) * np.sin(theta_incoming))

    # Account for the different paths by adding phase differences (Utilising complex numbers)
    A = (N_gold ** 2) * np.cos(theta_incoming) * np.cos(theta_refracted)
    A_plus_minus = N_air * N_glass + (N_air * (N_glass ** 3) * (np.sin(theta_refracted) ** 2) / (N_gold ** 2))
    A_plus, A_minus = A - A_plus_minus, A + A_plus_minus

    B = N_air * N_glass * np.cos(theta_incoming) * np.cos(theta_refracted)
    B_plus_minus = (N_glass ** 2) * (np.sin(theta_refracted) ** 2) - (N_gold ** 2)
    B_plus, B_minus = B + B_plus_minus, B - B_plus_minus

    # TODO This is a fudge, should be 2
    phase_diff = 1 * np.pi * d / wavelength

    R_parallel = (N_glass * np.cos(theta_incoming) - N_air * np.cos(theta_refracted) + 1j * phase_diff * A_plus) / (N_glass * np.cos(theta_incoming) + N_air * np.cos(theta_refracted) + 1j * phase_diff * A_minus)
    R_perpendicular = (N_air * np.cos(theta_refracted) - N_glass * np.cos(theta_incoming) + 1j * phase_diff * B_plus) / (N_air * np.cos(theta_refracted) + N_glass * np.cos(theta_incoming) + 1j * phase_diff * B_minus)

    R = np.array([R_parallel, R_perpendicular])

    ratio = R[0] / R[1]
    psi = np.atan(np.abs(ratio))
    delta = np.angle(ratio)

    # print("Psi: {:.4G}\tDelta: {:.4G}".format(np.degrees(psi), np.degrees(delta)))

    return get_matrix_from_psi_delta(psi, delta)

# A helper function which can be used to switch out the sample matrix easily
def get_sample_matrix(sample_angle_of_incidence, N_air, N_gold, N_glass, d, wavelength):
    # d = 0
    return get_fresnel_thin_film_hardcoded(sample_angle_of_incidence, N_air, N_gold, N_glass, d, wavelength)
