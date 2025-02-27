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

# Uses a more in-depth version of Snell's law to describe how the parallel and perpendicular components of the electric field are reflected differently
# returns an array with the parallel and perpendicular components of the reflected light [0-1]
# also returns the refracted angle, at which light has passed through the boundary
def modified_snells_law(theta_incoming, N1, N2):
    # Snell's law to find the outgoing angle
    theta_refracted = np.asin(N1 / N2 * np.sin(theta_incoming))

    # Calculate the reflected components for both the parallel and perpendicular components
    R_parallel = (np.tan(theta_incoming - theta_refracted) ** 2) / (np.tan(theta_incoming + theta_refracted) ** 2)
    R_perpendicular = (np.sin(theta_incoming - theta_refracted) ** 2) / (np.sin(theta_incoming + theta_refracted) ** 2)

    return (np.array([R_parallel, R_perpendicular]), theta_refracted)

# Use the modified Snell's law to describe the full traversal of light through the thin gold film: air -> gold -> reflection from glass -> air
def get_snell_thin_film_matrix(theta_incoming, N_air, N_gold, N_glass, d, wavelength):
    # Light enters gold via air-gold interface
    R_air_gold, theta_refracted_air_gold = modified_snells_law(theta_incoming, N_air, N_gold)

    # Light which was transmitted through the gold is reflected from the gold-glass interface
    R_gold_glass_at_100_percent, _ = modified_snells_law(theta_refracted_air_gold, N_gold, N_glass)
    # The true amount of reflected light present from this reflection (as a percentage of initial light intensity) Transmited, then reflected
    R_gold_glass = (1 - R_air_gold) * R_gold_glass_at_100_percent

    # Light which is reflected from the gold-glass interfacae, and must be transmitted through the gold-air interface (The angle is the same as the incoming angle previously)
    R_gold_air_at_100_percent, _ = modified_snells_law(theta_refracted_air_gold, N_gold, N_air)
    # The true amount of light which was reflected both times  Transmitted, Reflected, then Reflected Again
    R_gold_air = R_gold_glass * R_gold_air_at_100_percent

    # The true transmitted light after being transmitted, reflected, then transmimtted again
    T_gold_air = R_gold_glass * (1 - R_gold_air_at_100_percent)

    # # Sum the light which was initially reflected, and the light which was transmitted after all light bounces through the film
    # T_total = R_air_gold + T_gold_air + R_gold_glass
    T_total = R_air_gold + R_gold_glass

    ratio = T_total[0] / T_total[1]

    # Psi is the angle at which the fast axis acts (the semi-major axis of the ellipse)
    psi = np.pi - np.atan2(np.abs(T_total[0]), np.abs(T_total[1]))

    # Delta is the difference between the phase offset given, by this traversal, to the parallel and perpendicular components
    delta = np.abs(N_gold * d * 4 * (np.pi - np.atan2(np.imag(ratio), np.real(ratio))))

    # print("Psi: {:.4G}\tDelta: {:.4G}".format(psi * 180/np.pi, delta* 180/np.pi))

    return get_matrix_from_psi_delta(psi, delta)

# Use the fresnel equations to calculate the reflected components of the light
def fresnel_reflection(theta_incoming, N1, N2, wavelength, d):
    theta_refracted = np.asin((N1 / N2) * np.sin(theta_incoming))

    # beta = 2 * np.pi * N1 / wavelength * d * np.cos(theta_refracted)
    beta = 2 * np.pi * ( d / wavelength ) * np.sqrt(N1**2 - np.sin(theta_refracted)**2)

    # R_parallel = np.abs((N1 * np.cos(theta_refracted) - N2 * np.cos(theta_incoming)) / (N1 * np.cos(theta_refracted) + N2 * np.cos(theta_incoming))) ** 2
    # R_perpendicular = np.abs((N1 * np.cos(theta_incoming) - N2 * np.cos(theta_refracted)) / (N1 * np.cos(theta_incoming) + N2 * np.cos(theta_refracted))) ** 2

    # R_parallel = (N1 * np.cos(theta_refracted) - N2 * np.cos(theta_incoming)) / (N1 * np.cos(theta_refracted) + N2 * np.cos(theta_incoming)) ** 2
    # R_perpendicular = (N1 * np.cos(theta_incoming) - N2 * np.cos(theta_refracted)) / (N1 * np.cos(theta_incoming) + N2 * np.cos(theta_refracted)) ** 2

    R_parallel = (N1 * np.cos(theta_refracted) - N2 * np.cos(theta_incoming)) / (N1 * np.cos(theta_refracted) + N2 * np.cos(theta_incoming))
    R_perpendicular = (N1 * np.cos(theta_incoming) - N2 * np.cos(theta_refracted)) / (N1 * np.cos(theta_incoming) + N2 * np.cos(theta_refracted))

    R_parallel *= np.exp(-1j * beta)
    R_perpendicular *= np.exp(-1j * beta)

    # R_parallel = -1j * beta
    # R_perpendicular = -1j * beta

    return np.array([R_parallel, R_perpendicular])

# A thin film matrix which uses the fresnel equations
def get_fresnel_thin_film_matrix(theta_incoming, N_air, N_gold, N_glass, d, wavelength):
    R_air_gold = fresnel_reflection(theta_incoming, N_air, N_gold, wavelength, d)
    R_gold_glass_at_100_percent = fresnel_reflection(theta_incoming, N_gold, N_glass, wavelength, d)
    R_gold_air_at_100_percent = fresnel_reflection(theta_incoming, N_gold, N_air, wavelength, d)

    # Transmitted into gold from air, reflected from glass back into gold, transmitted from gold into air
    T_gold_air = (1 - R_air_gold) * R_gold_glass_at_100_percent * (1 - R_gold_air_at_100_percent)

    T_air_gold_glass_gold_glass_gold_air = (1 - R_air_gold) * R_gold_glass_at_100_percent * R_gold_air_at_100_percent * R_gold_glass_at_100_percent * (1 - R_gold_air_at_100_percent)

    # beta_gold = 2 * np.pi * N_gold / wavelength * d * np.cos(theta_incoming)
    # beta_glass = 2 * np.pi * N_glass / wavelength * d * np.cos(theta_incoming)

    # Transmitted_Light = R_air_gold + T_gold_air * np.array([np.exp(-2j * beta_gold), 1 + np.exp(-2j * beta_gold)]) + T_air_gold_glass_gold_glass_gold_air * np.array([np.exp(-2j * (beta_gold + beta_glass)), 1 + np.exp(-2j * (beta_gold + beta_glass))]) 

    # Reflection from air-gold and the transmitted from gold-air
    # Transmitted_Light = R_air_gold + T_gold_air + T_air_gold_glass_gold_glass_gold_air
    Transmitted_Light = R_air_gold + T_gold_air
    # Transmitted_Light = R_air_gold
    # Transmitted_Light = R_air_gold + R_gold_glass_at_100_percent

    ratio = Transmitted_Light[0] / Transmitted_Light[1]

    psi = np.atan(np.abs(ratio))

    # delta = np.angle(ratio)

    delta = np.abs(4 * np.pi * N_gold / wavelength * d * np.sin(theta_incoming))
    # delta = np.abs(4 * np.pi * N_gold / wavelength * d * np.cos(theta_incoming))
    # delta = 4 * np.pi * N_gold / wavelength * d * np.sin(theta_incoming)
    # delta = 4 * np.pi * np.real(N_gold) / wavelength * d * np.cos(theta_incoming)

    # n_gold = np.real(N_gold)
    # k_gold = np.imag(N_gold)
    # psi = (n_gold * np.cos(theta_incoming) * np.sqrt(1 - (k_gold ** 2 / (n_gold ** 2 + k_gold ** 2)))) / (np.sqrt(n_gold ** 2 + k_gold ** 2) * np.sqrt(1 - (np.sin(theta_incoming) ** 2 / (n_gold ** 2 + k_gold ** 2))))

    # print("Mine:\t Psi: {:.4G}\tDelta: {:.4G}".format(psi * 180/np.pi, delta *180/np.pi))


    #Describe this phase offset and the magnitude in a Jones' matrix
    return get_matrix_from_psi_delta(psi, delta)

# Fresnel equations used in "Determination of refractive index and layer thickness of nm-thin films via ellipsometry" by Peter Nestler and Christiane A. Helm
def get_fresnel_thin_film_hardcoded(theta_incoming, N_air, N_gold, N_glass, d, wavelength):
    theta_refracted = np.asin((N_air / N_glass) * np.sin(theta_incoming))

    A = N_gold ** 2 * np.cos(theta_incoming) * np.cos(theta_refracted)
    A_plus_minus = N_air * N_glass + (N_air * N_glass ** 3 * np.sin(theta_refracted) ** 2 / N_gold ** 2)
    A_plus, A_minus = A - A_plus_minus, A + A_plus_minus

    B = N_air * N_glass * np.cos(theta_incoming) * np.cos(theta_refracted)
    B_plus_minus = N_glass ** 2 * np.sin(theta_refracted) ** 2 - N_gold ** 2
    B_plus, B_minus = B + B_plus_minus, B - B_plus_minus

    phase_diff = 2 * np.pi * d / wavelength

    R_parallel = (N_glass * np.cos(theta_incoming) - N_air * np.cos(theta_refracted) + 1j * phase_diff * A_plus) / (N_glass * np.cos(theta_incoming) + N_air * np.cos(theta_refracted) + 1j * phase_diff * A_minus)
    R_perpendicular = (N_air * np.cos(theta_refracted) - N_glass * np.cos(theta_incoming) + 1j * phase_diff * B_plus) / (N_air * np.cos(theta_refracted) + N_glass * np.cos(theta_incoming) + 1j * phase_diff * B_minus)

    R = np.array([R_parallel, R_perpendicular])
    Transmitted_Light = R

    ratio = Transmitted_Light[0] / Transmitted_Light[1]
    
    psi = np.atan(np.abs(ratio))
    # delta = np.pi + np.angle(ratio)
    delta = np.angle(ratio)

    # print("Psi: ", psi * 180/np.pi, "\tDelta: ", delta * 180/np.pi)

    return get_matrix_from_psi_delta(psi, delta)

# The Thin Film Matrix used by the 2023 paper
def thin_film_matrix_2023( wavelength , angle , n1 , k1 , n2 , k2 , d ) :
    import cmath
    N1 = complex ( n1 , k1 ) #D e f i n e Gold r e f r a c t i v e i n d e x v a r i a b l e s
    N2 = complex ( n2 , k2 ) #D e f i n e G l a s s r e f r a c t i v e i n d e x v a r i a b l e s

    beta = 2 * np.pi * ( d / wavelength ) * cmath.sqrt(N1**2 - np.sin(angle)**2)

    # define components to build the coefficient of reflection for s-polarised and p-polarised formulas
    c_t_0 = np.cos(angle)
    c_t_1 = cmath.sqrt((1-(1/N1**2))*np.sin(angle)**2)
    c_t_2 = cmath.sqrt((1-(1/N2**2))*np.sin(angle)**2)

    r01p = (N1* c_t_0 - c_t_1 ) / (N1* c_t_0 + c_t_1 )
    r01s= ( c_t_0 - N1* c_t_1 ) / ( c_t_0 + N1* c_t_1 )
    r12p = (N2* c_t_1 - N1* c_t_2 ) / (N2* c_t_1 + N1* c_t_2 )
    r12s= (N1* c_t_1 - N2* c_t_2 ) / (N1* c_t_1 + N2* c_t_2 )

    # Coeff of reflection for p and s-polarised light
    Rp = ( r01p + r12p * np.exp(-1j * 2 * beta ) ) / ( 1 + r01p * r12p* np.exp(-1j * 2 * beta ) )
    Rs=(r01s+r12s*np.exp(-1j*2*beta))/(1+r01s*r12s*np.exp(-1j*2*beta))

    # calculate psi and delta from Rs and Rp
    psi=(np.pi-np.arctan2(np.abs(Rp),np.abs(Rs)))
    delta=(np.pi-np.arctan2(np.imag(Rp/Rs),np.real(Rp/Rs)))

    # print("Psi: {:.4G}\nDelta: {:.4G}".format(psi * 180/np.pi, delta* 180/np.pi))

    # Build Jones matrix for reflected light using psi and delta
    J_g=np.array([[np.sin(psi)*np.exp(1j*delta),0.0],[0.0,np.cos(psi)]])

    return J_g

def rihan_scattering_matrix(n_au, n_quartz, d_au, d_quartz, ang):
    from scipy.linalg import expm
    
    # Simulation parameters
    lam0 = 632.8e-9  # Free space wavelength
    k0 = (2 * np.pi) / lam0
    theta = ang # Elevation angle
    psi = 0 # Azimuthal angle
    pte = 1 / np.sqrt(2)  # Amplitude of TE polarization
    ptm = 1j * pte  # Amplitude of TM polarization
    ni = 1.0  # Incident medium refractive index
    
    # External materials
    ur1, er1 = 1.0, 1.0
    ur2, er2 = 1.0, 1.0
    
    # Define layers
    N = 2
    UR = np.array([1, 1])
    ER = np.array([n_au**2, n_quartz**2])
    L = np.array([d_au, d_quartz])
    
    # Initialize matrices
    Kx = ni * np.sin(theta) * np.cos(psi)
    Ky = ni * np.sin(theta) * np.sin(psi)
    Kzh = np.sqrt(1 - Kx**2 - Ky**2)
    
    Wh = np.eye(2)
    Qh = np.array([[Kx * Ky, 1 - Kx**2],
                   [Ky**2 - 1, -Kx * Ky]])
    Omh = 1j * Kzh * np.eye(2)
    Vh = Qh @ np.linalg.inv(Omh)
    
    Sg11 = np.zeros((2, 2), dtype=complex)
    Sg12 = np.eye(2)
    Sg21 = np.eye(2)
    Sg22 = np.zeros((2, 2), dtype=complex)
    
    # Reflection side
    Krz = np.sqrt(ur1 * er1 - Kx**2 - Ky**2)
    Pr = (1 / er1) * np.array([[Kx * Ky, ur1 * er1 - Kx**2],
                                [Ky**2 - ur1 * er1, -Kx * Ky]])
    Qr = (1 / ur1) * Pr
    Omr = 1j * Krz * np.eye(2)
    Wr = np.eye(2)
    Vr = Qr @ np.linalg.inv(Omr)
    
    H = np.linalg.inv(Vh) @ Vr
    Ar = np.eye(2) + H
    Br = np.eye(2) - H
    G = np.linalg.inv(Ar) @ Br
    
    Sref11 = -np.linalg.inv(Ar) @ Br
    Sref12 = 2 * np.eye(2) @ np.linalg.inv(Ar)
    Sref21 = 0.5 * np.eye(2) @ (Ar - G @ Br)
    Sref22 = G
    
    # Update global scattering matrices
    SA11, SA12, SA21, SA22 = Sref11, Sref12, Sref21, Sref22
    SB11, SB12, SB21, SB22 = Sg11, Sg12, Sg21, Sg22

    D = np.linalg.inv(np.eye(2) - SB11 @ SA22) @ SA12
    F = np.linalg.inv(np.eye(2) - SA22 @ SB11) @ SB21
    
    Sg11 = SA11 + D @ SB11 @ SA21
    Sg12 = D @ SB12
    Sg21 = F @ SA21
    Sg22 = SB22 + F @ SA22 @ SB12
    
    for ii in range(N):
        Kz = np.sqrt(UR[ii] * ER[ii] - Kx**2 - Ky**2)
        
        Q = (1 / UR[ii]) * np.array([[Kx * Ky, UR[ii] * ER[ii] - Kx**2],
                                      [Ky**2 - UR[ii] * ER[ii], -Kx * Ky]])
        Om = 1j * Kz * np.eye(2)
        V = Q @ np.linalg.inv(Om)
        H = np.linalg.inv(V) @ Vh
        A = np.eye(2) + H
        B = np.eye(2) - H
        X = expm(Om * k0 * L[ii])
        G = np.linalg.inv(A) @ B
        M = X @ G @ X
        L_mat = A - M @ B
        
        S11 = np.linalg.inv(L_mat) @ (M @ A - B)
        S12 = np.linalg.inv(L_mat) @ X @ (A - G @ B)
        S21 = S12
        S22 = S11
        
        # Update global scattering matrices
        SA11, SA12, SA21, SA22 = Sg11, Sg12, Sg21, Sg22
        SB11, SB12, SB21, SB22 = S11, S12, S21, S22

        D = np.linalg.inv(np.eye(2) - SB11 @ SA22) @ SA12
        F = np.linalg.inv(np.eye(2) - SA22 @ SB11) @ SB21
        
        Sg11 = SA11 + D @ SB11 @ SA21
        Sg12 = D @ SB12
        Sg21 = F @ SA21
        Sg22 = SB22 + F @ SA22 @ SB12

    psi = np.arctan(np.abs(Sg11[0, 0] / Sg11[1, 1]))
    delta = np.angle(Sg11[0, 0] / Sg11[1, 1])

    # print("Rihan:\tPsi: {:.4G}\tDelta: {:.4G}".format(psi * 180/np.pi, delta *180/np.pi))
    
    return get_matrix_from_psi_delta(psi, delta)

# A helper function which can be used to switch out the sample matrix easily
def get_sample_matrix(sample_angle_of_incidence, N_air, N_gold, N_glass, d, wavelength, sample_matrix_type):
    # d = 0
    
    match sample_matrix_type:
        case 0:
            return get_snell_thin_film_matrix(sample_angle_of_incidence, N_air, N_gold, N_glass, d, wavelength)
        case 1:
            return get_fresnel_thin_film_matrix(sample_angle_of_incidence, N_air, N_gold, N_glass, d, wavelength)
        case 2:
            return thin_film_matrix_2023(wavelength, sample_angle_of_incidence, np.real(N_gold), np.imag(N_gold), np.real(N_glass), np.imag(N_glass), d)
        case 3:
            return rihan_scattering_matrix(N_gold, N_glass, d, 0.01, sample_angle_of_incidence)
        case 4:
            return get_fresnel_thin_film_hardcoded(sample_angle_of_incidence, N_air, N_gold, N_glass, d, wavelength)
        case _:
            print("Unknown matrix type used")
            return np.identity(2)
