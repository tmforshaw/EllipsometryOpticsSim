import numpy as np

# A linear polariser which is rotated about an angle theta (w.r.t horizontal)
def get_rotated_linear_polariser_matrix(theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return np.matrix([[cos_theta ** 2, cos_theta * sin_theta], [cos_theta * sin_theta, sin_theta ** 2]])

# A quarter waveplate which is rotated about an angle theta (w.r.t horizontal) [The Compensator]
def get_rotated_quarter_wave_plate(theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    combined_sin_cos = (1 - 1j) * sin_theta * cos_theta

    return np.exp(-1j * np.pi / 4) * np.matrix([[cos_theta ** 2 + 1j * sin_theta ** 2, combined_sin_cos], [combined_sin_cos, sin_theta ** 2 + 1j * cos_theta ** 2]])

# Uses a more in-depth version of Snell's law to describe how the parallel and perpendicular components of the electric field are reflected differently
# returns an array with the parallel and perpendicular components of the reflected light [0-1]
# also returns the refracted angle, at which light has passed through the boundary
def modified_snells_law(theta_incoming, n1, n2, k1, k2):
    # Create complex variables from the components of the complex refractive indices
    # N1 = np.complex64(n1, k1)
    # N2 = np.complex64(n2, k2)
    N1 = n1  - 1j * k1
    N2 = n2  - 1j * k2

    # Snell's law to find the outgoing angle
    theta_refracted = np.asin(N1 / N2 * np.sin(theta_incoming))

    # Calculate the reflected components for both the parallel and perpendicular components
    R_parallel = (np.tan(theta_incoming - theta_refracted) ** 2) / (np.tan(theta_incoming + theta_refracted) ** 2)
    R_perpendicular = (np.sin(theta_incoming - theta_refracted) ** 2) / (np.sin(theta_incoming + theta_refracted) ** 2)

    return (np.array([R_parallel, R_perpendicular]), theta_refracted)

# Use the fresnel equations to calculate the reflected components of the light
def fresnel_reflection(theta_incoming, n, k):
    N =  n - 1j * k

    A = np.sqrt(N ** 2 - np.sin(theta_incoming) ** 2)
    B = np.cos(theta_incoming)

    R_parallel = (N ** 2 * B - A) / (N ** 2 * B + A)
    R_perpendicular = (B - A) / (B + A)

    return np.array([R_parallel, R_perpendicular])

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
    # T_total = R_air_gold + T_gold_air + R_gold_glass
    T_total = R_air_gold + R_gold_glass

    ratio = T_total[0] / T_total[1]
    
    # N_gold = np.complex64(n_gold, k_gold)
    N_gold = n_gold * - 1j * k_gold

    # TODO get distance and wavelength involved in a better way
    # Psi is the angle at which the fast axis acts (the semi-major axis of the ellipse)
    psi = np.pi - np.atan2(np.abs(T_total[0]), np.abs(T_total[1]))
    # psi = np.pi - np.atan(T_total[0]/T_total[1])

    # print("Psi: {}".format(psi))

    # Delta is the difference between the phase offset given, by this traversal, to the parallel and perpendicular components
    delta = N_gold * d / wavelength * 2 * (np.pi - np.atan2(np.imag(ratio), np.real(ratio)))
    # delta = (n_gold + n_air + n_glass) * d / wavelength * 2 * (np.pi - np.atan2(np.imag(ratio), np.real(ratio)))
    # delta = (d/wavelength * 2 * np.pi) * (n_gold * np.cos(theta_refracted_air_gold) + n_air * np.cos(theta_incoming))

    # delta = (2 * np.pi / wavelength * d) * np.cos(theta_refracted_air_gold)

    # delta = 2 * np.pi * (np.modf(np.real((2 * d * n_gold) * np.cos(theta_refracted_air_gold)) / wavelength)[0] + 0.5)

    # psi = 2 * d * np.sin(theta_refracted_air_gold) * np.cos(theta_incoming)
    # delta = (2 * np.pi * d) / (wavelength * np.cos(theta_refracted_air_gold))

    # print("Psi: {:.4G}\nDelta: {:.4G}".format(psi * 180/np.pi, delta* 180/np.pi))

    #Describe this phase offset and the magnitude in a Jones' matrix
    return (np.matrix([[np.sin(psi) * np.exp(1j * delta), 0], [0, np.cos(psi)]]))

# A thin film matrix which uses the fresnel equations
def get_thin_film_matrix_2(theta_incoming, n_air, n_gold, n_glass, k_air, k_gold, k_glass, d, wavelength):
    # _, theta_refracted_air_gold = modified_snells_law(theta_incoming, n_air, n_gold, k_air, k_gold)
    R = fresnel_reflection(theta_incoming, n_gold, k_gold)

    psi = np.atan(R[0] / R[1])
    
    # # N_gold = np.complex64(n_gold, k_gold)
    N_gold = n_gold - 1j * k_gold
    
    # print(psi)

    delta = 4 * np.pi / wavelength * N_gold * d * np.cos(theta_incoming)

    # #Describe this phase offset and the magnitude in a Jones' matrix
    # return (np.matrix([[np.cos(delta), 1j * np.sin(delta) / (N_gold * np.cos(theta_refracted_air_gold))], [1j * N_gold * np.sin(delta) * np.cos(theta_refracted_air_gold), np.cos(delta)]]))
    return np.matrix([[np.sin(psi) * np.exp(1j * delta), 0], [0, np.cos(psi)]])

# A thin film which uses the fresnel equations, as given by the PyPolar documentation
def get_thin_film_matrix_3(theta_incoming, n_air, n_gold, n_glass, k_air, k_gold, k_glass, d, wavelength):
    N_gold = n_gold * (1 - 1j * k_gold)

    theta_refracted = np.acos(np.sqrt(1 - (n_air / N_gold * np.sin(theta_incoming)) ** 2))
    
    R_parallel = (N_gold * np.cos(theta_incoming) - n_air * np.cos(theta_refracted)) / (N_gold * np.cos(theta_incoming) + n_air * np.cos(theta_refracted))
    R_perpendicular = (n_air * np.cos(theta_incoming) - N_gold * np.cos(theta_refracted)) / (n_air * np.cos(theta_incoming) + N_gold * np.cos(theta_refracted))

    R = np.array([R_parallel, R_perpendicular])

    psi = np.atan(R[0] / R[1])
    
    # # N_gold = np.complex64(n_gold, k_gold)
    # N_gold = n_gold * (1 - 1j * k_gold)
    
    # print(psi)

    delta = 2 * np.pi / wavelength * N_gold * d * np.cos(theta_incoming)

    # #Describe this phase offset and the magnitude in a Jones' matrix
    # return (np.matrix([[np.cos(delta), 1j * np.sin(delta) / (N_gold * np.cos(theta_refracted_air_gold))], [1j * N_gold * np.sin(delta) * np.cos(theta_refracted_air_gold), np.cos(delta)]]))
    return (np.matrix([[np.sin(psi) * np.exp(1j * delta), 0], [0, np.cos(psi)]]))

# The Thin Film Matrix used by the 2023 paper
def jonesmodelthinfilm( wavelength , angle , n1 , k1 , n2 , k2 , d ) :
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

# A helper function which can be used to switch out the sample matrix easily
def get_sample_matrix(sample_angle_of_incidence, n_air, n_gold, n_glass, k_air, k_gold, k_glass, d, wavelength):
    # return get_thin_film_matrix(sample_angle_of_incidence, n_air, n_gold, n_glass, k_air, k_gold, k_glass, d, wavelength)
    # return get_thin_film_matrix_2(sample_angle_of_incidence, n_air, n_gold, n_glass, k_air, k_gold, k_glass, d, wavelength)
    return get_thin_film_matrix_3(sample_angle_of_incidence, n_air, n_gold, n_glass, k_air, k_gold, k_glass, d, wavelength)
    # return jonesmodelthinfilm(wavelength, sample_angle_of_incidence, n_gold, k_gold, n_glass, k_glass, d)
