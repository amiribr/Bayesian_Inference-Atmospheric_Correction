import numpy as np
from numba import njit
@njit
def reflec_both(OMEGA):

    """OMEGA  Incident angle (radians)
    X3  Reflectance sum
    X4  Reflectance difference

       n1 sin(x1) = n2 sin(x2)

                    tan(x1-x2)**2
       Refl(par ) = -------------
                    tan(x1+x2)**2

                    sin(x1-x2)**2
       Refl(perp) = -------------
                    sin(x1+x2)**2

    Where:
       x1  Incident angle
       n1  Index refraction of Air
       x2  Refracted angle
       n2  Index refraction of Water"""

#   ! Index refraction of sea water
    REF = 4. / 3.
    if OMEGA < .00001:
        RHO_PLUS = .0204078
        RHO_MINUS = 0.
    else:
        X2 = np.arcsin(np.sin(OMEGA)/REF)
        PERP = (np.sin(OMEGA-X2)/np.sin(OMEGA+X2))**2
        PAR  = (np.tan(OMEGA-X2)/np.tan(OMEGA+X2))**2
        RHO_PLUS =  PERP+PAR
        RHO_PLUS = RHO_PLUS/2.
        RHO_MINUS = -PERP+PAR
        RHO_MINUS = RHO_MINUS/2.
        return RHO_PLUS,RHO_MINUS
@njit
def ASINN(OMEGA):

    """   This calculates ASIN(X) when X is near + or - 1"""

    if OMEGA>=1.:
        asinn = np.pi/2.0
    elif OMEGA<-1.:
        asinn =-np.pi/2.0
    else:
        asinn = np.arcsin(OMEGA)
    return asinn

@njit
def ACOSS(OMEGA):

    """   This calculates ASIN(X) when X is near + or - 1"""

    if OMEGA>=1.:
        acoss = 0
    elif OMEGA<=-1.:
        acoss = np.pi
    else:
        acoss = np.arccos(OMEGA)
    return acoss
@njit
def getglint_iqu(senz, solz, ϕ, wind_spd, wind_dir):
    """
       Calculate sun gliter coefficient, including polarization
       OMEGA(real) - Angle senz       (sensor zenith angle)
       X2(real) - Angle solz      (solar zenith angle)
       X3(real) - Angle ϕ       (sensor-sun azimuth)
       X4(real) - Wind speed     (m/s)
       X5(real) - Wind direction
       X6(real) - Sun glitter coefficient
       X7(real) - Q/I for glitter
       X8(real) - U/I for glitter """
    glitter_coef, glitter_Q, glitter_U = 0, 0, 0


    Y4 = max(wind_spd, 0.001)
    Y1 = np.deg2rad(senz)
    Y1 = max(Y1,1.e-7)
    Y2 = np.deg2rad(solz)
    Y2 = max(Y2,1.e-7)
    Y3 = np.deg2rad(ϕ)

    OMEGA = ACOSS(np.cos(Y1)*np.cos(Y2)-np.sin(Y1)*np.sin(Y2)*np.cos(Y3))/2.
    OMEGA = max(OMEGA,1.e-7)
    BETA   = ACOSS((np.cos(Y1)+np.cos(Y2))/(2.*np.cos(OMEGA)))
    BETA = max(BETA,1.e-7)
    ALPHA  = ACOSS((np.cos(BETA)*np.cos(Y2)-np.cos(OMEGA))/(np.sin(BETA)*np.sin(Y2)))
    if np.sin(Y3)<0:
        ALPHA = -ALPHA
#   from Cox & Munk
    SIGC   = .04964*np.sqrt(Y4)
    SIGU   = .04964*np.sqrt(Y4)

    CHI    = wind_dir
    ALPHAP = ALPHA+CHI
    SWIG   = np.sin(ALPHAP)*np.tan(BETA)/SIGC
    ETA    = np.cos(ALPHAP)*np.tan(BETA)/SIGU
    EXPON  = -(SWIG**2+ETA**2)/2.
    #EXPON = max(EXPON, -30.) # trap underflow
    #EXPON = min(EXPON, 30.) # trap overflow
    if EXPON<-30:
        EXPON = -30.         # trap underflow
    if EXPON>30.:
        EXPON = 30.         # trap overflow
    PROB   = np.exp(EXPON)/(2.*np.pi*SIGU*SIGC)

    RHO_PLUS,RHO_MINUS = reflec_both(OMEGA)

# Normal distribution
    glitter_coef  = RHO_PLUS*PROB/(4.*np.cos(Y1)*np.cos(BETA)**4)

    if OMEGA > .0001:
        CR = (np.cos(Y2) - np.cos(2.*OMEGA)*np.cos(Y1))/(np.sin(2.*OMEGA)*np.sin(Y2))
        SR = np.sin(Y2)*np.sin(np.pi-Y3) / np.sin(2.*OMEGA)
        ROT_ANG = np.sign(CR)*ASINN(SR)
    else:
        ROT_ANG = np.pi/2.

    C2R = np.cos(2.*ROT_ANG)
    S2R = np.sin(2.*ROT_ANG)

    glitter_Q  =  C2R * RHO_MINUS / RHO_PLUS
    glitter_U = -S2R * RHO_MINUS / RHO_PLUS



    return glitter_coef, glitter_Q, glitter_U

@njit
def getglint_iqu_hg(senz, solz, ϕ, wind_spd, wind_dir):
    """
       Calculate sun gliter coefficient, including polarization
       OMEGA(real) - Angle senz       (sensor zenith angle)
       X2(real) - Angle solz      (solar zenith angle)
       X3(real) - Angle ϕ       (sensor-sun azimuth)
       X4(real) - Wind speed     (m/s)
       X5(real) - Wind direction
       X6(real) - Sun glitter coefficient
       X7(real) - Q/I for glitter
       X8(real) - U/I for glitter """
    glitter_coef, glitter_Q, glitter_U = 0, 0, 0
    Y4 = max(wind_spd, 0.001)
    Y1 = np.deg2rad(senz)
    Y1 = max(Y1, 1.e-7)
    Y2 = np.deg2rad(solz)
    Y2 = max(Y2, 1.e-7)
    Y3 = np.deg2rad(ϕ)

    OMEGA = ACOSS(np.cos(Y1)*np.cos(Y2)-np.sin(Y1)*np.sin(Y2)*np.cos(Y3))/2.
    OMEGA = max(OMEGA, 1.e-7)
    BETA = ACOSS((np.cos(Y1)+np.cos(Y2))/(2.*np.cos(OMEGA)))
    BETA = max(BETA, 1.e-7)
    ALPHA = ACOSS((np.cos(BETA)*np.cos(Y2)-np.cos(OMEGA))/(np.sin(BETA)*np.sin(Y2)))
    if np.sin(Y3) < 0:
        ALPHA = -ALPHA
#   from Cox & Munk
    #SIGC = .04964*np.sqrt(Y4)
    #SIGU = .04964*np.sqrt(Y4)
    SIGC = .07307*np.sqrt(Y4)
    SIGU = .07307*np.sqrt(Y4)

    CHI = wind_dir
    ALPHAP = ALPHA+CHI
    SWIG = np.sin(ALPHAP)*np.tan(BETA)/SIGC
    ETA = np.cos(ALPHAP)*np.tan(BETA)/SIGU
    EXPON = -(SWIG**2+ETA**2)/2.
    if EXPON < -30:
        EXPON = -30.         # trap underflow
    if EXPON > 30.:
        EXPON = 30.         # trap overflow
    PROB = np.exp(EXPON)/(2.*np.pi*SIGU*SIGC)

    RHO_PLUS, RHO_MINUS = reflec_both(OMEGA)

# Normal distribution
    glitter_coef = RHO_PLUS*PROB/(4.*np.cos(Y1)*np.cos(BETA)**4)

    if OMEGA > .0001:
        CR = (np.cos(Y2) - np.cos(2.*OMEGA)*np.cos(Y1))/(np.sin(2.*OMEGA)*np.sin(Y2))
        SR = np.sin(Y2)*np.sin(np.pi-Y3) / np.sin(2.*OMEGA)
        ROT_ANG = np.sign(CR)*ASINN(SR)
    else:
        ROT_ANG = np.pi/2.

    C2R = np.cos(2.*ROT_ANG)
    S2R = np.sin(2.*ROT_ANG)

    glitter_Q = C2R * RHO_MINUS / RHO_PLUS
    glitter_U = -S2R * RHO_MINUS / RHO_PLUS

    return glitter_coef, glitter_Q, glitter_U

@njit
def getglint_iqu_cm(senz, solz, ϕ, wind_spd, wind_dir):
    """
       Calculate sun gliter coefficient, including polarization
       OMEGA(real) - Angle senz       (sensor zenith angle)
       X2(real) - Angle solz      (solar zenith angle)
       X3(real) - Angle ϕ       (sensor-sun azimuth)
       X4(real) - Wind speed     (m/s)
       X5(real) - Wind direction
       X6(real) - Sun glitter coefficient
       X7(real) - Q/I for glitter
       X8(real) - U/I for glitter """
    glitter_coef, glitter_Q, glitter_U = 0, 0, 0
    Y4 = max(wind_spd, 0.001)
    Y1 = np.deg2rad(senz)
    Y1 = max(Y1, 1.e-7)
    Y2 = np.deg2rad(solz)
    Y2 = max(Y2, 1.e-7)
    Y3 = np.deg2rad(ϕ)

    OMEGA = ACOSS(np.cos(Y1)*np.cos(Y2)-np.sin(Y1)*np.sin(Y2)*np.cos(Y3))/2.
    OMEGA = max(OMEGA, 1.e-7)
    BETA = ACOSS((np.cos(Y1)+np.cos(Y2))/(2.*np.cos(OMEGA)))
    BETA = max(BETA, 1.e-7)
    ALPHA = ACOSS((np.cos(BETA)*np.cos(Y2)-np.cos(OMEGA))/(np.sin(BETA)*np.sin(Y2)))
    if np.sin(Y3) < 0:
        ALPHA = -ALPHA
#   from Cox & Munk
    #SIGC = .04964*np.sqrt(Y4)
    #SIGU = .04964*np.sqrt(Y4)
    SIGC = 0.0437 + 0.07155*np.sqrt(Y4)
    SIGU = 0.0437 + 0.07155*np.sqrt(Y4)

    CHI = wind_dir
    ALPHAP = ALPHA+CHI
    SWIG = np.sin(ALPHAP)*np.tan(BETA)/SIGC
    ETA = np.cos(ALPHAP)*np.tan(BETA)/SIGU
    EXPON = -(SWIG**2+ETA**2)/2.
    if EXPON < -30:
        EXPON = -30.         # trap underflow
    if EXPON > 30.:
        EXPON = 30.         # trap overflow
    PROB = np.exp(EXPON)/(2.*np.pi*SIGU*SIGC)

    RHO_PLUS, RHO_MINUS = reflec_both(OMEGA)

# Normal distribution
    glitter_coef = RHO_PLUS*PROB/(4.*np.cos(Y1)*np.cos(BETA)**4)

    if OMEGA > .0001:
        CR = (np.cos(Y2) - np.cos(2.*OMEGA)*np.cos(Y1))/(np.sin(2.*OMEGA)*np.sin(Y2))
        SR = np.sin(Y2)*np.sin(np.pi-Y3) / np.sin(2.*OMEGA)
        ROT_ANG = np.sign(CR)*ASINN(SR)
    else:
        ROT_ANG = np.pi/2.

    C2R = np.cos(2.*ROT_ANG)
    S2R = np.sin(2.*ROT_ANG)

    glitter_Q = C2R * RHO_MINUS / RHO_PLUS
    glitter_U = -S2R * RHO_MINUS / RHO_PLUS

    return glitter_coef, glitter_Q, glitter_U
