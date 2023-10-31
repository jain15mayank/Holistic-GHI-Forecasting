# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 20:06:01 2023

@author: Mayank Jain
"""

import numpy as np


#***********************************************************************/
#* Name:    calcJD									*/
#* Type:    Function									*/
#* Purpose: Julian day from calendar day						*/
#* Arguments:										*/
#*   year : 4 digit year								*/
#*   month: January = 1								*/
#*   day  : 1 - 31									*/
#* Return value:										*/
#*   The Julian day corresponding to the date					*/
#* Note:											*/
#*   Number is returned for start of day.  Fractional days should be	*/
#*   added later.									*/
#***********************************************************************/
def calcJD(year, month, day):
    if (month <= 2):
        year -= 1;
        month += 12;
    A = np.floor(year/100)
    B = 2 - A + np.floor(A/4)
    JD = np.floor(365.25*(year + 4716)) + np.floor(30.6001*(month+1)) + day + B - 1524.5
    return JD

#***********************************************************************/
#* Name:    calcTimeJulianCent							*/
#* Type:    Function									*/
#* Purpose: convert Julian Day to centuries since J2000.0.			*/
#* Arguments:										*/
#*   jd : the Julian Day to convert						*/
#* Return value:										*/
#*   the T value corresponding to the Julian Day				*/
#***********************************************************************/
def calcTimeJulianCent(jd):
    T = (jd - 2451545.0)/36525.0
    return T

#***********************************************************************/
#* Name:    calGeomMeanLongSun							*/
#* Type:    Function									*/
#* Purpose: calculate the Geometric Mean Longitude of the Sun		*/
#* Arguments:										*/
#*   t : number of Julian centuries since J2000.0				*/
#* Return value:										*/
#*   the Geometric Mean Longitude of the Sun in degrees			*/
#***********************************************************************/
def calcGeomMeanLongSun(t):
    L0 = 280.46646 + (t * (36000.76983 + (0.0003032 * t)))
    while (L0 > 360.0):
        L0 -= 360.0
    while (L0 < 0.0):
        L0 += 360.0
    return L0		# in degrees

#***********************************************************************/
#* Name:    calGeomAnomalySun							*/
#* Type:    Function									*/
#* Purpose: calculate the Geometric Mean Anomaly of the Sun		*/
#* Arguments:										*/
#*   t : number of Julian centuries since J2000.0				*/
#* Return value:										*/
#*   the Geometric Mean Anomaly of the Sun in degrees			*/
#***********************************************************************/
def calcGeomMeanAnomalySun(t):
    M = 357.52911 + (t * (35999.05029 - (0.0001537 * t)))
    return M		# in degrees

#***********************************************************************/
#* Name:    calcEccentricityEarthOrbit						*/
#* Type:    Function									*/
#* Purpose: calculate the eccentricity of earth's orbit			*/
#* Arguments:										*/
#*   t : number of Julian centuries since J2000.0				*/
#* Return value:										*/
#*   the unitless eccentricity							*/
#***********************************************************************/
def calcEccentricityEarthOrbit(t):
    e = 0.016708634 - (t * (0.000042037 + (0.0000001267 * t)))
    return e		# unitless

#***********************************************************************/
#* Name:    calcSunEqOfCenter							*/
#* Type:    Function									*/
#* Purpose: calculate the equation of center for the sun			*/
#* Arguments:										*/
#*   t : number of Julian centuries since J2000.0				*/
#* Return value:										*/
#*   in degrees										*/
#***********************************************************************/
def calcSunEqOfCenter(t):
    m = calcGeomMeanAnomalySun(t)
    mrad = np.deg2rad(m)
    sinm = np.sin(mrad)
    sin2m = np.sin(mrad+mrad)
    sin3m = np.sin(mrad+mrad+mrad)
    
    C = (sinm * (1.914602 - (t * (0.004817 + (0.000014 * t))))) + (sin2m * (
        0.019993 - (0.000101 * t))) + (sin3m * 0.000289)
    return C;		# in degrees

#***********************************************************************/
#* Name:    calcSunTrueLong								*/
#* Type:    Function									*/
#* Purpose: calculate the true longitude of the sun				*/
#* Arguments:										*/
#*   t : number of Julian centuries since J2000.0				*/
#* Return value:										*/
#*   sun's true longitude in degrees						*/
#***********************************************************************/
def calcSunTrueLong(t):
    l0 = calcGeomMeanLongSun(t)
    c = calcSunEqOfCenter(t)
    O = l0 + c
    return O		# in degrees

#***********************************************************************/
#* Name:    calcSunTrueAnomaly							*/
#* Type:    Function									*/
#* Purpose: calculate the true anamoly of the sun				*/
#* Arguments:										*/
#*   t : number of Julian centuries since J2000.0				*/
#* Return value:										*/
#*   sun's true anamoly in degrees							*/
#***********************************************************************/
def calcSunTrueAnomaly(t):
    m = calcGeomMeanAnomalySun(t)
    c = calcSunEqOfCenter(t)
    v = m + c
    return v		# in degrees

#***********************************************************************/
#* Name:    calcSunRadVector								*/
#* Type:    Function									*/
#* Purpose: calculate the distance to the sun in AU				*/
#* Arguments:										*/
#*   t : number of Julian centuries since J2000.0				*/
#* Return value:										*/
#*   sun radius vector in AUs							*/
#***********************************************************************/
def calcSunRadVector(t):
    v = calcSunTrueAnomaly(t)
    e = calcEccentricityEarthOrbit(t)
    R = (1.000001018 * (1 - (e * e))) / (1 + (e * np.cos(np.deg2rad(v))))
    return R		# in AUs

#***********************************************************************/
#* Name:    calcSunApparentLong							*/
#* Type:    Function									*/
#* Purpose: calculate the apparent longitude of the sun			*/
#* Arguments:										*/
#*   t : number of Julian centuries since J2000.0				*/
#* Return value:										*/
#*   sun's apparent longitude in degrees						*/
#***********************************************************************/
def calcSunApparentLong(t):
    o = calcSunTrueLong(t)
    omega = 125.04 - (1934.136 * t)
    lambda_ = o - 0.00569 - (0.00478 * np.sin(np.deg2rad(omega)))
    return lambda_		# in degrees

#***********************************************************************/
#* Name:    calcMeanObliquityOfEcliptic						*/
#* Type:    Function									*/
#* Purpose: calculate the mean obliquity of the ecliptic			*/
#* Arguments:										*/
#*   t : number of Julian centuries since J2000.0				*/
#* Return value:										*/
#*   mean obliquity in degrees							*/
#***********************************************************************/
def calcMeanObliquityOfEcliptic(t):
    seconds = 21.448 - ( t*(46.8150 + ( t*( 0.00059 - ( t*(0.001813) )) )) )
    e0 = 23.0 + ((26.0 + (seconds/60.0))/60.0)
    return e0		# in degrees

#***********************************************************************/
#* Name:    calcObliquityCorrection						*/
#* Type:    Function									*/
#* Purpose: calculate the corrected obliquity of the ecliptic		*/
#* Arguments:										*/
#*   t : number of Julian centuries since J2000.0				*/
#* Return value:										*/
#*   corrected obliquity in degrees						*/
#***********************************************************************/
def calcObliquityCorrection(t):
    e0 = calcMeanObliquityOfEcliptic(t)
    omega = 125.04 - (1934.136 * t)
    e = e0 + (0.00256 * np.cos(np.deg2rad(omega)))
    return e		# in degrees

#***********************************************************************/
#* Name:    calcSunRtAscension							*/
#* Type:    Function									*/
#* Purpose: calculate the right ascension of the sun				*/
#* Arguments:										*/
#*   t : number of Julian centuries since J2000.0				*/
#* Return value:										*/
#*   sun's right ascension in degrees						*/
#***********************************************************************/
def calcSunRtAscension(t):
    e = calcObliquityCorrection(t)
    lambda_ = calcSunApparentLong(t)
    tananum = np.cos(np.deg2rad(e)) * np.sin(np.deg2rad(lambda_))
    tanadenom = np.cos(np.deg2rad(lambda_))
    alpha = np.rad2deg(np.arctan2(tananum, tanadenom))
    return alpha		# in degrees

#***********************************************************************/
#* Name:    calcSunDeclination							*/
#* Type:    Function									*/
#* Purpose: calculate the declination of the sun				*/
#* Arguments:										*/
#*   t : number of Julian centuries since J2000.0				*/
#* Return value:										*/
#*   sun's declination in degrees							*/
#***********************************************************************/
def calcSunDeclination(t):
    e = calcObliquityCorrection(t)
    lambda_ = calcSunApparentLong(t)
    sint = np.sin(np.deg2rad(e)) * np.sin(np.deg2rad(lambda_))
    theta = np.rad2deg(np.arcsin(sint))
    return theta		# in degrees

#***********************************************************************/
#* Name:    calcEquationOfTime							*/
#* Type:    Function									*/
#* Purpose: calculate the difference between true solar time and mean	*/
#*		solar time									*/
#* Arguments:										*/
#*   t : number of Julian centuries since J2000.0				*/
#* Return value:										*/
#*   equation of time in minutes of time						*/
#***********************************************************************/
def calcEquationOfTime(t):
    epsilon = calcObliquityCorrection(t)
    l0 = calcGeomMeanLongSun(t)
    e = calcEccentricityEarthOrbit(t)
    m = calcGeomMeanAnomalySun(t)
    
    y = np.tan(np.deg2rad(epsilon)/2.0)
    y *= y

    sin2l0 = np.sin(2.0 * np.deg2rad(l0))
    sinm   = np.sin(np.deg2rad(m))
    cos2l0 = np.cos(2.0 * np.deg2rad(l0))
    sin4l0 = np.sin(4.0 * np.deg2rad(l0))
    sin2m  = np.sin(2.0 * np.deg2rad(m))
    
    Etime = (y * sin2l0) - (2.0 * e * sinm) + (4.0 * e * y * sinm * cos2l0) - (
        0.5 * y * y * sin4l0) - (1.25 * e * e * sin2m)
    
    return np.rad2deg(Etime)*4.0	# in minutes of time

'''
zone = 0 # hrs to GMT value
daySavings = 0 # 60 if DST is True, else 0

latitude = 48.7
longitude = 2.2

year = 2019
month = 3
date = 31

ss = 0
mm = 45
hh = 5
ts = pd.Timestamp(year, month, date, hh, mm, ss)
'''

def calcSZAandSAA(TimeStamp, latitude, longitude, zone=0, daySavings=0):
    year = TimeStamp.year
    month = TimeStamp.month
    date = TimeStamp.day
    
    ss = TimeStamp.second
    mm = TimeStamp.minute
    hh = TimeStamp.hour
    
    zone += (17.5/60) # FIX to match with the actual website readings
    # timenow is GMT time for calculation
    timenow = hh + mm/60 + ss/3600 + zone - (daySavings/60)   # in hours since 0Z
    
    JD = calcJD(year, month, date)
    T = calcTimeJulianCent(JD + (timenow/24.0))
    
    eqTime = calcEquationOfTime(T)
    solarDec = calcSunDeclination(T)
    
    solarTimeFix = eqTime - (4.0 * longitude) + (60.0 * zone)
    trueSolarTime = (hh * 60.0) + mm + (ss/60.0) + solarTimeFix
    
    while (trueSolarTime > 1440):
        trueSolarTime -= 1440
    
    hourAngle = (trueSolarTime / 4.0) - 180.0
    if (hourAngle < -180):
        hourAngle += 360.0
    
    haRad = np.deg2rad(hourAngle)
    
    csz = (np.sin(np.deg2rad(latitude)) * np.sin(np.deg2rad(solarDec))) + (
        np.cos(np.deg2rad(latitude)) * np.cos(np.deg2rad(solarDec)) * np.cos(haRad))
    if (csz > 1.0):
        csz = 1.0
    elif (csz < -1.0):
        csz = -1.0
    
    zenith = np.rad2deg(np.arccos(csz))
    
    exoatmElevation = 90.0 - zenith
    if (exoatmElevation > 85.0):
        refractionCorrection = 0.0
    else:
        te = np.tan(np.deg2rad(exoatmElevation))
        if (exoatmElevation > 5.0):
            refractionCorrection = (58.1 / te) - (0.07 / (te*te*te)) + (0.000086 / (te*te*te*te*te))
        elif (exoatmElevation > -0.575):
            refractionCorrection = 1735.0 + (exoatmElevation * (-518.2 + (
                exoatmElevation * (103.4 + (exoatmElevation * (-12.79 + (exoatmElevation * 0.711)) )) )) );
        else:
            refractionCorrection = -20.774 / te;
        refractionCorrection = refractionCorrection / 3600.0;
    
    solarZen = zenith - refractionCorrection
    
    azDenom = np.cos(np.deg2rad(latitude)) * np.sin(np.deg2rad(zenith))
    if np.abs(azDenom) > 0.001:
        azRad = (( np.sin(np.deg2rad(latitude)) * np.cos(np.deg2rad(zenith)) ) - 
                 np.sin(np.deg2rad(solarDec))) / azDenom
        if np.abs(azRad) > 1:
            if azRad < 0:
                azRad = -1
            else:
                azRad = 1
        azimuth = 180 - np.rad2deg(np.arccos(azRad))
        if hourAngle > 0:
            azimuth = -azimuth
    else:
        if latitude > 0:
            azimuth = 180.0
        else:
            azimuth = 0.0
    if azimuth < 0:
        azimuth += 360
    
    return solarZen, azimuth

