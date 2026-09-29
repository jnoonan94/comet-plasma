#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 18:04:33 2020

@author: johnnoonan
"""

#import pandas as pd
#from astropy.io import fits
#import os
import numpy as np
#import math
#from scipy import signal
import scipy.special as special
#from scipy.signal import medfilt
#from scipy.optimize import curve_fit
from scipy import integrate
#from lmfit import Model
#from lmfit.models import DonaichModel,SkewedGaussianModel,GaussianModel, PolynomialModel
#import datetime
#import time
import matplotlib.pyplot as plt
from matplotlib import cm
#from scipy.interpolate import UnivariateSpline as unv_sp
#from pylab import rcParams
#import mpmath as mp
import matplotlib
#import decimal
#import g_factor_oi_triplet
#import g_factor_co_4pg
'''All constants for emission calculations are stored here'''
t_H2O = 8.2E4#lifetime of parent molecule in s, Combi and Delsemme 1980
t_H = 1.5E6#lifetime of daughter atom (H) in s
t_H2O_d = 1/(1.03e-5) #H2O dissociation lifetime, from Huebner et al. 1992
t_OH = 5e5#1/(6.54e-6) # lifetime for OH from Huebner et al. 1992
t_CO = 1/(8.0e-5) # lifetime of CO from Huebner et al. 2015
t_C = 2.5E5 #Opal and Carruthers 1977
t_O = 1.7E6 #Opal and Carruthers 1977
c_light = 3E8 #m/s
m_e = 9.11E-31 #m/s
epsilon = 8.854E-12 #farads/meter
q_e = -1.6E-19 #C
v_H = 12000 #m/s
v_C = 1000  #m/s
v_O = 1330  #m/s
v_OH = 1.15#1330 #m/s, Fink and Combi 2004
convert_to_nm = 1E9
planck = 6.6261E-34 #Js
metersperAU = 1.496e11


def au_to_m(d_au):
    return d_au*1.496e11
def mean_binning(x,col,binwidth):
    """Sums flux over a specified number of bins.
    x: wavelength array
    spec: column density array
    error: error array
    binwidth: size of bin, should be an interger.
    """
    # convenience function to bin 1D spectra and return the correct pixel coordinates for the centers
    nbins = int(len(col)/binwidth)
    bin_spec = np.array([])
    bin_centers = np.array([])
    for i in range(nbins):
        tmp_wave = x[i*binwidth:(i+1)*binwidth]
        tmp_spec = col[i*binwidth:(i+1)*binwidth]
        full_average = np.average(tmp_spec)
        # trim is a boolean mask for clipping outliers out of individual bins.
        # can adjust the sigma threshold by changing the 2, or just skip this step
        # by changing which trim_average line to comment out
        
        trim = np.logical_and(True, np.abs((tmp_spec-full_average) < 7 ))
        #if (len(tmp_wave) - np.count_nonzero(trim)>0):
        #    print(len(tmp_wave) - np.count_nonzero(trim))
        #trim_average = np.average(tmp_spec[trim],weights=tmp_weights[trim],returned=True)
        trim_average = np.average(tmp_spec,returned=True)
        bin_centers = np.append(bin_centers,np.mean(tmp_wave[trim]))
        bin_spec = np.append(bin_spec,trim_average[0])
    delta_x = x[1]-x[0]
    #bin_centers = x[0]+delta_x*bin_centers
    return bin_centers,bin_spec
def two_component_spatial_profile(molecule,radius,delta,delta_dot,r_h,r_dot,Q):
    '''This can be used for the following parent-daughter relationships:
        Parent | Daughter 
        H2O    | OH
        H2O    | H
        CS2    | S 
        CO2    | O
        S2     | S
        CO     | C
        CO     | O
        C3     | C2
        HCN    | CN
    This code can calculate the expected column density for any of these combinations provided the following
    input:
        molecule: a string ,'h2o', 'cs2', 'co2','s2,'co','c3','hcn'
        radius: a float, maximum radius from comet nucleus, in m
        size: a float if FOV is circular, this is the radius in arcseconds. A tuple if the FOV is a rectangle and this is the [height, width] in arcseconds
        delta: a float, distance from the observing point to the source in astronomical units
        r_h: a float, the heliocentric distance to the comet in astronomical units
    In the future an offset capability will be added for modeling non-centered observations.
    
    The output from this code produces a value with units of time in seconds, which when multiplied by a production rate then provides the total number of molecules in the FOV. '''
    radius_array = np.logspace(4,np.log10(radius),1000)
    col_density_array = []
    for radius in radius_array:
        col = LOS_integration(radius,r_h,2,molecule)
        col_density_array.append(Q*col/1e4)
    return radius_array,col_density_array
def three_component_spatial_profile(molecule,radius,delta,delta_dot,r_h,r_dot,Q):
    '''This can be used for the following parent-daughter relationships:
        Parent | Daughter 
        H2O    | OH
        H2O    | H
        CS2    | S 
        CO2    | O
        S2     | S
        CO     | C
        CO     | O
        C3     | C2
        HCN    | CN
    This code can calculate the expected column density for any of these combinations provided the following
    input:
        molecule: a string ,'h2o', 'cs2', 'co2','s2,'co','c3','hcn'
        radius: a float, maximum radius from comet nucleus, in cm
        size: a float if FOV is circular, this is the radius in arcseconds. A tuple if the FOV is a rectangle and this is the [height, width] in arcseconds
        delta: a float, distance from the observing point to the source in astronomical units
        r_h: a float, the heliocentric distance to the comet in astronomical units
    In the future an offset capability will be added for modeling non-centered observations.
    
    The output from this code produces a value with units of time in seconds, which when multiplied by a production rate then provides the total number of molecules in the FOV. '''
    radius_array = radius_array = np.linspace(100,radius,1000)
    col_density_array = []
    for radius in radius_array:
        col = LOS_integration(radius,r_h,3,molecule)
        col_density_array.append(Q*col/1e4)
    return radius_array,col_density_array        
def one_component(molecule,FOV,size,delta,delta_dot,r_h,r_dot,offset = False):
    '''This can be used for the following parent-daughter relationships:
        Parent | Daughter 
        H2O    | OH
        H2O    | H
        CS2    | S 
        CS2     | CS
        OCS     | CS
        H2CS    | CS
        CO2    | O
        S2     | S
        CO     | C
        CO     | O
        C3     | C2
        HCN    | CN
    This code can calculate the expected column density for any of these combinations provided the following
    input:
        molecule: a string ,'h2o', 'cs2', 'co2','s2,'co','c3','hcn'
        FOV: a string, shape of FOV 'circle', or 'rectangle'
        size: a float if FOV is circular, this is the radius in arcseconds. A tuple if the FOV is a rectangle and this is the [height, width] in arcseconds
        delta: a float, distance from the observing point to the source in astronomical units
        r_h: a float, the heliocentric distance to the comet in astronomical units
    In the future an offset capability will be added for modeling non-centered observations.
    
    The output from this code produces a value with units of time in seconds, which when multiplied by a production rate then provides the total number of molecules in the FOV. '''
    
    if FOV == 'circle':
        r_aperture = size
        r_fov_m = r_aperture / 206265.0 * au_to_m(delta)
        result,error = circular_aperture_integration(r_fov_m,r_h,molecule,1)
    if FOV == 'rectangle':
        box_fov_m = [size[0]  / 206265.0 * au_to_m(delta),size[1] / 206265.0 * au_to_m(delta)]
        result,error = rectangular_aperture_integration(box_fov_m,r_h,molecule,1)
 
    return result,error
def two_component(molecule,FOV,size,delta,delta_dot,r_h,r_do_ht,offset = False):
    '''This can be used for the following parent-daughter relationships:
        Parent | Daughter 
        H2O    | OH
        H2O    | H
        CS2    | S 
        CS2     | CS
        OCS     | CS
        H2CS    | CS
        CO2    | O
        S2     | S
        CO     | C
        CO     | O
        C3     | C2
        HCN    | CN
    This code can calculate the expected column density for any of these combinations provided the following
    input:
        molecule: a string ,'h2o', 'cs2', 'co2','s2,'co','c3','hcn'
        FOV: a string, shape of FOV 'circle', or 'rectangle'
        size: a float if FOV is circular, this is the radius in arcseconds. A tuple if the FOV is a rectangle and this is the [height, width] in arcseconds
        delta: a float, distance from the observing point to the source in astronomical units
        r_h: a float, the heliocentric distance to the comet in astronomical units
    In the future an offset capability will be added for modeling non-centered observations.
    
    The output from this code produces a value with units of time in seconds, which when multiplied by a production rate then provides the total number of molecules in the FOV. '''
    
    if FOV == 'circle':
        r_aperture = size
        r_fov_m = r_aperture / 206265.0 * au_to_m(delta)
        result,error = circular_aperture_integration(r_fov_m,r_h,molecule,2)
    if FOV == 'rectangle':
        box_fov_m = [size[0]  / 206265.0 * au_to_m(delta),size[1] / 206265.0 * au_to_m(delta)]
        result,error = rectangular_aperture_integration(box_fov_m,r_h,molecule,2)
 
    return result,error
def three_component(molecule,FOV,size,delta,delta_dot,r_h,r_dot,offset = False,odell = False):
    '''This can be used for the following parent-daughter relationships:
        Parent | Daughter | Granddaughter
        H2O    | OH       | O
        H2O    | OH       | H
        H2S    | HS       | S
        CO2    | CO       | O 
        
    This code can calculate the expected column density for any of these combinations provided the following
    input:
        molecule: a string ,'h2o', 'h2s', 'co2'
        FOV: a string, shape of FOV 'circle', or 'rectangle'
        size: a float if FOV is circular, this is the radius in arcseconds. A tuple if the FOV is a rectangle and this is the [height, width] in arcseconds
        delta: a float, distance from the observing point to the source in astronomical units
        delta_dot: a float, geocentric distance in km/s
        r_h: a float, the heliocentric distance to the comet in astronomical units
        r_dot: a float, the heliocentric velocity of the comet in km/s
        odell: This Boolean determines whether or not to use the O'Dell shortcut, which calculates the column density for a circular FOV centered on the nucleus when multiplied by a production rate.
    The output from this code, when odell = False, produces a value with units of time in seconds, which when multiplied by a production rate then provides the total number of molecules in the FOV. '''
    if FOV == 'circle':
        if odell == True:
            r_aperture = size
            r_fov_m = r_aperture / 206265.0 * au_to_m(delta)
            result,error = haser_density_three_comp_odell(r_fov_m,r_h,molecule,circle = True)
        else:
            r_aperture = size
            r_fov_m = r_aperture / 206265.0 * au_to_m(delta)
            result,error = circular_aperture_integration(r_fov_m,r_h,molecule,3)
    if FOV == 'rectangle':
        print('NOT YET AVAILABLE')
        box_fov_m = [size[0]/ 206265.0 * au_to_m(delta),size[1]/ 206265.0 * au_to_m(delta)]
        result,error = rectangular_aperture_integration(box_fov_m,r_h,molecule,3)
 
    return result,error

'''This section covers the functions that perform the integrations for both rectangular and ciruclar apertures. Still need to add in capability for offset pointings with correct algebra.'''
def circular_aperture_integration(r_aperture,r_h,molecule,num_comp):
    result,err = integrate.quad(LOS_circular_integration,10,r_aperture, args=(r_h,num_comp,molecule),limit = 5000)
    return result,err
def LOS_integration(r_p,r_h,num_comp,molecule):
    if num_comp == 2:
        result,err = integrate.quad(haser_density_two_comp,0,1E9,args=(r_p,r_h,molecule))#,epsabs=1.49e-08, epsrel=1.49e-08)
    elif num_comp == 3:
        result,err = integrate.quad(haser_density_three_comp,0,1E9,args=(r_p,r_h,molecule))#,epsabs=1.49e-08, epsrel=1.49e-08)
    return 2*result
def LOS_circular_integration(r_p,r_h,num_comp,molecule):
    circle = True
    if num_comp == 1:
        result, err = integrate.quad(haser_density_one_comp,10,float(1E9),args=(r_p,r_h,molecule,circle))
    if num_comp == 2:
        result,err = integrate.quad(haser_density_two_comp,0,1E9,args=(r_p,r_h,molecule,circle))#,epsabs=1.49e-08, epsrel=1.49e-08)
    elif num_comp == 3:
        result,err = integrate.quad(haser_density_three_comp,0,1E9,args=(r_p,r_h,molecule,circle))#,epsabs=1.49e-08, epsrel=1.49e-08)
    return 2*result
def rectangular_aperture_integration(box,r_h,molecule,num_comp):
    box_width = box[0] / 2.0
    box_length = box[1] / 2.0 # .....
    result, err = integrate.dblquad(LOS_rectangular_integration, box_width/1000, box_width, box_length/1000, box_length,args=(r_h,num_comp,molecule))
    return result * 4.0 , err
def LOS_rectangular_integration(x,y,r_h,num_comp,molecule):
    r_p = np.sqrt(x**2.0 + y**2.0) #impact parameter in m
    if num_comp == 1:
        result, err = integrate.quad(haser_density_one_comp,10,float(1E9),args=(r_p,r_h,molecule))
    if num_comp == 2:
        result, err = integrate.quad(haser_density_two_comp,10,float(1E9),args=(r_p,r_h,molecule))
    elif num_comp == 3:
        result, err = integrate.quad(haser_density_three_comp,10,float(1E9),args=(r_p,r_h,molecule))
    return 2.0 * result

'''This section covers the generic functions that are to be used for the two-component model, with both circular
and rectangular slits. Note that these functions do not take the production rate into account, they are intended to be used to solve for the production rate.'''

def haser_density_one_comp(l,r_p,r_hel,molecule,circle = False):
    '''
          _ _ _ l _ _ 
          |         /
    r_p   |        /         
          |       /
          |      /
          |     / r (calculated in each step based on supplied values of l and r_p)
          |    /
          |   /
          |  /
          | /
    comet *
    
    l, a float, is the distance away from a line projected straight from the nucleus that intersects with the line of sight.
    r_p, a float, is the impact parameter from the nucleus along the line of sight as seen from the observer.
    r_hel, a float, is the heliocentric distance in astronomical units.
    molecule, a string, is the name of the parent molecule being modeled'''
    if molecule == 'h2o':
        t_p = t_H2O*r_hel**2
        t_d_oh = t_OH*r_hel**2
        t_d = t_OH*r_hel**2
        v_p = outflow_vel_h2o(r_hel)
        v_d = outflow_vel_oh(r_hel)
    elif molecule == 'cs':
        #_p = 9.1e5*(r_hel)**2 #from Meier and A'Hearn 1997 for S, 
        t_p = 8.56e4*(r_hel)**2 #from Meier and A'Hearn 1997 for CS, also Stern et al. 1998
        #t_p = 340*(r_hel)**2 #1000 s from Feldman et al. 1999, 340 from Stern et al. 1998 and Huebner and Mukherjee 2015, 590 from Meier and A'Hearn 1997, Jackson, Butterworth, and Ballard 1986 (http://articles.adsabs.harvard.edu//full/1986ApJ...304..515J/0000516.000.html)
        v_p = outflow_vel_h2o(r_hel)
        #v_d = outflow_vel_cs(r_hel)  #m/s
        v_d = outflow_vel_oh(r_hel)
    elif molecule == 'ocs':
        #t_d = 9.1e5*(r_hel)**2 #from Meier and A'Hearn 1997 for S, 
        t_d = 1e5*(r_hel)**2 #from Meier and A'Hearn 1997 for CS, also Stern et al. 1998
        t_p = 1.2e4*(r_hel)**2 #Azoulay and Festou 1986 say total lifetime is 12,000, I get 1.56e5 s from Heubner and Mukherjee 2015
        v_p = outflow_vel_cs2(r_hel)
        v_d = outflow_vel_cs(r_hel)  #m/s
    elif molecule == 'h2cs':
        #t_d = 9.1e5*(r_hel)**2 #from Meier and A'Hearn 1997 for S, 
        t_d = 1e5*(r_hel)**2 #from Meier and A'Hearn 1997 for CS, also Stern et al. 1998
        t_p = 2.3e4*(r_hel)**2 #heubner and mukherjee 2015 for h2co, multiplied by 10 to account for weight of CS as suggested by
        v_p = outflow_vel_cs2(r_hel)
        v_d = outflow_vel_cs(r_hel)  #m/s
    elif molecule == 'co2':
        '''NOT YET READY NEED TO GET VALUES'''
        t_d = 9.1e5*(r_hel)**2 #from Meier and A'Hearn 1997 
        t_p = 1/(1.66e-7+1.99e-8+6.35e-7)*(r_hel)**2 #from Meier and A'Hearn 1997, Jackson, Butterworth, and Ballard 1986 (http://articles.adsabs.harvard.edu//full/1986ApJ...304..515J/0000516.000.html)
        v_p = outflow_vel_cs2(r_hel)
        v_d = outflow_vel_oh(r_hel)  #m/s
    elif molecule == 's2':
        t_d = 9.1e5*(r_hel)**2 #from Meier and A'Hearn 1997 
        t_p = 450*(r_hel)**2 #from A'Hearn, Feldman, and Schliecher 1983 (http://articles.adsabs.harvard.edu//full/1983ApJ...274L..99A/L000101.000.html)
        v_p = outflow_vel_cs2(r_hel)
        v_d = outflow_vel_oh(r_hel)  #m/s
    elif molecule == 'c2':
        t_p = 6.6 * 10.0**(4.0) * (r_hel**2.0) #s
        #t_p = 2.558e4 #data from scheicher 2010, a'hearn et al. 1995
        #v_p = outflow_vel_cs2(r_hel)
        v_p = outflow_vel_c2(r_hel)  #m/s    
    elif molecule == 'hcn':
        t_d = 2.1 * 10.0**(5.0) * (r_hel**2.0) #from Meier and A'Hearn 1997 
        t_p = 1.5924e4*(r_hel)**2 #data from scheicher 2010, a'hearn et al. 1995
        v_p = outflow_vel_cs2(r_hel)
        v_d = outflow_vel_oh(r_hel)  #m/s
    elif molecule == 'co':
        '''NOT YET READY NEED TO GET VALUES'''
        t_d = 9.1e5*(r_hel)**2 #SOURCE
        t_p = 590*(r_hel)**2 #SOURCE
        v_p = outflow_vel_oh(r_hel)
        v_d = outflow_vel_oh(r_hel)  #m/s
    elif molecule == 'h2s':
        t_d = t_H*(r_hel)**2 #from Meier and A'Hearn 1997 , Kim and A'Hearn 1992, but note it is for -28.5 km/s and is likely in need of update()
        t_p_d = (1/3.2e-4)*r_hel**2 #Double check Huebner et al. 1992
        t_p = 4150*(r_hel)**2 #from Meier and A'Hearn 1997, Eberhardt et al. 1994
        v_p = outflow_vel_h2o(r_hel)
        v_d = v_H  #m/s
    else:
        print('This molecule not yet available, please see documentation')
    r = np.sqrt(l**2 + r_p**2)
    b_p = (t_p*v_p)# scale length of parent molecule in m
    first = 1/(4*np.pi*r**2*v_p)
    second = np.exp(-r/b_p)
    mol_den = first*second
    if circle == True:
        mol_den = 2*np.pi*r_p*mol_den
    return mol_den
def haser_density_two_comp(l,r_p,r_hel,molecule,circle = False):
    '''
          _ _ _ l _ _ 
          |         /
    r_p   |        /         
          |       /
          |      /
          |     / r (calculated in each step based on supplied values of l and r_p)
          |    /
          |   /
          |  /
          | /
    comet *
    
    l, a float, is the distance away from a line projected straight from the nucleus that intersects with the line of sight.
    r_p, a float, is the impact parameter from the nucleus along the line of sight as seen from the observer.
    r_hel, a float, is the heliocentric distance in astronomical units.
    molecule, a string, is the name of the parent molecule being modeled'''
    if molecule == 'h2o':
        t_p = t_H2O*r_hel**2
        t_d_oh = t_OH*r_hel**2
        t_d = t_OH*r_hel**2
        v_p = outflow_vel_h2o(r_hel)
        v_d = outflow_vel_oh(r_hel)
    elif molecule == 'cs2':
        #t_d = 9.1e5*(r_hel)**2 #from Meier and A'Hearn 1997 for S, 
        t_d = 7.3e5*(r_hel)**2 #from Meier and A'Hearn 1997 for CS, also Stern et al. 1998
        t_p = 340*(r_hel)**2 #1000 s from Feldman et al. 1999, 340 from Stern et al. 1998 and Huebner and Mukherjee 2015, 590 from Meier and A'Hearn 1997, Jackson, Butterworth, and Ballard 1986 (http://articles.adsabs.harvard.edu//full/1986ApJ...304..515J/0000516.000.html)
        v_p = outflow_vel_h2o(r_hel)
        #v_d = outflow_vel_cs(r_hel)  #m/s
        #v_d = outflow_vel_oh(r_hel)
        v_d = v_p+10 #m/s
    elif molecule == 'ocs':
        #t_d = 9.1e5*(r_hel)**2 #from Meier and A'Hearn 1997 for S, 
        t_d = 7.3e5*(r_hel)**2 #from Meier and A'Hearn 1997 for CS, also Stern et al. 1998
        t_p = 1.2e4*(r_hel)**2 #Azoulay and Festou 1986 say total lifetime is 12,000, I get 1.56e5 s from Heubner and Mukherjee 2015
        v_p = outflow_vel_cs2(r_hel)
        v_d = outflow_vel_cs(r_hel)  #m/s
    elif molecule == 'h2cs':
        #t_d = 9.1e5*(r_hel)**2 #from Meier and A'Hearn 1997 for S, 
        t_d = 1e5*(r_hel)**2 #from Meier and A'Hearn 1997 for CS, also Stern et al. 1998
        t_p = 2.3e4*(r_hel)**2 #heubner and mukherjee 2015 for h2co, multiplied by 10 to account for weight of CS as suggested by
        v_p = outflow_vel_cs2(r_hel)
        v_d = outflow_vel_cs(r_hel)  #m/s
    elif molecule == 'co2':
        '''NOT YET READY NEED TO GET VALUES'''
        t_d = 9.1e5*(r_hel)**2 #from Meier and A'Hearn 1997 
        t_p = 1/(1.66e-7+1.99e-8+6.35e-7)*(r_hel)**2 #from Meier and A'Hearn 1997, Jackson, Butterworth, and Ballard 1986 (http://articles.adsabs.harvard.edu//full/1986ApJ...304..515J/0000516.000.html)
        v_p = outflow_vel_cs2(r_hel)
        v_d = outflow_vel_oh(r_hel)  #m/s
    elif molecule == 's2':
        t_d = 9.1e5*(r_hel)**2 #from Meier and A'Hearn 1997 
        t_p = 450*(r_hel)**2 #from A'Hearn, Feldman, and Schliecher 1983 (http://articles.adsabs.harvard.edu//full/1983ApJ...274L..99A/L000101.000.html)
        v_p = outflow_vel_cs2(r_hel)
        v_d = outflow_vel_oh(r_hel)  #m/s
    elif molecule == 'c3':
        t_d = 6.6 * 10.0**(4.0) * (r_hel**2.0) #c2
        t_p = 2.558e4 #data from scheicher 2010, a'hearn et al. 1995
        v_p = outflow_vel_h2o(r_hel)
        v_d = outflow_vel_c2(r_hel)  #m/s    
    elif molecule == 'hcn':
        t_d = 2.1 * 10.0**(5.0) * (r_hel**2.0) #from Meier and A'Hearn 1997 
        t_p = 1.5924e4*(r_hel)**2 #data from scheicher 2010, a'hearn et al. 1995
        v_p = outflow_vel_cs2(r_hel)
        v_d = outflow_vel_oh(r_hel)  #m/s
    elif molecule == 'co':
        '''NOT YET READY NEED TO GET VALUES'''
        t_d = 9.1e5*(r_hel)**2 #SOURCE
        t_p = 590*(r_hel)**2 #SOURCE
        v_p = outflow_vel_oh(r_hel)
        v_d = outflow_vel_oh(r_hel)  #m/s
    elif molecule == 'h2s':
        t_d = 7.3e5*(r_hel)**2 #from Meier and A'Hearn 1997 , Kim and A'Hearn 1992, but note it is for -28.5 km/s and is likely in need of update()
        t_p_d = (1/3.2e-4)*r_hel**2 #Double check Huebner et al. 1992
        t_p = 3125*(r_hel)**2 #from Meier and A'Hearn 1997, Eberhardt et al. 1994
        v_p = outflow_vel_h2o(r_hel)
        v_d = v_H  #m/s
    else:
        print('This molecule not yet available, please see documentation')
    r = np.sqrt(l**2 + r_p**2)
    b_p = (t_p*v_p)# scale length of parent molecule in m
    b_d = (t_d*v_d)# scale length of daughter atom in m
    first = 1/(4*np.pi*r**2*v_p)
    scale_length_ratio = (b_d/(b_p-b_d))
    second = (np.exp(-r/b_p)-np.exp(-r/b_d))
    mol_den = first*second*scale_length_ratio
    if circle == True:
        mol_den = 2*np.pi*r_p*mol_den
    return mol_den
    
'''This section covers the generic functions that are to be used for the three-component model, with both circular
and rectangular slits. Note that these functions do not take the production rate into account, they are intended to be used to solve for the production rate.'''
def haser_density_three_comp(l,r_p,r_hel,molecule,circle = False):
    '''
          _ _ _ l _ _ 
          |         /
    r_p   |        /         
          |       /
          |      /
          |     / r (calculated in each step based on supplied values of l and r_p)
          |    /
          |   /
          |  /
          | /
    comet *
    
    l, a float, is the distance away from a line projected straight from the nucleus that intersects with the line of sight.
    r_p, a float, is the impact parameter from the nucleus along the line of sight as seen from the observer.
    r_hel, a float, is the heliocentric distance in astronomical units.
    molecule, a string, is the name of the parent molecule being modeled'''
    if molecule == 'h2o':
        t_p = t_H2O*r_hel**2
        t_p_d = t_H2O_d*r_hel**2
        t_d = t_OH*r_hel**2
        t_gd = t_O*r_hel**2
        v_p = outflow_vel_h2o(r_hel)
        v_d = outflow_vel_oh(r_hel)
        v_gd = v_O #note that there should be a two-velocity component here that is not yet adequately implemented in this model. 
    elif molecule == 'cs2':
        t_d = 1e5*(r_hel)**2 #from Meier and A'Hearn 1997, Stern et al. 1998
        t_p_d = 590*(r_hel)**2 #from Meier and A'Hearn 1997, Jackson, Butterworth, and Ballard 1986 (http://articles.adsabs.harvard.edu//full/1986ApJ...304..515J/0000516.000.html)
        t_p = 1/(2.03e-3+8.92e-4)*r_hel**2 #From Huebner et al 1992
        t_gd = 9.1e5*(r_hel)**2 #from Huebner et al. 1992, at solar min. Decreases to 4.2e5 at solar max.
        v_p = outflow_vel_cs2(r_hel)
        v_d = outflow_vel_oh(r_hel)  #m/s
        v_gd = outflow_vel_oh(r_hel)
    elif molecule == 'co2':
        t_d = 9.1e5*(r_hel)**2 #from Meier and A'Hearn 1997, Jackson et al. 1982
        t_p = 1/(1.86e-6+1.76e-6+6.35e-7)*r_hel**2 #from Huebner et al. 1992 for CO2 -> CO + O
        t_p_d = 1/(1.66e-7)*r_hel**2
        t_gd = 5.1e6*(r_hel)**2 #from Xie and Mumma 1992, at solar min.
        v_p = outflow_vel_cs2(r_hel)
        v_d = outflow_vel_oh(r_hel)  #m/s
        v_gd = outflow_vel_oh(r_hel)
    elif molecule == 'h2s':
        t_d = 105*(r_hel)**2 #105s from Meier and A'Hearn 1997 , Kim and A'Hearn 1992, but note it is for -28.5 km/s and is likely in need of update()
        t_p = (1/(3.2e-4+5.64e-7+1.47e-7+7.26e-8))*r_hel**2 #Double check Huebner et al. 1992, 2015
        t_p_d = 4150*(r_hel)**2 #4150 from Meier and A'Hearn 1997, Eberhardt et al. 1994, 3124s from Huebner et al. 2015
        t_gd = 9.1e5*(r_hel)**2 # 9.1e5 for S from Huebner et al. 1992, at solar min. Decreases to 4.2e5 at solar max.
        v_p = outflow_vel_h2o(r_hel)
        v_d = outflow_vel_oh(r_hel)  #m/s
        v_gd =  outflow_vel_oh(r_hel) #m/s outflow_vel_oh(r_hel) for S
    else:
        print('This molecule not yet available, please see documentation')
    beta_1 = 1/(t_p*v_p)# inverse total scale length of parent in m
    beta_1_d = 1/(t_p_d*v_p) #dissociative length of parent in m, can be found in Huebner et al. 1992
    beta_2 = 1/(t_d*v_d)# inverse scale length of daughter in m
    beta_3 = 1/(t_gd*v_gd)# inverse scale length of granddaughter atom in m
    A = -(beta_1*beta_1_d)/((beta_1-beta_2)*(beta_3-beta_1))+(beta_1_d)/(beta_3-beta_1)
    B = -(beta_1*beta_1_d)/((beta_2-beta_1)*(beta_3-beta_1))+(beta_1_d*beta_2)/((beta_2-beta_1)*(beta_3-beta_2))-(beta_1_d)/(beta_3-beta_1)       
    C = -A-B
    r = np.sqrt(l**2 + r_p**2)
    #A = (beta_1*beta_2)/((beta_1-beta_2)*(beta_1-beta_3))
    #B = -A*(beta_1-beta_2)/(beta_2-beta_3)
    #C = -B*(beta_1-beta_2)/((beta_1-beta_3))
    first = 1/(4*np.pi*r**2*v_p)
    second = A*np.exp(-r*beta_1)+B*np.exp(-beta_2*r)+C*np.exp(-beta_3*r) 
    mol_den = first*second
    if circle == True:
        mol_den = 2*np.pi*r_p*mol_den
    return abs(mol_den)    
def haser_density_three_comp_odell(r_p,r_hel,molecule,circle = False):
    '''
          _ _ _ l _ _ 
          |         /
    r_p   |        /         
          |       /
          |      /
          |     / r (calculated in each step based on supplied values of l and r_p)
          |    /
          |   /
          |  /
          | /
    comet *
    
    l, a float, is the distance away from a line projected straight from the nucleus that intersects with the line of sight.
    r_p, a float, is the impact parameter from the nucleus along the line of sight as seen from the observer.
    r_hel, a float, is the heliocentric distance in astronomical units.
    molecule, a string, is the name of the parent molecule being modeled'''
    if molecule == 'h2o':
        t_p = t_H2O*r_hel**2
        t_d = t_OH*r_hel**2
        t_gd = t_H*r_hel**2
        v_p = outflow_vel_h2o(r_hel)
        v_d = outflow_vel_oh(r_hel)
        v_gd = v_H #note that there should be a two-velocity component here that is not yet adequately implemented in this model. 
    elif molecule == 'cs2':
        t_d = 1e5 #from Meier and A'Hearn 1997,
        t_p = 590*(r_hel)**2 #590 from Meier and A'Hearn 1997, Jackson, Butterworth, and Ballard 1986 (http://articles.adsabs.harvard.edu//full/1986ApJ...304..515J/0000516.000.html)
        t_gd = 9.1e5*(r_hel)**2  
        v_p = outflow_vel_cs2(r_hel)
        v_d = outflow_vel_oh(r_hel)  #m/s
        v_gd = outflow_vel_oh(r_hel)
    elif molecule == 'co2':
        '''NOT YET READY NEED TO GET VALUES'''
        t_d = 9.1e5*(r_hel)**2 #from Meier and A'Hearn 1997, Jackson et al. 1982
        t_p = 590*(r_hel)**2 #from Meier and A'Hearn 1997, Jackson, Butterworth, and Ballard 1986 (http://articles.adsabs.harvard.edu//full/1986ApJ...304..515J/0000516.000.html)
        t_gd = 5.1e6*(r_hel)**2 #from Xie and Mumma 1992, at solar min.
        v_p = outflow_vel_cs2(r_hel)
        v_d = outflow_vel_oh(r_hel)  #m/s
        v_gd = outflow_vel_oh(r_hel)
    elif molecule == 'h2s':
        t_d = 105*(r_hel)**2 #from Meier and A'Hearn 1997 , Kim and A'Hearn 1992, but note it is for -28.5 km/s and is likely in need of update()
        t_p = 4150*(r_hel)**2 #from Meier and A'Hearn 1997, Eberhardt et al. 1994
        t_gd = 9.1e5*(r_hel)**2 #from Huebner et al. 1992, at solar min. Decreases to 4.2e5 at solar max.
        v_p = outflow_vel_h2o(r_hel)
        v_d = outflow_vel_oh(r_hel)  #m/s
        v_gd =  outflow_vel_oh(r_hel) #m/s outflow_vel_oh(r_hel)
    else:
        print('This molecule not yet available, please see documentation')
    beta_1 = 1/(t_p*v_p)# inverse scale length of parent in m
    beta_2 = 1/(t_d*v_d)# inverse scale length of daughter in m
    beta_3 = 1/(t_gd*v_gd)# inverse scale length of grandparent atom in m
    print(beta_1,beta_2,beta_3)
    A = (beta_1*beta_2)/((beta_1-beta_2)*(beta_1-beta_3))
    B = -A*(beta_1-beta_2)/(beta_2-beta_3)
    C = -B*(beta_1-beta_2)/((beta_1-beta_3))
    print(A,B,C)
    first = 1/(2*np.pi*r_p*v_gd)
    second = A*H(beta_1*r_p)+B*H(beta_2*r_p)+C*H(beta_3*r_p) #This calculation is ONLY VALID FOR CIRCULAR APERTURES CENTERED ON THE NUCLEUS (O'Dell et al. 1988)
    col_den = first*second
    if circle == True:
        col_den = col_den
    else:
        print('Only circular aperture calculations available for 3-component Haser model at this stage')
    return abs(col_den),abs(col_den)/50    
def H(x):
    '''This is the integral for the Bessel function used in the O'Dell 3-component'''
    bessel_int = np.pi/2 - integrate.quad(special.k0,0,x)[0]
    print(bessel_int)
    return bessel_int
def plot_haser(col_data):
    plt.scatter(col_data['Pointing']/206265*col_data['Geocentric Distance (AU)']*1.49e8+1,col_data['N_H Ly-Al'])
    r,N_H_h2o_2 = two_component_spatial_profile('h2o',8/206265*0.2152*1.49e11,0.2152,11.64,1.15,7.96,5e27)
    r,N_H_h2s_2 = two_component_spatial_profile('h2s',8/206265*0.2152*1.49e11,0.2152,11.64,1.15,7.96,8e25)
    r,N_H_h2o_3 = three_component_spatial_profile('h2o',8/206265*0.2152*1.49e11,0.2152,11.64,1.15,7.96,5e27)
    r,N_H_h2s_3 = three_component_spatial_profile('h2s',8/206265*0.2152*1.49e11,0.2152,11.64,1.15,7.96,8e25)
    N_H_2=np.add(N_H_h2o_2,N_H_h2s_2)
    N_H_3= np.add(N_H_h2o_3,N_H_h2s_3)
    N_H = np.add(N_H_2,N_H_3)
    plt.plot(r/1000,N_H,label='Total')
    plt.plot(r/1000,N_H_h2o_2,label='H2O 2 Component')
    plt.plot(r/1000,N_H_h2s_2,label='H2S 2 Component')
    plt.plot(r/1000,N_H_h2o_3,label='H2O 3 Component')
    plt.plot(r/1000,N_H_h2s_3,label='H2S 3 Component')
    plt.errorbar(0/206265*0.2152*1.49e8,np.mean(N_H[(r<=(2.5/206265*0.2152*1.49e11))]), xerr=1.25/206265*0.2152*1.49e8)
    plt.errorbar(2.5/206265*0.2152*1.49e8,np.mean(N_H[(r>=(2.5/206265*0.2152*1.49e11))&(r<=(5/206265*0.2152*1.49e11))]), xerr=1.25/206265*0.2152*1.49e8)
    plt.errorbar(8/206265*0.2152*1.49e8,np.mean(N_H[(r>=(6.75/206265*0.2152*1.49e11))&(r<=9.25/206265*0.2152*1.49e11)]), xerr=1.25/206265*0.2152*1.49e8)
    plt.yscale('log')
    plt.legend()
'''The scripts below this point are mostly one-off type scripts that were used for specific purposes. Feel free to use them but they are not as well documented'''

def outflow_vel_cs2(r_hel):
    return 0.58*r_hel**(-0.5)*1000 #m/s Jackson, Butterworth, and Ballard 1986 (http://articles.adsabs.harvard.edu//full/1986ApJ...304..515J/0000516.000.html) and Delsemme 1982
def outflow_vel_cs(r_hel):
    return (0.58+0.01)*r_hel**(-0.5)*1000 #m/s Jackson, Butterworth, and Ballard 1986 (http://articles.adsabs.harvard.edu//full/1986ApJ...304..515J/0000516.000.html) and Delsemme 1982
def outflow_vel_h2o(r_hel):
    return 0.85*r_hel**(-0.5)*1000 #m/s , 0.6 km/s for 46P, 0.85 otherwise
def outflow_vel_c2(r_hel):
    return r_hel**(-0.5)*1000 #m/s , 0.6 km/s for 46P, 0.85 otherwise
def outflow_vel_oh(r_hel):
    return 1.33*r_hel**(-0.5)*1000 #m/s Fink and Combi 2004
def convert_to_photons(flux,wvlngth):
    return flux/(planck*c_light/wvlngth)
def rayleighs_to_erg_flux(flux_rayleigh):
    return fxn
def watts_to_photons(flux,wvlngth):
    return flux/(planck*c_light/wvlngth)
def doppler_shift(wavelength,v_hel):
    beta = v_hel/c_light
    sqrt_beta = np.sqrt((1-beta)/(1+beta))
    return wavelength/sqrt_beta
def solar_flux(cal_wavelength):
    wvlngth = cal_wavelength 
    #cal_flux = spectral_dataframe_ly_al['irradiance (W/m^2/nm)'][(spectral_dataframe_ly_al['wavelength (nm)'] >= wvlngth-0.1) & (spectral_dataframe_ly_al['wavelength (nm)'] >= wvlngth+0.1)]
    #print(cal_flux)
    cal_flux = spectral_dataframe_ly_al['irradiance (W/m^2/nm)'].sum()
    cal_flux = convert_to_photons(cal_flux,wvlngth)
    return cal_flux
def prod_rate_h2o(r_hel):
    return 3E28/(r_hel/1.5E11)**2    
def g_factor_h_al(v_hel,r):
    wv = 6563E-10 #m
    osc_strength = 0.6407
    constant = np.pi*q_e**2/(4*epsilon*m_e*c_light**2)
    cal_wvlngth = doppler_shift(wv,v_hel)
    sol_flux = solar_flux(cal_wvlngth)/r**2
    return constant*sol_flux*cal_wvlngth**2*osc_strength
def g_factor_ly_al(v_hel,r):
    wv = 1216E-10 #m
    osc_strength = 0.416
    constant = np.pi*q_e**2/(4*epsilon*m_e*c_light**2)
    cal_wvlngth = doppler_shift(wv,v_hel)
    sol_flux = solar_flux(cal_wvlngth)/r**2
    return constant*sol_flux*cal_wvlngth**2*osc_strength
def g_factor_CII1335(v_hel,r):
    wv = 1335E-10 #m
    osc_strength = 0.053
    constant = np.pi*q_e**2/(4*epsilon*m_e*c_light**2)
    cal_wvlngth = doppler_shift(wv,v_hel)
    sol_flux = solar_flux(cal_wvlngth)/r**2
    return constant*sol_flux*cal_wvlngth**2*osc_strength
def vol_integral(r,r_hel):
    return 1/(r**2*r_hel**(3/2))
def emissivity_e_impact_ly_al(Q,r_hel):
    cross_section = 3E-18 #meters squared for electron energies from 10-300 eV
    SE_67P = 2.26E23 #phts/s emission from 67P for Ly-al
    Q_67P = 1E28 #mol/s at peak
    d_67P = 2E5 #meters

    r_67P = 1.4*1.5E11 #meters, distance from Sun
    rc_67P = diamagnetic_cavity(7E27)
    rc = diamagnetic_cavity(Q)
    v_h2o_67P = outflow_vel_h2o(r_67P/1.5E11)
    v_h2o_targ = outflow_vel_h2o(r_hel/1.5E11)
    N_67P = Q_67P*v_h2o_67P
    #N_targ = Q*v_h2o_targ
    R_H2O_67P = v_h2o_67P*t_H2O
    R_H2O_targ = v_h2o_targ*t_H2O
    vol_int = integrate.quad(vol_integral,rc,R_H2O_targ, args = (r_hel), limit=200)[0]
    #source_emissivity = (SE_67P)*(Q/Q_67P)*(d_67P**2*r_67P**(3/2))*vol_int*(1-rc**3/R_H2O_targ**3)/(1-rc_67P**3/R_H2O_67P**3)
    source_emissivity = N_67P*cross_section*(Q/Q_67P)*(d_67P**2*r_67P**(3/2))*vol_int*(1-rc**3/R_H2O_targ**3)/(1-rc_67P**3/R_H2O_67P**3)
    return source_emissivity
def emissivity_e_impact_ly_b(Q,r_hel):
    cross_section = (2/7)*3E-18 #meters squared for electron energies from 10-300 eV
    SE_67P = 2.26E23 #phts/s emission from 67P for Ly-al
    Q_67P = 1E28 #mol/s at peak
    d_67P = 2E5 #meters
    
    r_67P = 1.4*1.5E11 #meters, distance from Sun
    rc_67P = diamagnetic_cavity(7E27)
    rc = diamagnetic_cavity(Q)
    v_h2o_67P = outflow_vel_h2o(r_67P/1.5E11)
    v_h2o_targ = outflow_vel_h2o(r_hel/1.5E11)
    N_67P = Q_67P*v_h2o_67P
    #N_targ = Q*v_h2o_targ
    R_H2O_67P = v_h2o_67P*t_H2O
    R_H2O_targ = v_h2o_targ*t_H2O
    vol_int = integrate.quad(vol_integral,rc,R_H2O_targ, args = (r_hel), limit=200)[0]
    #source_emissivity = (SE_67P)*(Q/Q_67P)*(d_67P**2*r_67P**(3/2))*vol_int*(1-rc**3/R_H2O_targ**3)/(1-rc_67P**3/R_H2O_67P**3)
    source_emissivity = N_67P*cross_section*(Q/Q_67P)*(d_67P**2*r_67P**(3/2))*vol_int*(1-rc**3/R_H2O_targ**3)/(1-rc_67P**3/R_H2O_67P**3)
    return source_emissivity
def emissivity_e_impact_C1657_CO2(Q,r_hel):
    cross_section = (2/7)*3E-18 #meters squared for electron energies from 10-300 eV
    SE_67P = 2.26E23 #phts/s emission from 67P for Ly-al
    Q_67P = 1E28 #mol/s at peak
    d_67P = 2E5 #meters
    
    r_67P = 1.4*1.5E11 #meters, distance from Sun
    rc_67P = diamagnetic_cavity(7E27)
    rc = diamagnetic_cavity(Q)
    v_h2o_67P = outflow_vel_h2o(r_67P/1.5E11)
    v_h2o_targ = outflow_vel_h2o(r_hel/1.5E11)
    N_67P = Q_67P*v_h2o_67P
    #N_targ = Q*v_h2o_targ
    R_H2O_67P = v_h2o_67P*t_H2O
    R_H2O_targ = v_h2o_targ*t_H2O
    vol_int = integrate.quad(vol_integral,rc,R_H2O_targ, args = (r_hel), limit=200)[0]
    #source_emissivity = (SE_67P)*(Q/Q_67P)*(d_67P**2*r_67P**(3/2))*vol_int*(1-rc**3/R_H2O_targ**3)/(1-rc_67P**3/R_H2O_67P**3)
    source_emissivity = N_67P*cross_section*(Q/Q_67P)*(d_67P**2*r_67P**(3/2))*vol_int*(1-rc**3/R_H2O_targ**3)/(1-rc_67P**3/R_H2O_67P**3)
    return source_emissivity
def emissivity_e_impact_C1335_CO(Q,r_hel):
    cross_section = 5*3E-18 #meters squared for electron energies from 10-300 eV
    SE_67P = 2.26E22 #phts/s emission from 67P for Ly-al
    Q_67P = 1E27 #mol/s at peak
    d_67P = 2E5 #meters

    r_67P = 1.4*1.5E11 #meters, distance from Sun
    rc_67P = diamagnetic_cavity(7E27)
    rc = diamagnetic_cavity(Q)
    v_h2o_67P = outflow_vel_h2o(r_67P/1.5E11)
    v_h2o_targ = outflow_vel_h2o(r_hel/1.5E11)
    N_67P = Q_67P*v_h2o_67P
    #N_targ = Q*v_h2o_targ
    R_H2O_67P = v_h2o_67P*t_H2O
    R_H2O_targ = v_h2o_targ*t_H2O
    vol_int = integrate.quad(vol_integral,rc,R_H2O_targ, args = (r_hel), limit=200)[0]
    #source_emissivity = (SE_67P)*(Q/Q_67P)*(d_67P**2*r_67P**(3/2))*vol_int*(1-rc**3/R_H2O_targ**3)/(1-rc_67P**3/R_H2O_67P**3)
    source_emissivity = N_67P*cross_section*(Q/Q_67P)*(d_67P**2*r_67P**(3/2))*vol_int*(1-rc**3/R_H2O_targ**3)/(1-rc_67P**3/R_H2O_67P**3)
    return source_emissivity
def emissivity_e_impact_CO_4PG_CO(Q,r_hel):
    cross_section = 5*3E-18 #meters squared for electron energies from 10-300 eV
    SE_67P = 2.26E22 #phts/s emission from 67P for Ly-al
    Q_67P = 1E27 #mol/s at peak
    d_67P = 2E5 #meters
    
    r_67P = 1.4*1.5E11 #meters, distance from Sun
    rc_67P = diamagnetic_cavity(7E27)
    rc = diamagnetic_cavity(Q)
    v_h2o_67P = outflow_vel_h2o(r_67P/1.5E11)
    v_h2o_targ = outflow_vel_h2o(r_hel/1.5E11)
    N_67P = Q_67P*v_h2o_67P
    #N_targ = Q*v_h2o_targ
    R_H2O_67P = v_h2o_67P*t_H2O
    R_H2O_targ = v_h2o_targ*t_H2O
    vol_int = integrate.quad(vol_integral,rc,R_H2O_targ, args = (r_hel), limit=200)[0]
    #source_emissivity = (SE_67P)*(Q/Q_67P)*(d_67P**2*r_67P**(3/2))*vol_int*(1-rc**3/R_H2O_targ**3)/(1-rc_67P**3/R_H2O_67P**3)
    source_emissivity = N_67P*cross_section*(Q/Q_67P)*(d_67P**2*r_67P**(3/2))*vol_int*(1-rc**3/R_H2O_targ**3)/(1-rc_67P**3/R_H2O_67P**3)
    return source_emissivity
def g_factor_ly_beta(v_hel,r):
    wv = 1026E-10 #m
    osc_strength = 0.079142
    constant = np.pi*q_e**2/(4*epsilon*m_e*c_light**2)
    cal_wvlngth = doppler_shift(wv,v_hel)
    sol_flux = solar_flux(cal_wvlngth)**2
    return constant*sol_flux*cal_wvlngth**2*osc_strength
def diamagnetic_cavity(Q):
    #Equation from Goetz et al. 2016 that is only dependent on Q
    return 7.2E-17*Q**0.678 # to get radius of cavity in meters
def e_impact_h2o_h_al(impact_param,r_hel):
    Q = prod_rate_h2o(r_hel)
    cross_section_integration = 6.4E-20 #m^2
    emissivity = Q**2/(impact_param**3*r_hel**1.5)*cross_section_integration
    return emissivity
def LOS_integration_plot_test(r_hel):
    rho_list = np.logspace(4,9,1000)
    km_list = np.logspace(1,6,1000)
    all_vals_H = []
    all_vals_OH = []
    all_vals_simp = []
    all_col = []
    Q = prod_rate_h2o(r_hel)
    for i,rho in enumerate(rho_list):
        val_H = LOS_integration_H(rho,r_hel)
        val_OH = LOS_integration_OH(rho,r_hel)
        val_simp = haser_simple(rho,r_hel)
        col_den = Q*haser_h_col(rho,r_hel)
        all_vals_H.append(val_H)
        all_vals_OH.append(val_OH)
        all_vals_simp.append(val_simp)
        all_col.append(col_den)
    plt.plot(km_list,all_vals_H, label = 'H Column Density (Integral)')
    plt.plot(km_list,all_vals_OH, label = 'OH Column Density')
    plt.plot(rho_list,all_vals_simp, label = 'H$_{2}$O Column Density')
    plt.plot(rho_list,all_col, label = 'H Column Density (Column Eq.)')
    plt.xlabel('Impact Parameter (km)')
    plt.ylabel('Column Density (m$^{-2}$)') 
    plt.legend()
    plt.ylim(1E13,1E18)
    plt.xscale('log')
    plt.yscale('log')
def col_density_check(ephem_dataframe):
    all_col = []
    for i,line in enumerate(ephem_dataframe['JD_TT']):
        r_hel = 1.5E11*np.sqrt(ephem_dataframe['X'].iloc[i]**2+ephem_dataframe['Y'].iloc[i]**2+ephem_dataframe['Z'].iloc[i]**2)
        Q = prod_rate_h2o(r_hel)
        col_den = Q*integrate.quad(haser_h_col,0,1E8, args = (r_hel), limit = 200)[0]
        #col_den = mp.quad(lambda x:haser_h2o(x,Q,r_hel),[0,1E8])
        total_col = np.pi*1E8**2*col_den
        all_col.append(total_col)
        Q = prod_rate_h2o(1.5E11)
        print(str(g_factor_h_al(37317)*Q*integrate.quad(haser_h_col,0,1E8, args = (1.5E11), limit = 500)[0]) + ' photons/s/m^2/nm')
        print(str(g_factor_h_al(37317)*Q*integrate.quad(haser_h_col,0,1E8, args = (1.5E11), limit = 500)[0] /(4*np.pi*(0.01*1.5E11)**2)) + ' photons/s/m^2/nm')   
        print(str(e_impact_h2o_h_al(1E4,1.5E11)/(4*np.pi*(0.0775*1.5E11)**2)/1E10)+' Rayleighs for e-impact on H-alpha')
        plt.figure(1)
        plt.plot(ephem_dataframe['JD_TT'],all_col)
        plt.xlabel('Julian Date')
        plt.ylabel('Total Molecules') 
        plt.yscale('log')
def haser_simple_plot(r_hel):
    rho_array = np.logspace(1,10,num=500)
    plt.plot(rho_array,haser_simple(rho_array,r_hel))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Impact Parameter (m)')
    plt.ylabel('Column Density (mol/m$^{2}$)')
def LUVOIR_comet_detection(Q,molecule,HST=False):
    if HST == False:
        ly_al_A_eff = 9 #meters squared for LUVOIR, per email from Walt Feb 26,2019
        telescope = 'LUVOIR'
    else:
        telescope = 'HST'
        ly_al_A_eff = 0.12 # meters squared for HST
    t_exp = 5400 #seconds
    plt.figure(1)
    plt.axhline(y=5)
    r = []
    cs_pht = []
    cs_e = []
    cs = []
    for r_hel in range(1,35):
        r.append(r_hel)
        #g_f_h_al = g_factor_h_al(0,r_hel)
        if molecule == 'H2O':
            g_f = g_factor_ly_al(0,r_hel)
            e_emission_rate = emissivity_e_impact_ly_al(10**Q,r_hel*1.5E11)
            tot_H = 10**Q*t_H
        elif molecule == 'CO':
            g_f = g_factor_CII1335(0,r_hel)
            print(g_f)
            e_emission_rate = emissivity_e_impact_C1335_CO(10**Q,r_hel*1.5E11)
            tot_H = 10**Q*t_C
        #g_f_ly_b = g_factor_ly_beta(0,r_hel)
       
        #for i,rho in enumerate(rho_list):
        #    tot_H = tot_H + 2*np.pi*rho*LOS_integration_H(1*10**Q,rho,r_hel*1.5E11)
        pht_emission_rate = g_f*tot_H
        pht_signal=((1/(4*np.pi*(r_hel*1.5E11)**2)*(pht_emission_rate)*ly_al_A_eff)*t_exp)**0.5
        e_signal=(e_emission_rate/(r_hel*1.5E11)**2*ly_al_A_eff*t_exp)**0.5
        comet_signal=(pht_signal**2+e_signal**2)**0.5
        cs.append(comet_signal)
        cs_pht.append(pht_signal)
        cs_e.append(e_signal)
    plt.scatter(r,cs,label='Total SNR')
    plt.scatter(r,cs_pht,label='Photon Excitation SNR')
    plt.scatter(r,cs_e,label='Electron Impact SNR')
    plt.xlabel('Heliocentric Distance (AU)')
    plt.ylabel('Signal to Noise Ratio')
    plt.yscale('log')
    plt.legend()
    plt.title('Q = 10$^{'+str(Q)+'}$ mol/s Observed with '+str(telescope))
    for Q in range(24,32):
        plt.figure(2)
        r = []
        cs = []
        es = []
        ps = []
        p_e = []
        for r_hel in range(1,40):
            r.append(r_hel)
            #g_f_h_al = g_factor_h_al(0,r_hel)
            if molecule == 'H2O':
                g_f = g_factor_ly_al(0,r_hel)
                e_emission_rate = emissivity_e_impact_ly_al(10**Q,r_hel*1.5E11)
                tot_H = 10**Q*t_H 
            elif molecule == 'CO':
                g_f = g_factor_CII1335(0,r_hel)
                e_emission_rate = emissivity_e_impact_C1335_CO(10**Q,r_hel*1.5E11)
                tot_H = 10**Q*t_C 
                #g_f_ly_b = g_factor_ly_beta(0,r_hel)

            pht_emission_rate = g_f*tot_H
            #for i,rho in enumerate(rho_list):
            #    tot_H = tot_H + 2*np.pi*rho*LOS_integration_H(1*10**Q,rho,r_hel*1.5E11)
            pht_signal=((1/(4*np.pi*((r_hel-0.9)*1.5E11)**2)*(pht_emission_rate)*ly_al_A_eff)*t_exp)**0.5
            e_signal=(e_emission_rate/((r_hel-0.9)*1.5E11)**2*ly_al_A_eff*t_exp)**0.5
            comet_signal=(pht_signal**2+e_signal**2)**0.5
            cs.append(comet_signal)
            es.append(e_signal)
            ps.append(pht_signal)
            p_e.append(e_signal/pht_signal)
        plt.scatter(r,cs,label='log(Q) ='+ str(Q))
        plt.xlabel('Heliocentric Distance (AU)')
        plt.ylabel('Total Signal to Noise Ratio')
        plt.title('Observed with '+str(telescope))
        plt.axhline(y=5)
        plt.legend()
        plt.yscale('log')
        plt.figure(3)
        plt.scatter(r,p_e,label = 'log(Q) ='+ str(Q))
        plt.xlabel('Heliocentric Distance (AU)')
        plt.ylabel('Electron Impact/Photofluorescence Emission')
        plt.axhline(y=1)
        plt.legend()
def HST_29P_detection(Q,molecule):
    telescope = 'HST'
    ly_al_A_eff = 0.12 # meters squared for HST
    t_exp = np.linspace(1600,30000,50)
    plt.figure(1)
    plt.axhline(y=5)
    g_factor_CO = 2.2E-7/(5.78)**2# phts/sec Feldman et al. 1976 scaled to 5.78AU
    r_hel = 5.77
    r = []
    cs_pht = []
    cs_e = []
    cs = []
    for t in t_exp:
            #r.append(r_hel)
        #g_f_h_al = g_factor_h_al(0,r_hel)
        if molecule == 'H2O':
            g_f = g_factor_ly_al(0,r_hel)
            e_emission_rate = emissivity_e_impact_ly_al(10**Q,r_hel*1.5E11)
            tot_H = 10**Q*t_H
        elif molecule == 'CO':
            g_f = g_factor_CII1335(0,r_hel)
            e_emission_rate = emissivity_e_impact_C1335_CO(10**Q,r_hel*1.5E11)
            tot_H = 10**Q*t_C
        #g_f_ly_b = g_factor_ly_beta(0,r_hel)
       
        #for i,rho in enumerate(rho_list):
        #    tot_H = tot_H + 2*np.pi*rho*LOS_integration_H(1*10**Q,rho,r_hel*1.5E11)
        pht_emission_rate = g_f*tot_H
        pht_signal=((1/(4*np.pi*(r_hel*1.5E11)**2)*(pht_emission_rate)*ly_al_A_eff)*t)**0.5
        e_signal=(e_emission_rate/(r_hel*1.5E11)**2*ly_al_A_eff*t)**0.5
        comet_signal=(pht_signal**2+e_signal**2)**0.5
        cs.append(comet_signal)
        cs_pht.append(pht_signal)
        cs_e.append(e_signal)
    plt.scatter(t_exp,cs,label='Total SNR')
    plt.scatter(t_exp,cs_pht,label='Photon Excitation SNR')
    plt.scatter(t_exp,cs_e,label='Electron Impact SNR')
    plt.xlabel('Exposure Time (s)')
    plt.ylabel('Signal to Noise Ratio')
    plt.yscale('log')
    plt.legend()
    plt.title('Q = 10$^{'+str(Q)+'}$ mol/s Observed with '+str(telescope))    
def diamagnetic_cavity_plot(self):
    Q = np.linspace(24,31,7)
    plt.figure(1)
    plt.plot(Q,diamagnetic_cavity(10**Q))
    plt.xlabel('Log(Q)')
    plt.ylabel('Diamagnetic Cavity Radius (km)')
    plt.yscale('log')            
def watts_to_photons(flux,wvlngth):
    return flux/(planck*c_light/wvlngth)
def hst_sensitivity(flux_lim):
    '''flux_lim is in ergs/cm^2/s'''
    flux_lim_phts = watts_to_photons(flux_lim/1e7,1419.3*1e-10)
    r_hel_range = np.linspace(1,15,300)
    v_hel_range = np.linspace(-30,30,60)
    Q_full,r_full,v_full = [],[],[]
    for r_hel in r_hel_range:
        for v_hel in v_hel_range:
            r_hel_m = au_to_m(r_hel)
            delta = r_hel-1.5
            delta_cm = au_to_m(delta)*100
            r_cos = 1.25/ 206265.0 * au_to_m(delta)
            g_co4pg,g_co4pg_total = g_factor_co_4pg.co_4pg(r_hel,v_hel)
            t_COscale = t_CO/(r_hel**2)
            #H = aperture_integration_CO(r_cos,r_hel_m)
            Q = flux_lim_phts/(g_co4pg_total*t_COscale)*4*np.pi*delta_cm**2
            Q_full.append(Q)
            r_full.append(r_hel)
            v_full.append(v_hel)
    fig,ax = plt.subplots()
    Q_err = 0.5*np.array(Q_full)
    sc = ax.errorbar(r_full,Q_full,yerr=Q_err)
    #cbar = fig.colorbar(sc)
    #cbar.ax.set_ylabel('Heliocentric Velocity (km/s)')
    plt.xlabel('Heliocentric Distance (au)')
    plt.ylabel('Upper Limit for Q$_{CO}$ detection via CO 4PG')
    plt.yscale('log')
    #plt.colorbar()
def plots(r_hel):
    impact_param = np.logspace(4,9, num=500)
    plt.figure(1)
    Q = prod_rate_h2o(r_hel)
    all_vals_H = []
    all_vals_OH = []
    for i,rho in enumerate(impact_param):
        val_H = LOS_integration_H(rho,r_hel)
        val_OH = LOS_integration_OH(rho,r_hel)
        all_vals_H.append(val_H)
        all_vals_OH.append(val_OH)
        #plt.plot(impact_param,Q*haser_h_col(impact_param,1.5E11))
        plt.plot(impact_param,all_vals_H, label = 'H')
        plt.plot(impact_param,all_vals_OH, label = 'OH')
        plt.xscale('log')
        plt.yscale('log')   
        plt.xlabel('Impact Parameter (m)')
        plt.ylabel('Column Density (mol/m$^{2}$)')
        plt.legend()

    #plt.figure(2)
    #plt.plot(ephem_dataframe['JD_TT'],np.sqrt(ephem_dataframe['X']**2+ephem_dataframe['Y']**2+ephem_dataframe['Z']**2))
    #plt.xlabel('Julian Date')
    #plt.ylabel('Heliocentric Distance (AU)') 
    
    #plt.figure(4)
    #plt.plot(np.sqrt(ephem_dataframe['X']**2+ephem_dataframe['Y']**2+ephem_dataframe['Z']**2),outflow_vel_h2o(np.sqrt(ephem_dataframe['X']**2+ephem_dataframe['Y']**2+ephem_dataframe['Z']**2)))
    #plt.xlabel('Heliocentric Distance (AU)')
    #plt.ylabel('Water Outflow Velocity (m/s)')     
def gen_dataset(r_hel):
    impact_param = np.logspace(2,9, num=500)
    theta = np.linspace(0,359,num = 360)
    theta,rads = np.meshgrid(theta, impact_param)
    #print(rads)
    for i,rad in enumerate(rads):
        for j,single in enumerate(rad):
            val_H = LOS_integration_OH(single,r_hel)
            if j ==0:
                Z_row=np.array([val_H])
            else:
                Z_row =np.append(Z_row,[val_H])
            #print(val_H)
        if i == 0:
            Z = Z_row
            print('First Try')
            #print(Z)
        else:
            Z=np.vstack((Z,Z_row))
            #print(Z)
    return Z
def gen_dataset_hst_cos(r_hel,delta):
    r_cos = 1.25/ 206265.0 * au_to_m(delta)
    arcsecond_conv = 1/ 206265.0 * au_to_m(delta)
    impact_param = np.linspace(100,25*arcsecond_conv+5*r_cos, num=500)
    theta = np.linspace(0,359,num = 360)
    theta,rads = np.meshgrid(theta, impact_param)
    #print(rads)
    for i,rad in enumerate(rads):
        for j,single in enumerate(rad):
            val_H2O = LOS_integration_H2O(single,r_hel)
            if j ==0:
                Z_row=np.array([val_H2O])
            else:
                Z_row =np.append(Z_row,[val_H2O])
            #print(val_H)
        if i == 0:
            Z = Z_row
            print('First Try')
            #print(Z)
        else:
            Z=np.vstack((Z,Z_row))
            #print(Z)
    return Z
def gen_dataset_hst_stis(r_hel,delta):
    r_cos = 1.25/ 206265.0 * au_to_m(delta)
    arcsecond_conv = 1/ 206265.0 * au_to_m(delta)
    impact_param = np.linspace(100,52*arcsecond_conv+5*r_cos, num=500)
    theta = np.linspace(0,359,num = 360)
    theta,rads = np.meshgrid(theta, impact_param)
    #print(rads)
    for i,rad in enumerate(rads):
        for j,single in enumerate(rad):
            val_H2O = LOS_integration_H2O(single,r_hel)
            if j ==0:
                Z_row=np.array([val_H2O])
            else:
                Z_row =np.append(Z_row,[val_H2O])
            #print(val_H)
        if i == 0:
            Z = Z_row
            print('First Try')
            #print(Z)
        else:
            Z=np.vstack((Z,Z_row))
            #print(Z)
    return Z
def radial_plots(Z):  
    fig = plt.figure(1)
    ax = fig.add_subplot(111,polar = True)
    impact_param = np.logspace(4,9, num=500)
    theta = np.linspace(0,359,num = 360)
    theta,rads = np.meshgrid(theta, impact_param)
    norm = matplotlib.colors.LogNorm(vmin = np.max(Z)/1000, vmax = np.max(Z), clip = False)
    cax = ax.pcolormesh(theta,rads, Z ,norm = norm, cmap = cm.plasma)
    plt.colorbar(cax)
    #cax.set_rscale('log') 
    plt.figure(2)
    plt.imshow(Z,origin='lower')
def hst_cos_radial_plots(Z,delta):  
    fig = plt.figure(1)
    ax = fig.add_subplot(111,polar = True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    r_cos = 1.25/ 206265.0 * au_to_m(delta)
    arcsecond_conv = 1/ 206265.0 * au_to_m(delta)
    impact_param = np.linspace(100,8*arcsecond_conv+5*r_cos, num=500)
    theta = np.linspace(0,359,num = 360)
    theta,rads = np.meshgrid(theta, impact_param)
    #norm = matplotlib.colors.LogNorm(vmin = np.min(Z)*10, vmax = np.max(Z), clip = False)
    cax = ax.pcolormesh(theta,rads, Z/10**-17 , cmap = cm.plasma)
    def get_cartesian(r,theta):
        x=r*np.cos(theta)
        y=r*np.sin(theta)
        return x,y
    x_1,y_1 = get_cartesian(8*arcsecond_conv,225*np.pi/180)
    x_2,y_2 = get_cartesian(2.5*arcsecond_conv,225*np.pi/180)
    offset_1 = plt.Circle((0,0),r_cos,transform=ax.transData._b,color='black',fill=False)
    offset_2 = plt.Circle((x_1,y_1),r_cos,transform=ax.transData._b,color='white',fill=False)
    offset_3 = plt.Circle((x_2,y_2),r_cos,transform=ax.transData._b,color='white',fill=False)
    ax.annotate('Centered Aperture',xy=(90*np.pi/180,arcsecond_conv),xytext=(90*np.pi/180,5*arcsecond_conv),color = 'white',arrowprops=dict(facecolor='white'))
    ax.annotate('8" Offset',xy=(220*np.pi/180,8*arcsecond_conv),xytext=(180*np.pi/180,12*arcsecond_conv),color = 'white',arrowprops=dict(facecolor='white'))
    ax.annotate('2.5" Offset',xy=(227*np.pi/180,2.5*arcsecond_conv),xytext=(270*np.pi/180,6*arcsecond_conv),color = 'white',arrowprops=dict(facecolor='white'))
    cbar = plt.colorbar(cax)
    cbar.ax.set_ylabel('N/Q (10$^{-15}$ s cm$^{-2}$)')
    #cax.set_rscale('log')
    x,y = get_cartesian(rads,theta)
    
    prof_1 = Z[(x<=0.25*arcsecond_conv) & (x>=-0.25*arcsecond_conv)]
    prof_2 = Z[(x<=2.5*arcsecond_conv+0.25*arcsecond_conv) & (x>=2.5*arcsecond_conv-0.25*arcsecond_conv)]
    prof_3 = Z[(x<=8*arcsecond_conv+0.25*arcsecond_conv) & (x>=8*arcsecond_conv-0.25*arcsecond_conv)]
    ax.add_artist(offset_1)
    ax.add_artist(offset_2)
    ax.add_artist(offset_3)
    bins_cen,col_cen = mean_binning(y[(x<=0.25*arcsecond_conv) & (x>=-0.25*arcsecond_conv)],prof_1,51)
    bins_2,col_2 = mean_binning(y[(x<=2.5*arcsecond_conv+0.25*arcsecond_conv) & (x>=2.5*arcsecond_conv-0.25*arcsecond_conv)],prof_2,51)
    bins_8,col_8 = mean_binning(y[(x<=8*arcsecond_conv+0.25*arcsecond_conv) & (x>=8*arcsecond_conv-0.25*arcsecond_conv)],prof_3,51)
    plt.figure(2)
    plt.scatter(bins_cen,col_cen, label='Centered')
    plt.scatter(bins_2,col_2,label='2.5" offset')
    plt.scatter(bins_8,col_8,label='8" offset')
    plt.xlabel('Y Distance (m)')
    plt.ylabel('Normalized Column Density')
    plt.legend()
    plt.figure(3)
    plt.scatter(rads[(x<=0.25*arcsecond_conv) & (x>=-0.25*arcsecond_conv)],prof_1, label='Centered')
    plt.scatter(rads[(x<=2.5*arcsecond_conv+0.25*arcsecond_conv) & (x>=2.5*arcsecond_conv-0.25*arcsecond_conv)],prof_2,label='2.5" offset')
    plt.scatter(rads[(x<=8*arcsecond_conv+0.25*arcsecond_conv) & (x>=8*arcsecond_conv-0.25*arcsecond_conv)],prof_3,label='8" offset')
    plt.xlabel('Radial Distance (m)')
def LOS_integration_H2O(r_p,r_hel):  #### radius in m
    #r_max = 1E8 #m
    result,err = integrate.quad(haser_H2O_den,100,1E9,args=(r_p,r_hel))#,epsabs=1.49e-08, epsrel=1.49e-08)
    return 2*result
def LOS_integration_OH(r_p,r_hel):  #### radius in m
    #r_max = 1E8 #m
    result,err = integrate.quad(haser_OH_den,100,1E9,args=(r_p,r_hel))#,epsabs=1.49e-08, epsrel=1.49e-08)
    return 2*result
def LOS_integration_CO(r_p,r_hel):  #### radius in m
    #r_max = 1E8 #m
    result,err = integrate.quad(haser_CO_den,100,1E9,args=(r_p,r_hel))#,epsabs=1.49e-08, epsrel=1.49e-08)
    return 2*result
def LOS_integration_H(r_p,r_hel):  #### radius in m
    #r_max = 1E8 #m
    Q = prod_rate_h2o(r_hel)
    result = integrate.quad(haser_H_den,1,1E9,args=(r_p,r_hel))#,epsabs=1.49e-08, epsrel=1.49e-08)
    return 2*Q*result[0]
def haser_H_den(l,r_p,r_hel):
    v_H2O = outflow_vel_h2o(r_hel/1.5E11)
    r = np.sqrt(l**2 + r_p**2)
    b_p = (t_H2O*v_H2O)# scale length of parent molecule in m
    b_d = (t_H*v_H)# scale length of daughter atom in m
    first = 1/(4*np.pi*r**2*v_H)
    scale_length_ratio = (b_d/(b_p-b_d))
    second = (np.exp(-r/b_p)-np.exp(-r/b_d))
    mol_den = first*second*scale_length_ratio
    return mol_den
def haser_OH_den(l,r_p,r_hel):
    v_H2O = outflow_vel_h2o(r_hel/1.5E11)
    r = np.sqrt(l**2 + r_p**2)
    b_p = (t_H2O*v_H2O)# scale length of parent molecule in m
    b_d = (t_OH*v_OH)# scale length of daughter atom in m
    first = 1/(4*np.pi*r**2*v_OH)
    scale_length_ratio = (b_d/(b_p-b_d))
    second = (np.exp(-r/b_p)-np.exp(-r/b_d))
    mol_den = first*second*scale_length_ratio
    return mol_den
def haser_simple(rho,r_hel):
    v = outflow_vel_h2o(r_hel)
    Q = prod_rate_h2o(r_hel)
    b_p =t_H2O*v_H2O# scale length of parent molecule in m
    return 1/(4*np.pi*v_H2O*rho**2)*np.exp(-rho/b_p)
def haser_CO_den(l,r_p,r_hel):
    v_CO = 1000 #m/s
    r = np.sqrt(l**2 + r_p**2)
    b_p = (t_CO*v_CO)# scale length of parent molecule in m
    first = 1/(4*np.pi*r**2*v_CO)
    second = (np.exp(-r/b_p))
    mol_den = first*second
    return mol_den
def haser_H2O_den(l,r_p,r_hel):
    v_H2O = outflow_vel_h2o(r_hel/1.5E11)
    r = np.sqrt(l**2 + r_p**2)
    b_p = (t_H2O*v_H2O)# scale length of parent molecule in m
    first = 1/(4*np.pi*r**2*v_H2O)
    second = np.exp(-r/b_p)
    mol_den = first*second
    return mol_den
def LOS_integration_S(r_p,r_hel):  #### radius in m
    #r_max = 1E8 #m
    result,err = integrate.quad(haser_S_den,100,1E9,args=(r_p,r_hel))#,epsabs=1.49e-08, epsrel=1.49e-08)
    return 2*result
def LOS_integration_S_CS2(r_p,r_hel):  #### radius in m
    #r_max = 1E8 #m
    result,err = integrate.quad(haser_S_den_CS2,100,1E9,args=(r_p,r_hel))#,epsabs=1.49e-08, epsrel=1.49e-08)
    return 2*result
def LOS_integration_S_H2S(r_p,r_hel):  #### radius in m
    #r_max = 1E8 #m
    result,err = integrate.quad(haser_S_den_H2S,100,1E9,args=(r_p,r_hel))#,epsabs=1.49e-08, epsrel=1.49e-08)
    return 2*result
def LOS_integration_S_S2(r_p,r_hel):  #### radius in m
    #r_max = 1E8 #m
    result,err = integrate.quad(haser_S_den_S2,100,1E9,args=(r_p,r_hel))#,epsabs=1.49e-08, epsrel=1.49e-08)
    return 2*result
def aperture_integration_H(distance,r_hel):
    result,err = integrate.quad(LOS_integration_H,100,distance, args=(r_hel))
    return result
def aperture_integration_OH(date,distance,r_hel):
    result,err = integrate.quad(LOS_integration_OH,100,distance, args=(r_hel,date))
    return result
def aperture_integration_CO(distance,r_hel):
    result,err = integrate.quad(LOS_integration_CO,100,distance, args=(r_hel))
    return result
def aperture_integration_S(distance,r_hel):
    result,err = integrate.quad(LOS_integration_S,100,distance, args=(r_hel))
    return result
def aperture_integration_S_from_CS2(distance,r_hel):
    result,err = integrate.quad(LOS_integration_S_CS2,100,distance, args=(r_hel))
    return result 
def aperture_integration_S_from_H2S(distance,r_hel):
    result,err = integrate.quad(LOS_integration_S_H2S,100,distance, args=(r_hel))
    return result 
def aperture_integration_S_from_S2(distance,r_hel):
    result,err = integrate.quad(LOS_integration_S_S2,100,distance, args=(r_hel))
    return result

def cn_density(z,b):
    r = np.sqrt(z**2.0 + b**2.0)
    factor1 = 1.0 / (4.0 * np.pi * r**2.0 * v_outflow * 1000.0)
    factor2 = cn_daughter_length / (cn_parent_length - cn_daughter_length)
    expfactor = (np.exp(-r / ( cn_parent_length * 1000.0)) - np.exp(-r / (cn_daughter_length * 1000.0)))
    #print(r / ( cn_parent_length * 1000.0), r / ( cn_parent_length * 1000.0 * v_outflow))
    return factor1 * factor2 * expfactor

def c2_density(z,b):
    r = np.sqrt(z**2.0 + b**2.0)
    factor1 = 1.0 / (4.0 * np.pi * r**2.0 * v_outflow * 1000.0)
    factor2 = c2_daughter_length / (c2_parent_length - c2_daughter_length)
    expfactor = float(np.exp(-r / ( c2_parent_length * 1000.0)) - np.exp(-r / (c2_daughter_length * 1000.0)))

    return factor1 * factor2 * expfactor
def LOS_andspatial_integral_CN(x,y):
    b = np.sqrt(x**2.0 + y**2.0)
    result, err = integrate.quad(cn_density,100,float(1E9), args=(b,))
    return 2.0 * result

def LOS_andspatial_integral_C2(x,y):
    b = np.sqrt(x**2.0 + y**2.0)
    result, err = integrate.quad(c2_density,100,float(1E9),args=(b,))
    return 2.0 * result
def LOS_andspatial_integral_OH(x,y):
    b = np.sqrt(x**2.0 + y**2.0)
    result, err = integrate.quad(haser_OH_den,100,float(1E9),args=(b,1.23))
    return 2.0 * result
def box_integral_CN_noQ(box):
    box_width = box[0] * metersperAU / 2.0
    box_length = box[1] * metersperAU / 4.0 # just gonna multiply by 2.....
    #print(box_width * 2.0 / metersperAU * 206265.0 / delta,box_length* 2.0 / metersperAU * 206265.0 / delta)
    result, err = integrate.dblquad(LOS_andspatial_integral_CN, box_width/100, box_width, box_length/100, box_length)
    #print(result)
    return result * 4.0 #4x for box symmetry
def box_integral_C2_noQ(box):
    box_width = box[0] * metersperAU / 2.0
    box_length = box[1] * metersperAU / 4.0 # just gonna multiply by 2.....
    #print(box_width * 2.0 / metersperAU * 206265.0 / delta,box_length* 2.0 / metersperAU * 206265.0 / delta)
    result, err = integrate.dblquad(LOS_andspatial_integral_C2, 0, box_width, 0, box_length)
    #print(result)
    return result * 4.0 #4x for box symmetry
def box_integral_OH_noQ(box,delta):
    box_width = box[0] * au_to_m(delta)*100 / 2.0
    box_length = box[1] * au_to_m(delta)*100 / 4.0 # just gonna multiply by 2.....
    #print(box_width * 2.0 / metersperAU * 206265.0 / delta,box_length* 2.0 / metersperAU * 206265.0 / delta)
    result, err = integrate.dblquad(LOS_andspatial_integral_OH, 0, box_width, 0, box_length)
    #print(result)
    return result * 4.0 #4x for box symmetry
def box_integral_CN(box,Q):
    box_width = box[0] * metersperAU / 2.0
    box_length = box[1] * metersperAU / 4.0 # just gonna multiply by 2.....
    #print(box_width * 2.0 / metersperAU * 206265.0 / delta,box_length* 2.0 / metersperAU * 206265.0 / delta)
    result, err = integrate.dblquad(LOS_andspatial_integral_CN, box_width/100, box_width, box_length/100, box_length)
    #print(result)
    return result * 4.0 * Q

def box_integral_C2(box,Q):
    box_width = box[0] * metersperAU / 2.0
    box_length = box[1] * metersperAU / 4.0 # just gonna multiply by 2.....
    #print(box_width * 2.0 / metersperAU * 206265.0 / delta,box_length* 2.0 / metersperAU * 206265.0 / delta)
    result, err = integrate.dblquad(LOS_andspatial_integral_C2, 0, box_width, 0, box_length)
    #print(result)
    return result * 4.0 * Q