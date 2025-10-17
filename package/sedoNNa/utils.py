'''
This code is not quite standalone yet
'''



import numpy as np
import os
import sys
import glob
sys.path.insert(0, os.environ['astro_code_dir'])
from Astro_useful_funcs import *
from Analysis_useful_funcs import *
from scipy.interpolate import interp1d
from astropy.constants import c


def get_cmfgen_lc(cmfgen_spectra_dir, filt, filt_profile_dir = '../../../filter_profs'):
    filt_names = {'SDSS_u':'SLOAN_SDSS.u', 'SDSS_g':'SLOAN_SDSS.g', 'SDSS_r':'SLOAN_SDSS.r', 'SDSS_i':'SLOAN_SDSS.i', 'SDSS_z':'SLOAN_SDSS.z'}
    filt_name = filt_names[filt]
    cmf_spectra = glob.glob(cmfgen_spectra_dir+"/*fl")
    
    fold = cmfgen_spectra_dir.split('/')[-3]
    idx = fold[2:fold.index("_")]
    
    time_array = np.loadtxt(cmfgen_spectra_dir+"/he"+str(idx)+"_list", dtype = object)
    times_list = time_array[:,0]
    idxs_list = time_array[:,1]
    
    times_tor = []
    mags_tor = []
    
    for cmf_spec in cmf_spectra:
        short_name = cmf_spec.split('/')[-1][:-3]
        
        if short_name not in idxs_list:
            continue
        
        time = times_list[list(idxs_list).index(short_name)]
            
        time = float(time)
            
        spec_data = np.loadtxt(cmf_spec)
        lam = spec_data[:,0]
        flam = spec_data[:,1] # this is at a distance of 1 kpc. I need to convert it to be at 10 pc so I get proper AB magnitudes
        
        flam *= 100**2.0 # now it's at a distance of 10pc. 
        
        filter_data = np.loadtxt(filt_profile_dir+"/"+str(filt_name)+".dat")
        filter_wavs = filter_data[:, 0]
        filter_transmission = filter_data[:, 1]

        # Normalize filter transmission: area under curve = 1
        filter_transmission /= np.trapz(filter_transmission, filter_wavs)

        #get the magnitude using 
        mag = float(compute_photometry(lam, flam, filter_wavs, filter_transmission))
        mags_tor.append(mag)
        times_tor.append(time)
    return np.asarray(mags_tor), np.asarray(times_tor)


def interp1d_torch(x, xp, fp):
    """1D linear interpolation for PyTorch tensors, equivalent to numpy.interp.
    - x: target points [N]
    - xp: known points [M], must be increasing
    - fp: known values at xp [M]
    Returns: interpolated values at x [N]
    """
    # Ensure xp is increasing
    inds = torch.searchsorted(xp, x, right=True).clamp(1, len(xp)-1)
    x_lo = xp[inds-1]
    x_hi = xp[inds]
    y_lo = fp[inds-1]
    y_hi = fp[inds]
    slope = (y_hi - y_lo) / (x_hi - x_lo)
    return y_lo + slope * (x - x_lo)

def compute_photometry(wavelengths, fluxes, filter_wavelengths, filter_transmission):
    """
    wavelengths: 1D array [Angstrom]
    fluxes: 1D array [erg/s/cm^2/Angstrom]
    filter_wavelengths: 1D array [Angstrom]
    filter_transmission: 1D array
    Returns: AB magnitude
    """
    # Convert inputs to astropy quantities
    wave = wavelengths * u.Angstrom
    flux = fluxes * u.erg / (u.s * u.cm**2 * u.Angstrom)

    # Interpolate filter onto spectrum wavelength grid
    interp_T = interp1d(filter_wavelengths, filter_transmission, bounds_error=False, fill_value=0.0)
    T = interp_T(wavelengths)

    # Compute effective flux (λ-weighted integral)
    numerator = np.trapz((flux * T * wave).to(u.erg / u.s / u.cm**2), x=wave)
    denominator = np.trapz((T * wave).to(u.Angstrom), x=wave)
    flux_lambda = (numerator / denominator).to(u.erg / (u.s * u.cm**2 * u.Angstrom))

    # Compute effective wavelength
    eff_lambda = (np.trapz(wave * T, wave) / np.trapz(T, wave)).to(u.Angstrom)

    # Convert to F_nu: F_nu = F_lambda * (lambda^2 / c)
    flux_nu = (flux_lambda * eff_lambda**2 / c).to(u.erg / (u.s * u.cm**2 * u.Hz))

    # Compute AB magnitude
    ab_mag = -2.5 * np.log10(flux_nu.value) - 48.6
    return ab_mag


import torch

def tensor_compute_photometry(wavelengths, fluxes, filter_wavelengths, filter_transmission, eps=1e-30):
    """
    All inputs: 1D torch tensors, physically in:
    - wavelengths: [Angstrom]
    - fluxes: [erg/s/cm^2/Angstrom]
    - filter_wavelengths: [Angstrom]
    - filter_transmission: dimensionless
    Returns: AB magnitude (tensor, grads preserved)
    """

    # Speed of light in Angstrom/s (for cgs)
    c = 2.99792458e18

    # Interpolate filter to wavelengths grid
    # (quadratic interpolation is differentiable in PyTorch)
    T = interp1d_torch(wavelengths, filter_wavelengths, filter_transmission)

    # Numerator: int(F_lambda * T * lambda d_lambda)
    numerator = torch.trapz(fluxes * T * wavelengths, wavelengths)
    # Denominator: int(T * lambda d_lambda)
    denominator = torch.trapz(T * wavelengths, wavelengths) + eps  # avoid /0

    # Effective mean flux in filter
    flux_lambda = numerator / denominator

    # Effective wavelength (for filter)
    eff_lambda = (
        torch.trapz(wavelengths * T, wavelengths) /
        (torch.trapz(T, wavelengths) + eps)
    )  # [Angstrom]

    # Convert F_lambda to F_nu
    flux_nu = (flux_lambda * eff_lambda ** 2 / c) + eps  # [erg/s/cm^2/Hz]

    # AB magnitude (add eps for stability if required)
    ab_mag = -2.5 * torch.log10(flux_nu + eps) - 48.6
    return ab_mag


def lc_from_spec_final(times, freq, flux, filter_prof_dir, wav_low = -np.inf, wav_high = np.inf, time_low = 0, time_high = np.inf):

    c = 2.99792458e8 * u.meter/u.second
    angstrom = 1.0E-10 * u.meter
    erg = 1.0E-7 * u.kilogram*(u.meter/u.second)**2
    cm = 1.0E-2 * u.meter
    hz = 1/u.second
    freq_flux = erg/(cm**2*u.second*hz)
    wav_flux = erg/(cm**2*angstrom*u.second)
    sec_per_day = 86400
    
    
    #convert that into flux and wavelength arrays at each time
    times = times*u.second.to(u.day)
    freq = freq*hz
    flux = flux*erg/(u.second*hz) # this is the total emitted power at that frequency. So this is in units of erg/s/hz. I need to divide by the surface area of a sphere with radius of 10 parsec
    flux /= (4 * np.pi * (10.0*u.parsec)**2)
    flux = flux.to(freq_flux) #now, this is in units of erg/s/hz/cm^2

    flux_lam = (flux * freq**2/c).to(wav_flux).value
    wav = (c/freq).to(angstrom).value

    gidx = np.where((wav > wav_low)&(wav < wav_high))
    times = times[gidx]
    flux_lam = flux_lam[gidx]
    wav = wav[gidx]

    gidx = np.where((times > time_low)&(times < time_high))
    times = times[gidx]
    flux_lam = flux_lam[gidx]
    wav = wav[gidx]


    lc_dict = {"times":[]}
    filter_files = glob.glob(filter_prof_dir+"/*.dat")
    filter_profs = [i.split('/')[-1][:-4] for i in filter_files]
    #filter_profs = ['SLOAN_SDSS.u', 'SLOAN_SDSS.g', 'SLOAN_SDSS.r', 'SLOAN_SDSS.i', 'SLOAN_SDSS.z']
    #for each time, get photometry in ugriz filters
    unique_times = list(set(times))
    for unique_time in unique_times:
        time_idx = np.where(times == unique_time)
        unique_flux_lam = flux_lam[time_idx]
        unique_wav = wav[time_idx]
        time_days = unique_time


        lc_dict['times'].append(time_days)
        for filter_prof in filter_profs:
            if filter_prof not in lc_dict.keys():
                lc_dict[filter_prof] = []

            filter_data = np.loadtxt(filter_prof_dir+"/"+str(filter_prof)+".dat")
            filter_wavs = filter_data[:, 0]
            filter_transmission = filter_data[:, 1]

            # Normalize filter transmission: area under curve = 1
            filter_transmission /= np.trapz(filter_transmission, filter_wavs)

            flux_phot = compute_photometry(unique_wav, unique_flux_lam, filter_wavs, filter_transmission)
            lc_dict[filter_prof].append(flux_phot)
    return lc_dict


import numpy as np
import os
import sys
sys.path.insert(0, os.environ['astro_code_dir'])
from Astro_useful_funcs import *
from Analysis_useful_funcs import *
sys.path.insert(0, os.environ['karth_home']+"/aAOperation/Building Riem/riem_profiles/")
from Riem_Structures import *
from decimal import Decimal

import torch
from torchvision import datasets,transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import csv
import subprocess
import re
from scipy.interpolate import interp1d
import colorsys
import warnings
import json

hz = u.second**-1
jansky = 1.0E-23 * u.erg/(u.second*(u.cm)**2*hz)
angstrom = 1.0E-10 * u.meter

true  = True
false = False
def Str(g):
    return str(g)

def stR(g):
    return str(g)

def sTr(g):
    return str(g)

num_input = 256 #every autoencoder takes in 256 numbers as built, so I will spit out distributions that are 256 long, and all the model fed into sedona will have 256

time_0 = 8.68906E+04
msun_grams = 1.988E33
kms_in_cms = 1.0E5


avg_fractions_data = np.loadtxt("/n/holylabs/LABS/avillar_lab/Users/kyadavalli/useful_funcs/Average Mass Fractions by Layer.txt")

woosley_models = "/n/holylabs/LABS/avillar_lab/Users/kyadavalli/Running Stan Original Mixed Models/running models/"


cleaning_spec_script = '''import os
import glob
import sys
sys.path.insert(0, os.environ['astro_code_dir'])
from Astro_useful_funcs import *
from Analysis_useful_funcs import *

if not os.path.isfile("spectrum_final.h5"):
    spectrum_files = []
    max_num = 0
    for file in os.listdir():
        if not os.path.isdir(file) and "spectrum_" in file:
            spectrum_files.append(file)
            num = int(file[file.index("spectrum_")+9:file.index(".")])
            if num > max_num:
                max_num = num

    os.system("cp spectrum_"+str(max_num)+".h5 spectrum_final.h5")
    print("Maximum spectrum found is "+str(max_num))
    
files = glob.glob("spectrum*")
files_to_del = []
for file in files:
    if "final" not in file:
        files_to_del.append(file)
for file in files_to_del:
    os.remove(file)'''



cleaning_up_script = '''import os
import glob
import sys
sys.path.insert(0, os.environ['karth_home']+"/useful_funcs/")
from support_funcs1 import *

chk_files = glob.glob("*chk*")

highest_chk = -np.inf
file_to_use = ''
for chk_file in chk_files:
    dig = int(extract_digits(os.path.basename(chk_file)))
    if dig > highest_chk:
        highest_chk = dig
        file_to_use = chk_file
            
for f in chk_files:
    if file_to_use != f:
        rm(f)
        
        
if os.path.exists("chk_final.h5"):
    rm(file_to_use)
    file_to_use = "chk_final.h5"




mkdir("storage/")
remaining_files = os.listdir()
for file in remaining_files:
    if file not in ['lightcurve.out', 'storage', 'mod.mod', 'cleaning_up.py', 'sample.txt', 'spectrum_final.dat', 'spectrum_final.h5'] and file[0] != '.':
        try:
            mv(file, "storage/")
        except:
            print("Tried and failed to move "+str(file)+" to storage")
            pass
os.system("tar zcf storage.tar.gz storage; rm -r storage/")'''

def rebin_cdf_preserving(vel_list, masses, num_edges=256):
    """
    Rebin a mass histogram into a new set of uniformly spaced velocity bin edges.

    Parameters:
      vel_edges: array-like, shape (N+1,), original bin edges [v0, v1, ..., vN]
      masses: array-like, shape (N,), masses in each original bin
      num_edges: int, number of edges for new binning (resulting number of bins = num_edges-1)

    Returns:
      new_vel_edges: array, shape (num_edges,)
      new_masses: array, shape (num_edges-1,)
    """
    vel_edges = np.insert(vel_list, 0, 0)
    vmin = vel_edges[0]
    vmax = vel_edges[-1]
    new_vel_edges = np.linspace(vmin, vmax, num_edges)
    
    # Compute original cumulative mass (CDF) at each bin edge
    cdf = np.zeros_like(vel_edges)
    cdf[1:] = np.cumsum(masses)
    
    # Interpolate original CDF onto new edges
    new_cdf = np.interp(new_vel_edges, vel_edges, cdf)
    
    # Differences between new CDFs are the masses in new bins
    new_masses = np.diff(new_cdf)
    return new_vel_edges[1:], new_masses
    
eps_ni = 3.9E10 #erg/g/s
eps_co = 6.8E9 #erg/g/s
t_ni = 8.8 #d
t_co = 111.3 #d
def get_arnett_mni(time, bol_lc):
    
    #time better be in days and lc better be the bol_lc in erg/s
    #returns nickel mass in grams
    
    peak_idx = find_nearest(bol_lc, np.max(bol_lc))
    peak_time_d = time[peak_idx]
    peak_lum = np.max(bol_lc)
    
    denom = (eps_ni-eps_co)*np.exp(-1.0*peak_time_d/t_ni) + eps_co*np.exp(-1.0*peak_time_d/t_co)
    return peak_lum/denom

def realign_lc(time, lc, earliest_days = 0):
    
    gidx = np.where(time > earliest_days)
    lc_T = lc[gidx]
    max_idx = find_nearest(lc, np.max(lc_T))
    time -= time[max_idx]
    
    return time, lc
    
def trim_lc(time, lc, min_lim = np.nan, max_time = 75, mag_space = False):
    
    nonzero_idx = np.where(lc != 0)
    lcf = lc[nonzero_idx]
    
    if not mag_space and -13 > np.min(lcf) > -25 and -13 > np.max(lcf) > -25:
        mag_space = True
        warnings.warn("Setting magspace to true for trimming lc")
    
    if mag_space:
        lc = 10**(-1.0*lc)
        
        
    
    
    early_lc_idx = np.where(time < 8)
    early_lc = lc[early_lc_idx]
    early_time = time[early_lc_idx]
    
    early_lc_peak = np.max(early_lc)
    early_peak_idx = find_nearest(lc, early_lc_peak)
    early_peak_time = time[early_peak_idx]
    if early_lc_peak/lc[early_peak_idx-1] > 1.5 and early_lc_peak/lc[early_peak_idx+1] > 1.5:
        lc = lc[early_peak_idx+4:]
        time = time[early_peak_idx+4:]
        
    
    if not np.isnan(max_time):
        gidx = np.where(time < max_time)
        time = time[gidx]
        lc = lc[gidx]
    
    
    if np.isnan(min_lim):
        min_lim = 1.0E-1 * np.max(lc)
    
    
    gidx = np.where(lc > min_lim)
    time = time[gidx]
    lc = lc[gidx]
    
    if mag_space:
        lc = -1.0*np.log10(lc)

    
    return time, lc
    

def trim_realign_lc(time, lc, min_lim = np.nan, max_time = 75, earliest_days = 0):
    #this function will realign the time and lc so that the time of peak will be time = 0, and will trim off parts of the lc that dies off
    
    
    time, lc = trim_lc (time, lc, min_lim = min_lim, max_time = max_time)    
    
    time, lc = realign_lc(time, lc, earliest_days = earliest_days)

    
    return time, lc

def clean_val(val):
    if is_float(val):
        val = float(val)
        if np.log10(abs(val)) > 2 or np.log10(abs(val)) < -2:
            val = '%.1E' % Decimal(str(val))
        else:
            val = round(val, 2)
        val = "$"+str(val)+"$"
    return str(val)
    
    

def get_latex_var(var):
    keys = {'D': r'$\rm \eta_{vel}$', 
            'R_2': r'$\rm{\eta_{He^4}}$', 
            'R_28': r'$\rm{\eta_{Ni^{56}}}$',
            'R_opacity': r'$\rm{\eta_{bulk}}$',
            'max_vel': r'$\Delta$ Vel',
            'min_vel': 'Min. Vel',
            'total_2': r'$\rm{m_{He^{4}}}$', 
            'He Mass': r'$\rm{m_{He^{4}}}$',
            'total_28': r'$\rm{m_{Ni^{56}}}$',
            'Ni Mass': r'$\rm{m_{Ni^{56}}}$',
            'total_opacity':r'$\rm{m_{bulk}}$',
            'Op Mass':r'$\rm{m_{bulk}}$',
            'total_ejecta': r'$\rm{m_{ejecta}}$',
            'Ej Mass': r'$\rm{m_{ejecta}}$',
            'ejecta Mass': r'$\rm{m_{ejecta}}$',
            'velte':r'$vel_{10}$',
            'velt':r'$vel_{25}$',
            'velf':r'$vel_{50}$',
            'vels':r'$vel_{75}$',
            'veln':r'$vel_{90}$',
            'vel_10p':r'$vel_{10}$',
            'vel_25p':r'$vel_{25}$',
            'vel_50p':r'$vel_{50}$',
            'vel_75p':r'$vel_{75}$',
            'vel_90p':r'$vel_{90}$',
            'Vel_10p':r'$vel_{10}$',
            'Vel_25p':r'$vel_{25}$',
            'Vel_50p':r'$vel_{50}$',
            'Vel_75p':r'$vel_{75}$',
            'Vel_90p':r'$vel_{90}$',
            'gamnh':r'$r_{\rm 95,He^{4}}$',
            'gamnn':r'$r_{\rm 95,Ni^{56}}$',
            "Ni Vel Rat":r'$r_{\rm 95,Ni^{56}}$',
            'gamnop':r'$r_{\rm 95,bulk}$',
            "Op Vel Rat":r'$r_{\rm 95,bulk}$',
            'he_vel_rat':r'$r_{\rm 95,He^{4}}$',
            'He Vel Rat':r'$r_{\rm 95,He^{4}}$',
            'ni_vel_rat':r'$r_{\rm 95,Ni^{56}}$',
            'op_vel_rat':r'$r_{\rm 95,bulk}$',
            'mass_vel':r'$r_{\rm 95, mass}$',
            'Mass Vel':r'$r_{\rm 95, mass}$',
            'sigma_28':r'$\sigma_{\rm Ni^{56}}$'
           }
    if var not in keys.keys():
        raise ValueError(f"Error: Var not found in keys to get latex variable {var}")
        
    return keys[var]


def normalize_val(val, var):
    keys = {'D': {'Description':"Vel Latent Variable", "Unit":'', "const":1}, 
            'R_2': {'Description':"Helium Latent Variable", "Unit":'', "const":1}, 
            'R_28': {'Description':"Nickel Latent Variable", "Unit":'', "const":1},
            'R_opacity': {'Description':"Bulk Latent Variable", "Unit":'', "const":1},
            'max_vel': {'Description':"Delta Velocity", "Unit":'km/s', "const":1.0E5},
            'min_vel': {'Description':"Minimum Velocity", "Unit":'km/s', "const":1.0E5},
            'total_2': {'Description':"Total Helium Mass", "Unit":r'$M_{\rm \odot}$', "const":msun_grams},
            'He Mass': {'Description':"Total Helium Mass", "Unit":r'$M_{\rm \odot}$', "const":msun_grams},
            'total_28': {'Description':"Total Nickel Mass", "Unit":r'$M_{\rm \odot}$', "const":msun_grams},
            'Ni Mass': {'Description':"Total Nickel Mass", "Unit":r'$M_{\rm \odot}$', "const":msun_grams},
            'total_opacity': {'Description':"Total Bulk Mass", "Unit":r'$M_{\rm \odot}$', "const":msun_grams},
            'Op Mass':{'Description':"Total Bulk Mass", "Unit":r'$M_{\rm \odot}$', "const":msun_grams},
            'total_ejecta': {'Description':"Total Ejecta Mass", "Unit":r'$M_{\rm \odot}$', "const":msun_grams},
            'Ej Mass':{'Description':"Total Ejecta Mass", "Unit":r'$M_{\rm \odot}$', "const":msun_grams},
            'ejecta Mass':{'Description':"Total Ejecta Mass", "Unit":r'$M_{\rm \odot}$', "const":msun_grams},
            'vel_10p': {'Description':"10 Percentile Velocity", "Unit":r'km/s', "const":1.0E5},
            'velte': {'Description':"10 Percentile Velocity", "Unit":r'km/s', "const":1.0E5},
            'vel_25p': {'Description':"25 Percentile Velocity", "Unit":r'km/s', "const":1.0E5},
            'velt': {'Description':"25 Percentile Velocity", "Unit":r'km/s', "const":1.0E5},
            'vel_50p': {'Description':"Median Velocity", "Unit":r'km/s', "const":1.0E5},
            'velf': {'Description':"Median Velocity", "Unit":r'km/s', "const":1.0E5},
            'vel_75p': {'Description':"75 Percentile Velocity", "Unit":r'km/s', "const":1.0E5},
            'vels': {'Description':"75 Percentile Velocity", "Unit":r'km/s', "const":1.0E5},
            'vel_90p': {'Description':"90 Percentile Velocity", "Unit":r'km/s', "const":1.0E5},
            'veln': {'Description':"90 Percentile Velocity", "Unit":r'km/s', "const":1.0E5},
            'Vel_10p': {'Description':"10 Percentile Velocity", "Unit":r'km/s', "const":1.0E5},
            'Velte': {'Description':"10 Percentile Velocity", "Unit":r'km/s', "const":1.0E5},
            'Vel_25p': {'Description':"25 Percentile Velocity", "Unit":r'km/s', "const":1.0E5},
            'Velt': {'Description':"25 Percentile Velocity", "Unit":r'km/s', "const":1.0E5},
            'Vel_50p': {'Description':"Median Velocity", "Unit":r'km/s', "const":1.0E5},
            'Velf': {'Description':"Median Velocity", "Unit":r'km/s', "const":1.0E5},
            'Vel_75p': {'Description':"75 Percentile Velocity", "Unit":r'km/s', "const":1.0E5},
            'Vels': {'Description':"75 Percentile Velocity", "Unit":r'km/s', "const":1.0E5},
            'Vel_90p': {'Description':"90 Percentile Velocity", "Unit":r'km/s', "const":1.0E5},
            'Veln': {'Description':"90 Percentile Velocity", "Unit":r'km/s', "const":1.0E5},
            'gamnh': {'Description':"Ratio Velocity Containing 0.95 Helium Mass", "Unit":'', "const":1},
            'gamnn': {'Description':"Ratio Velocity Containing 0.95 Nickel Mass", "Unit":'', "const":1},
            "Ni Vel Rat": {'Description':"Ratio Velocity Containing 0.95 Nickel Mass", "Unit":'', "const":1},
            'gamnop': {'Description':"Ratio Velocity Containing 0.95 Bulk Mass", "Unit":'', "const":1},
            "Op Vel Rat": {'Description':"Ratio Velocity Containing 0.95 Bulk Mass", "Unit":'', "const":1},
            'mass_vel': {'Description':"Ratio Velocity Containing 0.95 Mass", "Unit":'', "const":1},
            'Mass Vel': {'Description':"Ratio Velocity Containing 0.95 Mass", "Unit":'', "const":1},
            'he_vel_rat': {'Description':"Ratio Velocity Containing 0.95 Helium Mass", "Unit":'', "const":1},
            'He Vel Rat': {'Description':"Ratio Velocity Containing 0.95 Helium Mass", "Unit":'', "const":1},
            'ni_vel_rat': {'Description':"Ratio Velocity Containing 0.95 Nickel Mass", "Unit":'', "const":1},
            'op_vel_rat': {'Description':"Ratio Velocity Containing 0.95 Bulk Mass", "Unit":'', "const":1},
            "Op Vel Rat": {'Description':"Ratio Velocity Containing 0.95 Bulk Mass", "Unit":'', "const":1},
            'sigma_28': {'Description':"Nickel Sigma", "Unit":r'km/s', "const":1.0E5}
           }
    if var not in keys.keys():
        raise ValueError(f"Error: Var not found in keys to normalize variable {var}")
        
        
    latex_var = get_latex_var(var)
    tor = keys[var]
    tor['latex_var'] = latex_var
    
    val_tor = val/keys[var]['const']
    
        
    return val_tor, tor

    
def adjust_lc_peak_time(time, lc):
    time = np.asarray(time)
    lc = np.asarray(lc)
    
    peak_time_idx = find_nearest(lc, np.max(lc))
    peak_time = time[peak_time_idx]
    tor_time = time-peak_time
    
    return tor_time
    
def linear_interp_colors(rgb1, rgb2, N):
    hsv1 = np.asarray(colorsys.rgb_to_hsv(r = rgb1[0], g = rgb1[1], b = rgb1[2]))
    hsv2 = np.asarray(colorsys.rgb_to_hsv(r = rgb2[0], g = rgb2[1], b = rgb2[2]))
    
    m = (hsv2-hsv1)/N
    hsv_colors = np.asarray([hsv1 + m*i for i in range(N)])
    tor = []
    for hsv_color in hsv_colors:
        h,s,v = hsv_color[0],hsv_color[1],hsv_color[2]
        tor.append(colorsys.hsv_to_rgb(h,s,v))
    return tor

'''
def get_moment1(ML, rad, moment_idx):
    
    num = 50000
    new_rad = np.linspace(rad[0], rad[-1], num=num)
    
    interpolator = interp1d(rad, ML, kind='linear')
    new_ML = interpolator(new_rad)
    x = np.linspace(0, 1, num)
    total_mass = np.sum(new_ML)
    mom = (np.sum([new_ML[i]/total_mass * (x[i])**moment_idx for i in range(len(new_ML))]))**(1/moment_idx)
    
    return mom

def get_moment(ML, rad, idx = 1):
    x = rad
    y = ML
    
    xp = [0]
    for i in x:
        xp.append(i)
        
    if idx == 1:
        int1 = np.sum([np.pi * y[i] * (xp[i+1]**4.0-xp[i]**4.0) for i in range(len(y))])
    if idx == 2:
        int1 = np.sum([4/5 * np.pi * y[i] * (xp[i+1]**5.0-xp[i]**5.0) for i in range(len(y))])
        
    int2 = np.sum([4*np.pi/3 * y[i] * (xp[i+1]**3.0-xp[i]**3.0) for i in range(len(y))])
    
    mom = int1/int2
    if idx == 2:
        mom = mom**0.5
    
    return (mom-rad[0])/(rad[-1] - rad[0])
    
'''

def get_model(file_loc, model):
    model.load_state_dict(torch.load(file_loc, map_location = torch.device("cpu")))
    model.eval()
    return model
    
    
def remove_drops(dist):
    i = 0
    while i < len(dist) - 1:
        if dist[i+1] < dist[i]:
            j = i + 1
            while j < len(dist) and dist[j] < dist[i]:
                j += 1
            
            if j < len(dist):
                step_count = j-i
                for k in range(1, step_count):
                    dist[i+k] = dist[i] + (dist[j] - dist[i]) * (k/step_count)
            i = j
        else:
            i += 1
    return dist

def check_drops(dist):
    for i in range(1, len(dist)):
        if dist[i] < dist[i-1]:
            print("at i = "+str(i)+" dist is "+str(dist[i])+", whereas it is "+str(dist[i-1])+" before it")

            
def get_vel_profile(D, max_vel, min_vel):

    #load Dash
    dash3_model = get_model(file_loc = "/n/holylabs/LABS/avillar_lab/Users/kyadavalli/OperationZ/Building Dash on GPU/Production Version of Dash/dash3.pt", model = DashAutoencoder())
    
    #get velocity from dash
    vel = remove_drops(get_profile(latent_var = D, max_val = max_vel, model = dash3_model, min_val = min_vel))
    check_drops(vel)
    
    print('return vel profile with length: '+str(len(vel)))
    return vel

def get_indices(lst, targets):
    indices = []
    for target in targets:
        if target in lst:
            indices.append(lst.index(target))
    tor = np.asarray(indices)
    return tor


def get_profile(latent_var,  model, total_mass = np.nan, max_val = np.nan, min_val = np.nan):
    dist = model.decoder(torch.from_numpy(np.asarray([latent_var], dtype = float))).detach().numpy()
    if len(dist[np.where(dist<0)]) > 0:
        pos_vals = dist[np.where(dist>0)]
        min_pos_val = min(pos_vals)
        dist[np.where(dist<0)] = min_pos_val
        
    
    if not np.isnan(total_mass) and (np.isnan(max_val) and np.isnan(min_val)):
        #use the total mass to normalize the distribution
        #this should be true for O, Cr, Ni, and opacity
        dist_sum = np.sum(dist)
        dist *= total_mass/dist_sum
    elif np.isnan(total_mass) and (not np.isnan(max_val) and not np.isnan(min_val)):
        #use min_val and max_val to normalize the distribution
        #this should only be true for the velocity distribution
        dist *= max_val
        dist += min_val
    else:
        raise ValueError('This combination to get a profile ('+str(model)+') makes no sense\nTotal_mass: '+str(total_mass)+" | max_val: "+str(max_val)+" | min_val: "+str(min_val))
    

    return dist

def get_temp_profile():
    woosley_models = "/n/holylabs/LABS/avillar_lab/Users/kyadavalli/Running Stan Original Mixed Models/running models/"
    orig_rad, orig_vel, temp, orig_density, abundances = get_cols_from_file(woosley_models+"he6.00v/mod.mod")
    return interpolate_dist(temp, 256)

def get_element_profile(R, element, total_mass = np.nan, max_val = np.nan, min_val = np.nan):
    if element == '2_4':
        element_model = get_model(file_loc = "/n/holylabs/LABS/avillar_lab/Users/kyadavalli/aAOperation/Building Riem/riems/riem_"+element.replace("_", ".")+".pt", model = Autoencoder_2_4())
    elif element == '28_56':
        element_model = get_model(file_loc = "/n/holylabs/LABS/avillar_lab/Users/kyadavalli/aAOperation/Building Riem/riems/riem_"+element.replace("_", ".")+".pt", model = Autoencoder_28_56())
    elif element == 'opacity':
        element_model = get_model(file_loc = "/n/holylabs/LABS/avillar_lab/Users/kyadavalli/aAOperation/Building Riem/riems/riem_opacities.pt", model = OpacityAutoencoder())
    

    element_profile = get_profile(latent_var = R, total_mass = total_mass, max_val = max_val, model = element_model, min_val = min_val)
    
    #print("For element: "+str(element)+" got a profile with mass "+str(np.sum(element_profile)))
    return element_profile



def get_density_abundances(vel, mass_dict, time = time_0):
    rad = vel * time

    encoded_density = []
    encoded_abundances = {}
    #for i in range(len(mass_dict['28.56'])):
    for i in range(len(rad)):
        layer_mass = np.sum([mass_dict[j][i] for j in mass_dict.keys()])
        if i == 0:
            rad_in = 0
        else:
            rad_in = rad[i-1]
            
        rad_out = rad[i]
        
        vol = 4*np.pi/3 * (rad_out**3.0 - rad_in**3.0)
        dens = layer_mass/vol
        
        if vol == 0:
            #print("Volume is zero, layer velocity: "+str(vel[i]))
            pass
        if dens < 0:
            print("Density is negative, layer_mass: "+str(layer_mass)+" layer vol: "+str(vol))
            print("Rad out: "+str(rad_out)+" rad in: "+str(rad_in)+"\n")
        if np.isinf(dens):
            print("Density is infinity, , layer_mass: "+str(layer_mass)+" layer vol: "+str(vol))
            print("Rad out: "+str(rad_out)+" rad in: "+str(rad_in)+"\n")
            
        encoded_density.append(dens)
        for element in mass_dict.keys():
            if element not in encoded_abundances.keys():
                encoded_abundances[element] = []
            abu = mass_dict[element][i]/layer_mass
            encoded_abundances[element].append(abu)
    
    encoded_density = np.asarray(encoded_density)
    return encoded_density, encoded_abundances


def random_draw(prior):
    if prior["log"]:
        return 10.0** (np.random.uniform(low = np.log10(prior['min']), high = np.log10(prior['max']) ))
    else:
        return np.random.uniform(low = prior['min'], high = prior['max'])

def sample_priors(priors, num_samples = 1):
    tor = {}
    dimensions = list(priors.keys())
    for dimension in dimensions:
        if dimension not in tor.keys():
            tor[dimension] = np.zeros(num_samples)
        for sample in range(num_samples):
            tor[dimension][sample] = random_draw(priors[dimension])
    return tor

def write_samples(samples, save_loc = "samples.txt"):
    df = pd.DataFrame(samples)
    

    num_new_samples = len(df)    
    samples_idx = np.arange(num_new_samples)
    

    df.to_csv(save_loc, index=False)
    return samples_idx
    
      
    
        

def interpolate_dist(dist, num_points):
    x = np.arange(num_points)
    x_orig = np.arange(len(dist))
    xp = x_orig * ((len(x)/len(x_orig)))
    dist_interp = np.interp(x, xp, dist)
    
    return dist_interp

def load_vels(required_length = 1024):
    woosley_models = '/n/holylabs/LABS/avillar_lab/Users/kyadavalli/Running Stan Original Mixed Models/modfiles/'
    files = list(os.listdir(woosley_models))
    vels = []


    for file in files:
        if ".mod" in file:
            _, vel, _, _, _ = get_cols_from_file(woosley_models+file)
            vel = np.asarray(vel)
            x = np.arange(required_length)
            x_orig = np.arange(len(vel))
            xp = x_orig * ((len(x)/len(x_orig)))
            vel_interp = np.interp(x, xp, vel) 
            vel_interp -= np.min(vel_interp)
            vel_interp /= np.max(vel_interp)
            vels.append(vel_interp)
    return np.asarray(vels)

def get_isotope_idx(line3, data_from_model, isotope):
        splitted = line3.split()
        i = splitted.index(isotope)
        return i

def get_abudances(line3, data_from_model, isotope = "28.56"):
    i = get_isotope_idx(line3, data_from_model, isotope = isotope)
    return np.asarray([float(i) for i in data_from_model[3+i]])

def get_all_abundances(line3, data_from_model):
    splitted = line3.split()
    all_abundances = {}
    for i in range(len(splitted)):
        all_abundances[splitted[i]] = np.asarray([float(i) for i in data_from_model[3+i]])
    return all_abundances

def get_mass(rad, density, store_dir = '', plots = False, abundances = None, verbose = False):
    if abundances is None:
        abundances = np.ones(len(rad))
    mass = np.zeros(len(rad))
    for i in range(len(rad)):
        if i == 0:
            rad_in = 0
        else:
            rad_in = rad[i-1]
            
        rad_out = rad[i]
        vol = 4*np.pi/3 * (rad_out**3.0 - rad_in**3.0)
        if vol < 0:
            print("Volume is negative, "+str(vol))
        mass[i] = density[i] * vol * abundances[i]
    if plots:
        plot_masses(rad, mass, store_dir)
    total_mass = np.sum(mass)
    if verbose:
        print("Called get mass")
        print("Found total mass : "+str(total_mass))
    return total_mass

def get_volume_list_from_file(file):
    rad, vel, temp, density, abundances = get_cols_from_file(file = file)
    vol_list = []
    for i in range(len(rad)):
        if i == 0:
            rad_in = 0
        else:
            rad_in = rad[i-1]
        rad_out = rad[i]
        vol = 4*np.pi/3 * (rad_out**3.0 - rad_in**3.0)
        vol_list.append(vol)
    return np.asarray(vol_list)

def get_rad_list_from_file(file):
    rad, vel, temp, density, abundances = get_cols_from_file(file = file)
    return rad

def get_mass_list(rad, density, abundances = None):
    if abundances is None:
        abundances = np.ones(len(rad))
    mass = np.zeros(len(rad))
    for i in range(len(rad)):
        
        if i == 0:
            rad_in = 0
        else:
            rad_in = rad[i-1]
            
        rad_out = rad[i]
        vol = 4*np.pi/3 * (rad_out**3.0 - rad_in**3.0)
        if vol < 0:
            print("volume is negative: "+str(vol))
        mass[i] = density[i] * vol * abundances[i]
    return mass



def get_data_from_file(file):
    model_data = read_file(file, skip_lines= [0,1,2])
    data = model_data['dict']
    return data

def get_elements_from_file(file):
    model_data = read_file(file, skip_lines= [0,1,2])
    skipped_lines = model_data['skipped_lines']
    element_list = skipped_lines[2].split(" ")
    tor = []
    for element in element_list:
        if "\n" in element:
            element = element[:5]
        if len(element) > 0:
            tor.append(element)

    return tor

def get_header_from_file(file):
    model_data = read_file(file, skip_lines= [0,1,2])
    skipped_lines = model_data['skipped_lines']
    nx, rmin, texp, nelems = skipped_lines[1].split()
    nx = int(nx)
    rmin = float(rmin)
    texp = float(texp)
    nelemens = int(nelems)

    return {"nx":nx, "rmin":rmin, "texp":texp, 
    "nelemens":nelemens, "skipped_lines":skipped_lines}

def get_time_of_mod(file):
    header_dict = get_header_from_file(file)
    texp = header_dict['texp']
    return texp

def get_cols_from_file(file):
    data = get_data_from_file(file = file)
    header_dict = get_header_from_file(file)
    nx = header_dict['nx']
    rmin = header_dict['rmin']
    texp = header_dict['texp']
    nelemens = header_dict['nelemens']


    temp = np.asarray([float(i) for i in data[2]])
    vel = np.asarray([float(i) for i in data[0]]) #velocity in cm/s
    rad = vel*texp #rad in cm
    density = np.asarray([float(i) for i in data[1]])
    abundances = get_all_abundances(line3 = header_dict['skipped_lines'][2], data_from_model = data)

    return rad, vel, temp, density, abundances


def find_vel_containing_mass(mass_list, vel_list, frac = 0.95):
    cum_mass = np.cumsum(np.asarray(mass_list))
    to_find_mass = frac*cum_mass[-1]
    outer_vel = vel_list[-1]
    old_mass = 0
    for i in range(len(cum_mass)):
        if cum_mass[i] > to_find_mass and old_mass < to_find_mass:
            return vel_list[i]/outer_vel
        old_mass = cum_mass[i]
    return np.nan


def find_vel_containing_mass_from_file(file, element = None, frac = 0.95):
    
    mass_list = get_mass_list_from_file(file, element)
    vel_list = get_velocity_list_from_file(file)
    
    return find_vel_containing_mass(mass_list, vel_list, frac = 0.95)
        
    
def get_dict_len(dic):
    for key in dic.keys():
        print("key: "+str(key)+" | length: "+str(len(dic[key])))

def get_mass_list_from_file(file, element = None):
    rad, vel, temp, density, abundances = get_cols_from_file(file = file)
    
    #print("get_mass_list_from_file, element: "+str(element)+" | length of abundances: "+str(get_dict_len(abundances)))
    if element is None:
        return get_mass_list(rad, density)
    elif element == 'opacity':
        return get_opacity_mass_list_from_file(file)
    else:
        return get_mass_list(rad, density, abundances = abundances[element])
    
    
    
def get_density_list_from_file(file, element = None):
    rad, vel, temp, density, abundances = get_cols_from_file(file = file)
    
    if element is None:
        return density
    else:
        tor = np.asarray([density[i] * abundances[element][i] for i in range(len(density))])
        
    return tor

def get_abundances_from_file(file, element = None):
    _, _, _, _, abundances = get_cols_from_file(file = file)
    
    if element is None:
        return abundances
    return abundances[element]
def get_p_vel_from_file(file, percentile = 50):
    vel_list = get_velocity_list_from_file(file)
    

    return np.percentile(vel_list, percentile)
    

def get_opacity_density_list_from_file(file):
    rad, vel, temp, density, abundances = get_cols_from_file(file = file)
    
    to_add = np.asarray(get_mass_list(rad, density, abundances['1.1']))
    tor = np.zeros(len(to_add))
    for element in opacity_elements:
        to_add = np.asarray([density[i] * abundances[element][i] for i in range(len(density))])
        tor += to_add
        
    return tor
    
    
def get_opacity_mass_list_from_file(file):
    rad, _, _, density, abundances = get_cols_from_file(file = file)
    to_add = np.asarray(get_mass_list(rad, density, abundances['1.1']))
    tor = np.zeros(len(to_add))
    for opacity_element in opacity_elements:
        if opacity_element in abundances.keys():
            to_add = np.asarray(get_mass_list(rad, density, abundances[opacity_element]))
            tor += to_add
        
    return tor
    
def get_mass_dict_from_file(file):
    element_list = get_elements_from_file(file)
    tor = {}
    for element in element_list:
        tor[element] =  get_mass_list_from_file(file, element = element)
    return tor


def get_vel_list_from_file(file):
    return get_velocity_list_from_file(file)

def get_velocity_list_from_file(file):
    _, vel, _, _, _ = get_cols_from_file(file = file)
    return vel

def get_rad_list_from_file(file):
    rad, _, _, _, _ = get_cols_from_file(file = file)
    return rad

def get_temp_list_from_file(file):
    _, _, temp, _, _ = get_cols_from_file(file = file)
    return temp


def mass_CDF(mass_list):
    mass_CDF_tor = mass_list.copy()
    for i in range(len(mass_list)):
        mass_CDF_tor[i] = np.sum(mass_list[:i])
    return mass_CDF_tor


def plot_densities(all_rads, all_densities, labels, colors, saveloc = 'Densities.pdf', plot_legends = False):
    plt, _, _ = get_pretty_plot()
    col_idx = 0

    for i in range(len(all_rads)):

        label = labels[i]
        linestyle = '-'
        if "Unmixed" in label:
            linestyle = ':'
        if len(colors) > 0:
            plt.plot(all_rads[i], all_densities[i], linewidth = 4, color = colors[col_idx], label = labels[i], linestyle = linestyle)
        else:
            plt.plot(all_rads[i], all_densities[i], linewidth = 4, label = labels[i], linestyle = linestyle)

        if 'shoulder' in label.lower():
            col_idx += 1

    if plot_legends:	
        plt.legend(fontsize = 15, ncol = 2)
    plt.yscale('log')
    plt.xscale('log')
    #plt.ylim([np.min(all_densities)/2, np.max(all_densities)*5])
    plt.xlabel("Radius (cm)", fontsize = 50)
    plt.ylabel(r"Density ($g~cm^{-3}$)", fontsize = 50)
    plt.savefig(saveloc, bbox_inches = 'tight')
    plt.close()
    
    
def extract_digits(filename):
    # Use regex to find all digits in the filename
    digits = re.findall(r'\d+', filename)
    # Join the digits and convert to an integer
    return int(''.join(digits)) if digits else None
    
def columns_in_file(file_path, required_columns):
    try:
        with open(file_path, 'r') as file:
            header_line = file.readline().strip()
            
            column_names = [name.strip('#').replace(' ', '') for name in header_line.split('\t')]
            
            # Check if all required columns are present
            missing_columns = [col for col in required_columns if col not in column_names]
        
        if not missing_columns:
            return True
        else:
            return False
    
    except Exception as e:
        print("Exception e: "+str(e))
        return False

def read_file_with_hash_columns(file_path):
    try:
        # Read the entire file into a DataFrame, treating the first row as the header
        with open(file_path, 'r') as file:
            # Read the first line to extract column names
            header_line = file.readline().strip()
            column_names = [name.strip('#').replace(' ', '') for name in header_line.split('\t')]
            
            # Read the rest of the data into a DataFrame
            df = pd.read_csv(file, delim_whitespace=True, header=None, comment = '#')
            
            # Set the new column names
            df.columns = column_names

        return df

    except Exception as e:
        print("An error occurred while reading the file:", e)
        return None
    
def get_lc_from_file(file_path, filt = 'Lbol(erg/s)'):
    lc_df = read_file_with_hash_columns(file_path)
    if filt not in lc_df:
        return None, None
    lc = np.asarray(lc_df[filt], dtype='float')
    times = np.asarray(lc_df['Time(Days)'], dtype='float')
    
    return lc, times
    

elements = ['1.1', '2.4', '6.12', '7.14', '8.16', '10.20', '12.24', '14.28', '16.32', '18.36', '20.40', '22.44', '24.48', '26.52', '26.56', '27.56', '28.56']
opacity_elements = ['1.1', '6.12', '7.14', '8.16', '10.20', '12.24', '14.28', '16.32', '18.36', '20.40', '22.44', '24.48', '26.52', '26.56', '27.56']
functional_elements = ['2.4', '28.56']
opacity_elements_idxs = get_indices(elements, opacity_elements)

space_colors = get_colors(6, 'space_person')




def get_samples(all_lcs):
    samples = []
    for lc_file in all_lcs:
        sample = lc_file.split("/")[-2]
        samples.append(sample)
    return samples

def get_all_lcs(all_lcs):
    
    all_lc = []
    all_times = []
    
    for lc_file in all_lcs:
        try:
            lc, times = get_lc_from_file(lc_file, 'Lbol(erg/s)')
            if lc is not None:
                all_lc.append(lc)
                all_times.append(times)
        except:
            print("Tried and failed to read lightcurve file: "+str(lc_file))
        
    return all_lc, all_times
    

def plot_lcs(all_lcs, batch_size, out_dir, vlines = None, labels = None, ylim = [None, None]):
    mkdir(out_dir)
    samples = get_samples(all_lcs)
    all_lc, all_times = get_all_lcs(all_lcs)
    plt, _, _ = get_pretty_plot()
    plotted = []
    
    num_plotted = 0
    colors = get_colors(batch_size, "space_person")
    max_lum = -np.inf
    min_lum = np.inf
    for i, lc_file in enumerate(all_lc):
        sample = samples[i]

        lc = all_lc[i]
        times = all_times[i]
        
        peak_idx = find_nearest(lc, np.max(lc))
        times -= times[peak_idx]
        plotted.append(sample)


        idx = find_nearest(times, 70)
        m = lc[idx]
        if m < min_lum:
            min_lum = m
        if np.max(lc) > max_lum:
            max_lum = np.max(lc)

        if labels is None:
            label = sample
        else:
            label = labels[i]
            
        plt.plot(times, lc, label = label, color = colors[num_plotted %batch_size], linewidth = 3, alpha = 0.7)
        num_plotted += 1

        if (num_plotted%batch_size == 0 and num_plotted > 0) or i == len(all_lc) - 1:
            plt.xlabel("Time (d)", fontsize = 35)
            plt.ylabel("Bol Lc (erg/s)", fontsize = 35)
            if ylim[0] is None:
                ylim[0] = 0.1*min_lum
            if ylim[1] is None:
                ylim[1] = 3*max_lum

            plt.ylim(ylim)
            plt.yscale("log")

            plt.legend(fontsize = 25)
            if vlines is not None:
                plt.vlines(vlines, ylim[0], ylim[1], linewidth = 3, color = 'black', linestyle = '--')
            min_lum = np.inf
            max_lum = -np.inf
            plt.savefig(out_dir+"/"+str(int(num_plotted/batch_size))+".pdf", bbox_inches = 'tight')
            plt.close()
            plt, _, _ = get_pretty_plot()
            
    return plotted


def find_largest_int_file(files):
    ints = np.asarray([extract_digits(i) for i in files])
    max_int = np.max(ints)
    idx = find_nearest(ints, max_int)
    tor = files[idx]
    return tor

#find samples currently running or pending
def decode_jobs(command, prefix = ''):
    result = subprocess.check_output(command).decode('utf-8')
    jobs = str(result).replace(" ", '').replace('\t', '').split('\n')
    samples = []
    for i in jobs:
        if len(prefix) > 0 and prefix not in i:
            continue
            
        g = extract_digits(i)
        if g is not None:
            samples.append(str(g))
    
    return samples

def get_jobs_running(prefix = ''):
    return decode_jobs(['squeue', '-u', 'kyadavalli', '-h', '-t', 'running', '-r', '-O', 'name'], prefix = prefix)

def get_jobs_pending(prefix = ''):
    return decode_jobs(['squeue', '-u', 'kyadavalli', '-h', '-t', 'pending', '-r', '-O', 'name'], prefix = prefix)

def plot_lc(lc_file, out_dir, sample = '', outdir_suffix = '', filters = ['SDSS_u', 'SDSS_g', 'SDSS_r', 'SDSS_i', 'SDSS_z'], colors = [sKy_colors['blue'], sKy_colors['honest green'], sKy_colors['orange'], sKy_colors['red'], sKy_colors['dull brown']], title = ''):
    
    
    plt, _, _ = get_pretty_plot()
    lc, times = get_lc_from_file(lc_file)
    plt.plot(times, lc, linewidth = 3, color = 'black', label = 'Bol Lc')
    plt.xlabel("Time (d)", fontsize = 35)
    plt.ylabel("Bolometric LC "+str(sample), fontsize = 35)
    plt.yscale('log')
    if len(title)> 0:
        plt.title(title, fontsize = 35)

    plt.ylim([0.001*np.max(lc), np.max(lc)*2])
    plt.xlim([-3, 70])
    plt.legend(fontsize = 20, ncol = 2)
    plt.savefig(out_dir+"_"+sample+"_"+"bol_lc.pdf", bbox_inches = 'tight')
    plt.close()
    

    hz = u.second**-1
    jansky = 1.0E-23 * u.erg/(u.second*(u.cm)**2*hz)



    #plotting light curve
    filters = ['SDSS_u', 'SDSS_g', 'SDSS_r', 'SDSS_i', 'SDSS_z']
    colors = [sKy_colors['blue'], sKy_colors['honest green'], sKy_colors['orange'], sKy_colors['red'], sKy_colors['dull brown']]

    plt, _, _ = get_pretty_plot()
    max_val = -np.inf
    min_max = np.inf
    for filt in filters:
        color = colors[filters.index(filt)]
        lightcurve_4d, times_4d = get_lc_from_file(lc_file, filt = filt)
        if 'SDSS' in filt:
            lightcurve_4d = (10**(-1.0*lightcurve_4d/2.5) * 3631*jansky * 4*np.pi*(10*u.pc)**2).decompose().to(u.erg/(u.second*hz)).value


        if np.max(lightcurve_4d) > max_val:
            max_val = np.max(lightcurve_4d)
        if np.max(lightcurve_4d) < min_max:
            min_max = np.max(lightcurve_4d)

        plt.plot(times_4d, lightcurve_4d, linewidth = 3, color = color, alpha = 0.6)
        plt.plot(np.nan, np.nan, linewidth = 6, label = str(filt), color = color)

    plt.xlabel("Time (d)", fontsize = 35)
    plt.ylabel("LC ("+str(sample)+")", fontsize = 35)
    plt.yscale('log')
    if len(title)> 0:
        plt.title(title, fontsize = 35)
    ul = 2*max_val
    ll = 0.01*min_max
    plt.ylim([ll, ul])
    plt.xlim([-3, 70])
    plt.legend(fontsize = 20, ncol = 2)
    plt.savefig(out_dir+"_"+sample+"_"+"lc.pdf", bbox_inches = 'tight')
    plt.close()


def plot_lc_model(lc_file, mod_file, out_dir, sample = '', outdir_suffix = '', filters = ['SDSS_u', 'SDSS_g', 'SDSS_r', 'SDSS_i', 'SDSS_z'], colors = [sKy_colors['blue'], sKy_colors['honest green'], sKy_colors['orange'], sKy_colors['red'], sKy_colors['dull brown']], title = ''):
    

    hz = u.second**-1
    jansky = 1.0E-23 * u.erg/(u.second*(u.cm)**2*hz)



    #plotting light curve
    filters = ['SDSS_u', 'SDSS_g', 'SDSS_r', 'SDSS_i', 'SDSS_z']
    colors = [sKy_colors['blue'], sKy_colors['honest green'], sKy_colors['orange'], sKy_colors['red'], sKy_colors['dull brown']]

    plt, _, _ = get_pretty_plot()
    max_val = -np.inf
    min_max = np.inf
    for filt in filters:
        color = colors[filters.index(filt)]
        lightcurve_4d, times_4d = get_lc_from_file(lc_file, filt = filt)
        if 'SDSS' in filt:
            lightcurve_4d = (10**(-1.0*lightcurve_4d/2.5) * 3631*jansky * 4*np.pi*(10*u.pc)**2).decompose().to(u.erg/(u.second*hz)).value


        if np.max(lightcurve_4d) > max_val:
            max_val = np.max(lightcurve_4d)
        if np.max(lightcurve_4d) < min_max:
            min_max = np.max(lightcurve_4d)

        plt.plot(times_4d, lightcurve_4d, linewidth = 3, color = color, alpha = 0.6)
        plt.plot(np.nan, np.nan, linewidth = 6, label = str(filt), color = color)

    plt.xlabel("Time (d)", fontsize = 35)
    plt.ylabel("LC ("+str(sample)+")", fontsize = 35)
    plt.yscale('log')
    if len(title)> 0:
        plt.title(title, fontsize = 35)
    ul = 2*max_val
    ll = 0.01*min_max
    plt.ylim([ll, ul])
    plt.xlim([-3, 70])
    plt.legend(fontsize = 20, ncol = 2)
    plt.savefig(out_dir+"_"+sample+"_"+"lc.pdf", bbox_inches = 'tight')
    plt.close()




    #plotting nickel mass dist and velocity on same plot
    nickel_mass_list = get_mass_list_from_file(mod_file, element = '28.56')/msun_grams
    he_mass_list = get_mass_list_from_file(mod_file, element = '2.4')/msun_grams
    opacity_mass_list = get_opacity_mass_list_from_file(mod_file)/msun_grams
    ul = 2*np.max([nickel_mass_list, he_mass_list, opacity_mass_list])
    ll = 0.01*np.min([np.max(nickel_mass_list), np.max(he_mass_list), np.max(opacity_mass_list)])


    velocity_list = get_velocity_list_from_file(mod_file)/1.0E5
    x = np.arange(len(nickel_mass_list))/len(nickel_mass_list)

    fig, ax1 = plt.subplots(figsize=(15, 10)) 
    ax1.set_xlabel('Depth in Ejecta', fontsize = 35) 
    plt.xticks(fontsize = 35)
    ax1.set_ylabel(r'Mass $(M_\odot)$', fontsize = 35) 
    ax1.plot(x, nickel_mass_list, color = color_schemes['chill'][0], label = r'$\rm{Ni^{56}}$', linestyle = '-', linewidth = 4) 
    ax1.plot(x, he_mass_list, color = color_schemes['chill'][1], label = r'$\rm{He^{4}}$', linestyle = '-.', linewidth = 4) 
    ax1.plot(x, opacity_mass_list, color = color_schemes['chill'][2], label = r'Opacity Mass', linestyle = '--', linewidth = 4) 
    plt.legend(fontsize = 15)
    ax1.tick_params(axis ='y', labelsize = 25) 
    ax1.set_yscale('log')
    ax1.set_ylim([ll, ul])


    ax2 = ax1.twinx() 
    ax2.set_ylabel('Velocity (km/s)', fontsize = 35) 
    ax2.plot(x, velocity_list, color = space_colors[2], linewidth = 4) 
    ax2.tick_params(axis ='y', labelcolor = space_colors[2], labelsize = 25) 
    if len(title) >0: 
        plt.title(title, fontsize = 35)
    plt.savefig(out_dir+"Mass_Vel.pdf", bbox_inches = 'tight')
    plt.close()

def get_mag_rise_time(time, mag, mag_diff = 1):
    peak_mag = np.min(mag)
    peak_time = time[find_nearest(mag, peak_mag)]
    
    idx = np.where(mag > peak_mag + mag_diff)
    high_mag_times = time[idx]
    idx1 = np.where(high_mag_times < peak_time)
    high_mag_times = high_mag_times[idx1]

    
    if len(high_mag_times) > 0:
    

        time_0 = np.max(high_mag_times)
        rise_time = peak_time - time_0
        
        return rise_time
    return np.nan

def get_mag_fall_time(time, mag, mag_diff = 1):
    peak_mag = np.min(mag)
    peak_time = time[find_nearest(mag, peak_mag)]
    
    idx = np.where(mag > peak_mag + mag_diff)
    high_mag_times = time[idx]
    idx1 = np.where(high_mag_times > peak_time)
    high_mag_times = high_mag_times[idx1]
    
    if len(high_mag_times)> 0:
    
        time_1 = np.min(high_mag_times)
        fall_time = time_1 - peak_time

        return fall_time
    
    return np.nan

    
def get_rise_time(time, lum, frac_diff_lum = 10.0**0.4):
    peak_lum = np.max(lum)
    peak_time = time[find_nearest(lum, peak_lum)]
    
    idx = np.where(lum < peak_lum/frac_diff_lum)
    low_lum_times = time[idx]
    idx1 = np.where(low_lum_times < peak_time)
    low_lum_times = low_lum_times[idx1]

    
    if len(low_lum_times) > 0:
    

        time_0 = np.max(low_lum_times)
        rise_time = peak_time - time_0
        
        return rise_time
    return np.nan

def get_fall_time(time, lum, frac_diff_lum = 10.0**0.4):
    peak_lum = np.max(lum)
    peak_time = time[find_nearest(lum, peak_lum)]
    
    idx = np.where(lum < peak_lum/frac_diff_lum)
    low_lum_times = time[idx]
    idx1 = np.where(low_lum_times > peak_time)
    low_lum_times = low_lum_times[idx1]
    
    if len(low_lum_times)> 0:
    
        time_1 = np.min(low_lum_times)
        fall_time = time_1 - peak_time

        return fall_time
    
    return np.nan

'''
def get_moment_from_file(file, element, moment_idx = 1):
    vel_list = get_velocity_list_from_file(file)
    ML = get_density_list_from_file(file, element)
    
    #return get_moment(ML, vel_list, moment_idx = moment_idx)
    return get_moment(ML, vel_list, idx = moment_idx)


def get_moments_from_file(file, moment_idx = 1):
    vel_list = get_velocity_list_from_file(file)
    ni_ML = get_density_list_from_file(file, element = '28.56')
    he_ML = get_density_list_from_file(file, element = '2.4')
    op_ML = get_opacity_density_list_from_file(file)
    ML = get_density_list_from_file(file)
    
    
    #return get_moment(he_ML, vel_list, moment_idx = moment_idx), get_moment(ni_ML, vel_list, moment_idx = moment_idx), get_moment(op_ML, vel_list, moment_idx = moment_idx)
    return get_moment(ML, vel_list, moment_idx), get_moment(he_ML, vel_list, moment_idx), get_moment(ni_ML, vel_list, moment_idx), get_moment(op_ML, vel_list, moment_idx)

'''



def get_vel_percentiles_from_file(file, percentiles = [10, 25, 50, 75, 90]):
    from scipy.stats import percentileofscore
    
    vel_list = get_velocity_list_from_file(file)
    
    
    tor = {}
    for percentile in percentiles:
        tor[percentile] = np.percentile(vel_list, percentile)
    return tor
    
    

def plot_corner(data_dict, save_loc, contours = True):
    import corner
    """
    Generate a corner plot from a dictionary of parameter samples.
    
    Parameters:
    data_dict (dict): A dictionary where keys are parameter names and values are lists or arrays of samples.
    """
    
    # Convert dictionary values to a 2D NumPy array (samples x parameters)
    param_names = list(data_dict.keys())
    samples = np.column_stack([data_dict[key] for key in param_names])
    
    # Create the corner plot
    fig = corner.corner(samples, labels=param_names, plot_contours = contours)
    
    # Save the plot
    plt.savefig(save_loc)
    plt.close()
    
def plot_pair(data_dict, save_loc, corner = True):
    import corner
    import seaborn as sns
    import itertools
    
    
    # Convert dictionary to DataFrame
    df = pd.DataFrame(data_dict)
    
    # Create a list of variable names (keys of the dictionary)
    variables = list(data_dict.keys())
    
    # Define combinations of 2 variables (combinations are unordered, hence we use itertools.combinations)
    variable_combinations = list(itertools.combinations(variables, 2))

    # Create a pairplot with scatter plots for combinations and histograms on the diagonal
    sns.pairplot(df, vars=variables, corner = corner)
    
    cols_to_use = ['foo', 'bar', 'hue']
    lims_by_col = {"D":[0, 1], "R_2":[0, 1],  "R_28":[0, 1], "R_opacity":[0, 1],  "min_vel":[1.5E7, 3.0E8], "max_vel":[8.0E8, 3.5E9], "total_2":[5.0E31, 2.0E33],  "total_28":[8.0E30, 5.0E32], "total_opacity":[1.2E31, 1.6E34]}
    #lims_by_col = {'foo':[-100, 100], 'bar':[-2, 2]}
    '''
    for ax in pl.axes.flatten():
        xlab = ax.get_xlabel()
        if len(xlab)==0: continue
        ax.set_xlim(lims_by_col[xlab])

        ylab = ax.get_ylabel()
        if len(ylab)==0: continue
        ax.set_xlim(lims_by_col[ylab])'''
    
    
    
    # Show the plot
    plt.savefig(save_loc)
    plt.close()
    
    
def get_batch_script(job_name, hours = 1):
    batch_script = """#!/bin/bash -l
#SBATCH -n 48
#SBATCH -t 0-"""+str(hours)+""":00
#SBATCH -p itc_cluster,sapphire,shared
#SBATCH --mem-per-cpu=5000
#SBATCH -o out_%j.out
#SBATCH -e err_%j.err
#SBATCH -J """+str(job_name)+"""
# load modules
module load gcc/12.2.0-fasrc01
module load mpich/4.1-fasrc01
module load gsl/2.7-fasrc01
module load hdf5/1.14.0-fasrc01
module load python/3.10.9-fasrc01
mamba activate sedona_venv



srun --mpi=pmix $SEDONA_HOME/src/sedona6.ex param_lc_lte_exp.lua &> log.log
python cleaning_spec.py
python $lcfilt -s spectrum_final.h5 -b SDSS_u,SDSS_g,SDSS_r,SDSS_i,SDSS_z,B,V,Cousins_R,Cousins_I
rm plt*
python cleaning_up.py
"""
    return batch_script


def get_param_script(start_time = 4.0, stop_time = 90, dt = 0.5, output_step = 1.0, restart = False, bound_free = False):
    
    tor = '''-- param file for running the light curve of a 1D Type Ia-like supernova
-- using LTE and line expansion opacity
-- atomic data taken from line file

grid_type    = "grid_1D_sphere"   
model_file   = "mod.mod"
hydro_module = "homologous"

sedona_home        = os.getenv('SEDONA_HOME')
defaults_file      = sedona_home.."/defaults/sedona_defaults.lua"
data_atomic_file   = sedona_home.."/data/cmfgen_levelcap100.hdf5"

-- helper variable
days = 3600.0*24

-- total number of particles used initially
particles_n_initialize       = 1e6
-- number of particles emitted per time step from radioactivity
particles_n_emit_radioactive = 2e5

-- time start/stop and stepping
tstep_time_start             = '''+str(start_time)+'''*days
tstep_time_stop              = '''+str(stop_time)+'''*days
tstep_max_dt                 = '''+str(dt)+'''*days
tstep_max_delta              = 0.1

-- frequency grid to calculate and store opacities
nu1 = 1e13
nu2 = 1e20
transport_nu_grid   = {nu1,nu2,0.0003,1}

-- frequency grid to calculate output spectrum
nu1_spec = nu1*1.1
spectrum_nu_grid   = {nu1_spec,nu2,0.002,1}
spectrum_time_grid = {'''+str(start_time)+'''*days,'''+str(stop_time)+'''*days,'''+str(output_step)+'''*days}
output_write_radiation = 0

-- opacity settings
opacity_grey_opacity         = 0.0
opacity_epsilon              = 1.0
opacity_electron_scattering  = 1
opacity_free_free            = 1
opacity_bound_bound          = 0
opacity_line_expansion       = 1
opacity_fuzz_expansion       = 0
opacity_bound_free           = '''
    if bound_free:
        tor += '0'
    else:
        tor += '1'
    
    
    tor+='''

-- transport settings
transport_steady_iterate        = 0
transport_radiative_equilibrium = 1

--timestepping
run_do_checkpoint = 1
run_chk_walltime_interval = 0.1*3600.0
run_chk_walltime_max = 4.0*3600.0
run_checkpoint_name_base = "chk"'''
    
    if restart:
        tor += '''run_do_restart = 1
run_restart_file = "chk.h5"'''
    return tor

def redshift_to_distance_modulus(redshift):
    # Calculate the distance modulus
    distance = cosmo.luminosity_distance(redshift)
    distance_modulus = 5 * np.log10(distance.to(u.pc).value) - 5
    return distance_modulus

def json_dict(file):
    with open(file) as f:
        return json.load(f)
    
    
def write_dict_to_file(dic, out_file):
    header = '# '
    tow = []
    for k in dic.keys():
        tow.append(np.asarray(dic[k]))
        header += k+" "
    header += '\n'
    
    tow = np.asarray(tow).T
    
    write_to_file(out_file, header, append = False)
    write_to_file(out_file, tow, append = True)
    
def dictionary_signature(dic):
    for k in dic.keys():
        print("dic with key: "+str(k)+" has length: "+str(len(dic[k])))
        
def get_peak_time_band(file, sedona_filt):
    sedona_lc, sedona_time = get_lc_from_file(file, filt = sedona_filt)
    
    sedona_lc = 10**(-1.0*sedona_lc/2.5)
    return sedona_time[find_nearest(sedona_lc, np.max(sedona_lc))]
        
def osc_to_ascii(file_to_conv, out_file, obj, keys_to_skip = ['source', 'u_time', 'bandset'], keys_to_pad = ['e_magnitude'], required_keys = ['band', 'time', 'magnitude'], out_dict = {'time':[], 'magnitude':[], 'e_magnitude':[], 'band':[]}, verbose = False):

    dic = json_dict(file_to_conv)[obj]
    phot = dic['photometry']
    
    for i, dp in enumerate(phot):
        keys_found = []
        append_this = True
        
        for required_key in required_keys:
            if required_key not in dp.keys():
                append_this = False
        
        if not append_this:
            continue
            
        for k in dp.keys():
            keys_found.append(k)
            if k in keys_to_skip:
                continue
            if k not in out_dict.keys():
                out_dict[k] = []
            out_dict[k].append(dp[k])
        for kk in keys_to_pad:
            if kk not in keys_found:
                out_dict[kk].append(np.nan)
    
    if verbose:
        dictionary_signature(out_dict)
    write_dict_to_file(out_dict, out_file)


def generate_run(out_dir, param_script, batch_script, sample_df, N = num_input, mod_tor = None, mod_file = None):

    if (mod_tor is None and mod_file is None):
        raise ValueError('Issue in generating run for outdir: '+str(out_dir)+'\nEither mod_tor or mod_file has to be not None')
    if (mod_tor is not None and mod_file is not None):
        raise ValueError('Issue in generating run for outdir: '+str(out_dir)+'\nmod_tor and mod_file can\'t both be not None')
    mkdir(out_dir)
    
    #copy the clean up scripts to the model dir
    rm(out_dir+"/cleaning_up.py")
    write_to_file(out_dir+"/cleaning_up.py", cleaning_up_script, append = True)
    
    rm(out_dir+"/cleaning_spec.py")
    write_to_file(out_dir+"/cleaning_spec.py", cleaning_spec_script, append = True)
    
    
    if mod_tor is not None:
    #write header to model file
        tow = """1D_sphere  SNR
"""+str(N)+"""  0. """+str(time_0)+""" 17
1.1 2.4 6.12 7.14 8.16 10.20 12.24 14.28 16.32 18.36 20.40 22.44 24.48 26.52 26.56 27.56 28.56
"""

        model_file = out_dir+"/mod.mod"
        rm(model_file)
        write_to_file(model_file, tow, append = True)
        write_to_file(model_file, mod_tor, append = True)
    else:
        cp(mod_file, out_dir)
    
    #write param file
    param_file = out_dir+"/param_lc_lte_exp.lua"
    rm(param_file)
    write_to_file(param_file, param_script, append = True)
    
    #write batch file
    batch_file = out_dir+"/run_batch.sub"
    rm(batch_file)
    write_to_file(batch_file, batch_script, append = True)
    
    #write sample.txt
    sample_file = out_dir+"/sample.txt"
    rm(sample_file)
    sample_df.to_csv(sample_file, sep = ',', index = False)
    
    
def get_mass_dict(R_2, R_28, R_opacity, total_2, total_28, total_opacity):
    
    #generate mass distributions
    mass_dict = {}
    for element in elements:
        mass_dict[element] = np.zeros(256)


    he_dist = get_element_profile(R = R_2, total_mass = total_2, element = '2_4')
    idx = np.where(he_dist < 1.0E5)
    he_dist[idx] = 0.0
    mass_dict['2.4'] = he_dist


    Ni_dist = get_element_profile(R = R_28, total_mass = total_28, element = '28_56')
    idx = np.where(Ni_dist < 1.0E5)
    Ni_dist[idx] = 0.0
    mass_dict['28.56'] = Ni_dist

    
    Op_dist = get_element_profile(R = R_opacity, total_mass = total_opacity, element = 'opacity')
    idx = np.where(Op_dist < 1.0E5)
    Op_dist[idx] = 0.0
    opacity_masses = Op_dist
    
    #get density and abundances 
    for layer_idx in range(num_input):
        opacity_mass = opacity_masses[layer_idx]
        functional_mass = 0.0

        for functional_element in functional_elements:
            functional_mass += mass_dict[functional_element][layer_idx]



        for opacity_element in opacity_elements:

            #in layer layer_idx, what fraction of the opacity mass should be this opacity element?
            element_idx = elements.index(opacity_element)


            opacity_element_frac = avg_fractions_data[layer_idx, element_idx]
            summable = avg_fractions_data[layer_idx, opacity_elements_idxs]
            opacities_layer = np.sum(summable)

            element_mass_layer = (opacity_element_frac/opacities_layer) * opacity_mass
            mass_dict[opacity_element][layer_idx] = element_mass_layer
    
    return mass_dict, opacity_masses

    
    

    
def construct_run(out_dir, D, R_2, R_28, R_opacity, min_vel, max_vel, total_2, total_28, total_opacity, start_time = 4.0, stop_time = 70, dt = 0.2, hours = 1, mass_dict = None, vel = None, job_name = 'job'):
    
    
    #generate velocity distribution
    if vel is None:
        vel = get_vel_profile(D = D, max_vel = max_vel, min_vel = min_vel)


    #generate mass distributions
    if mass_dict is None:
        mass_dict, opacity_masses = get_mass_dict(R_2, R_28, R_opacity, total_2, total_28, total_opacity)



    density, abundances = get_density_abundances(vel, mass_dict)

    #generate temperature profile
    temp = get_temp_profile()
    temp = interpolate_dist(temp, len(density))


    #create array to write to model file
    tor = [vel, density, temp]
    for idx in abundances:
        tor.append(np.asarray([float(j) for j in abundances[idx]]))

    '''   
    for jkjk in range(len(tor)):
        print("length of tor in idx ("+str(jkjk)+"): "+str(len(tor[jkjk])))'''
    tor = np.asarray(tor).T

    param_script = get_param_script(start_time = start_time, stop_time = stop_time, dt = dt, restart = False)
    batch_script = get_batch_script(job_name = job_name, hours = hours)

    #vmin, vmax, D, total_2, R2, total_28, R28, total_op, Rop


    #D R_2 R_28 R_opacity min_vel max_vel total_2 total_28 total_opacity 
    sample = {
        'D': [D],
        'R_2': [R_2],
        'R_28': [R_28],
        'R_opacity':[R_opacity],
        'min_vel':[min_vel],
        'max_vel':[max_vel],
        'total_2':[total_2],
        'total_28':[total_28],
        'total_opacity':[total_opacity]
    }
    sample_df = pd.DataFrame(sample)

    #generate mod file
    generate_run(out_dir, param_script, batch_script, sample_df, N = len(vel), mod_tor = tor)

    
    
    
def convert_tardis_to_sedona(tardis_param_file, out_file, verbose = False):
    v = verbose
    file_to_read = tardis_param_file
    
    read_in = read_file(file_to_read, separators = [','])
    lines_to_skip = 0
    for i, line in enumerate(read_in['lines']):
        if len(line) == 1 and i > 0:
            lines_to_skip = i
            
    if v:
        print("found lines_to_skip: "+str(lines_to_skip)+" for file: "+str(tardis_param_file))

    read_in = read_file(file_to_read, skip_lines = list(np.arange(lines_to_skip+1)), separators = [','], header_row = 0)

    data = read_in['dict']
    skipped_lines = read_in['skipped_lines']
    header_list = list(data.keys())
    if v:
        print("headers: "+str(header_list))
    if v:
        print("skipped lines: "+str(skipped_lines))


    #read parameters from the model file
    time_of_model_days = float(skipped_lines[2].split(' ')[1])
    if v:
        print("found time_of_model_days: "+str(time_of_model_days))
    time_of_model_seconds = str(round(time_of_model_days * 86400, 2))
    if v:
        print("found time_of_model_seconds: "+str(time_of_model_seconds))

    element_dict = {"H":1, "He":2, "C":6, "N":7, "O":8, "Ne":10, "Mg": 12, "Si": 14, "S":16, "Ar":18, "Ca":20, "Ti":22, "Cr":24, "Fe":26, "Ni":28}

    element_names = header_list[3:]
    num_elements = len(element_names)
    if v:
        print("found "+stR(num_elements)+" elements")
        print("elements: "+str(element_names))
    element_string_sedona = ''
    for element in element_names:
        atomic_name = ''
        idx = 0
        for i in element:
            if not is_int(i):
                atomic_name += i
                idx += 1

        atomic_num = element_dict[atomic_name]
        atomic_mass = element[idx:]
        if v:
            print("Found atomic name "+str(atomic_name)+" for element: "+str(element)+", with atomic number "+str(atomic_num)+" and atomic mass: "+str(atomic_mass))
        element_string_sedona += str(atomic_num)+"."+str(atomic_mass)+" "


    #create lists of variables to put into sedona model file
    vel_list = np.asarray([float(i) for i in data['velocity']])
    density_list = np.asarray([float(i) for i in data['density']])
    temp_list = np.asarray([float(i) for i in data['t_rad']])
    if v:
        print("found "+str(len(vel_list))+" velocities")
        print("found "+str(len(density_list))+" densities")
        print("found "+str(len(temp_list))+" temps")

    #now, construct the header lines for the sedona model file
    header_lines = """1D_sphere  SNR
"""+str(len(vel_list))+""" 0. """+time_of_model_seconds+""" """+str(num_elements)+"""
"""+str(element_string_sedona)+"""
"""

    if v:
        print("constructed header lines:\n"+Str(header_lines))


    #construct the rest of the model file now
    rm(out_file)
    write_to_file(out_file, header_lines, append = False)
    tow = []
    for key in data.keys():
        tow.append(np.asarray(data[key]))
    tow = np.asarray(tow).T
    write_to_file(out_file, tow, append = True)




