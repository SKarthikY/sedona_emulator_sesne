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
