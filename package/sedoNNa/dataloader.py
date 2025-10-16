import os
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from astropy import units as u
import time as TTT

from numpy.polynomial.chebyshev import chebfit

# Physical constants
c = 3.0E8 * u.meter/u.second
angstrom = 1.0E-10 * u.meter
erg = 1.0E-7 * u.kilogram*(u.meter/u.second)**2
cm = 1.0E-2 * u.meter
hz = 1/u.second
freq_flux = erg/(cm**2*u.second*hz)
wav_flux = erg/(cm**2*angstrom*u.second)
sec_per_day = 86400

class NormalizeSpectralData:
    def __init__(self, stats):
        self.stats = stats

    def __call__(self, sample):
        descriptor = torch.tensor(sample['descriptor'], dtype=torch.float32) \
            if not isinstance(sample['descriptor'], torch.Tensor) else sample['descriptor']
        time = torch.tensor(sample['time'], dtype=torch.float32) \
            if not isinstance(sample['time'], torch.Tensor) else sample['time']
        flux = torch.tensor(sample['flux'], dtype=torch.float32) \
            if not isinstance(sample['flux'], torch.Tensor) else sample['flux']
        # Normalize descriptor
        descriptor = (descriptor - self.stats['descriptor_mean']) / self.stats['descriptor_std']

        # Normalize time
        time = (time - self.stats['time_mean']) / self.stats['time_std']
        
        # normalize Fluxes
        flux = (flux - self.stats['fluxes_mean']) / self.stats['fluxes_std']

        return {
            'descriptor': descriptor,
            'time': time,
            'wav': sample['wav'],
            'flux': flux,
            'sample_id':sample['sample_id']
        }
    
class FastSupernovaDataset(torch.utils.data.Dataset):
    def __init__(self, pt_file=None, samples=None, transform=None):
        if samples is not None:
            self.samples = samples
        elif pt_file is not None:
            self.samples = torch.load(pt_file)
        else:
            raise ValueError("Must supply samples or pt_file")
        self.transform = transform
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.transform: sample = self.transform(sample)
        return sample

class SupernovaSpectralDataset(Dataset):
    def __init__(self, data_dir, N_wavelengths=602, wav_min=3000, wav_max=10000, transform=None, sample_ids=None):
        self.data_dir = data_dir
        self.transform = transform
        self.N_wavelengths = N_wavelengths
        self.wav_min = wav_min
        self.wav_max = wav_max
        self.fixed_wav_grid = np.linspace(wav_min, wav_max, N_wavelengths)
        self.samples = []
        
        if sample_ids is not None:
            sample_id_iter = sample_ids
        else:
            sample_id_iter = os.listdir(data_dir)
            
            
        for sample_id in sample_id_iter:
            sample_path = os.path.join(data_dir, sample_id)
            if not os.path.isdir(sample_path):
                continue

            spec_path = os.path.join(sample_path, 'spectrum_final.dat')
            if not os.path.exists(spec_path):
                continue
            sample_txt = os.path.join(sample_path, 'sample.txt')

            # Load data
            data = pd.read_csv(spec_path, sep=r'\s+', header=None, names=['time', 'frequency', 'flux', 'fluxerr'], comment='#')
            sep = ','
            if 'Callie' in sample_path:
                sep = ' '
            descriptor_df = pd.read_csv(sample_txt, sep=sep, header=0)
            descriptor = torch.tensor([
                descriptor_df['D'].values[0],
                descriptor_df['R_2'].values[0],
                descriptor_df['R_28'].values[0],
                descriptor_df['R_opacity'].values[0],
                descriptor_df['min_vel'].values[0],
                descriptor_df['max_vel'].values[0],
                descriptor_df['total_2'].values[0],
                descriptor_df['total_28'].values[0],
                descriptor_df['total_opacity'].values[0],
            ], dtype=torch.float32)

            # Convert frequency to wavelength and trim
            c = 3.0E8 * u.meter/u.second
            angstrom = 1.0E-10 * u.meter
            erg = 1.0E-7 * u.kilogram*(u.meter/u.second)**2
            cm = 1.0E-2 * u.meter
            hz = 1/u.second
            freq_flux = erg/(cm**2*u.second*hz)
            wav_flux = erg/(cm**2*angstrom*u.second)

            frequency = data['frequency'].values * hz
            flux_fnu = data['flux'].values * freq_flux
            wav = (c / frequency)
            wav_cm = wav.to(cm)
            flux_flambda = (flux_fnu * c / wav_cm**2).to(wav_flux)
            wav_angstrom = wav.to(angstrom).value

            mask = (wav_angstrom > wav_min) & (wav_angstrom < wav_max) & np.isfinite(flux_flambda.value)
            wav = wav_angstrom[mask]
            flux = flux_flambda.value[mask]
            time = data['time'].values[mask]

            mask = (wav > wav_min) & (wav < wav_max)
            wav = wav[mask]
            flux = flux[mask]
            time = time[mask]

            sec_per_day = 86400
            mask_time = (time > 4*sec_per_day) & (time < 30*sec_per_day)
            wav = wav[mask_time]
            flux = flux[mask_time]
            time = time[mask_time]

            time_values = np.unique(time)
            for t in time_values:
                this_mask = (time == t)
                wav_t = wav[this_mask]
                flux_t = flux[this_mask]
                wav_t = wav_t[::-1]
                flux_t = flux_t[::-1].copy()

                # Interpolate spectrum to fixed grid
                if len(wav_t) < 10:
                    continue

                flux_interp = np.interp(self.fixed_wav_grid, wav_t, flux_t, left=0, right=0)

                # Floor zeros and negatives for log-safety
                flux_safe = np.where(flux_interp > 1e-10, flux_interp, 1e-10)
                log_flux = np.log10(flux_safe)

                
                self.samples.append({
                    'descriptor': descriptor.clone(),
                    'time': torch.tensor(t, dtype=torch.float64),
                    'wav': torch.tensor(self.fixed_wav_grid, dtype=torch.float64),
                    'flux': torch.tensor(log_flux, dtype=torch.float64),  # should be log10(flux)
                    'sample_id': sample_id
                })
                

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample