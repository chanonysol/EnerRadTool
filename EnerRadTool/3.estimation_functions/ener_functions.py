# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:50:28 2024

@author: solen

Description :
    Functions participating in the energy estimation
    Constants
    Integration
    Spectral forms etc.
"""

from numpy import argwhere, mean, array, arange, concatenate, \
    zeros_like, log10, exp, median, ones, vstack
from numpy.fft import rfft, rfftfreq
from numpy.linalg import lstsq

from scipy.interpolate import splrep, splev
from obspy import Stream
from scipy.interpolate import splrep, splev

from multiprocessing import Pool
#from results_correction import
from time_rupture_functions import find_tr

from colorama import Back, Fore, Style
from tqdm import tqdm
import sys
from copy import deepcopy

def simpson_integration(data, dt):
    integral = zeros_like(data)
    integral[0] = 0
    for ind in range(1, len(data)):
        integral[ind] = (dt/6*(data[ind]
                               + 4*mean(data[ind-1:ind+1])
                               + data[ind-1])
                         + integral[ind-1])
    return integral


#def write_ener_files(path,name):

# TODO search tr in stream
# TODO add option save energy if not don't
# TODO writing file to stock the values maybe? or wait for final


def ener_flux_station(disp_trace, spectra=False):
    # Fourier transform
    spec = rfft(disp_trace.data)
    freq = rfftfreq(disp_trace.stats.npts, d=disp_trace.stats.delta)  # .round(4)
    # Limit to the Broband frequencies limit
    index = argwhere((freq >= 1/70) & (freq <= 2))[:, 0]
    spec, freq = spec[index], freq[index]

    # attenuation parameter
    t_star = 0.5-0.5*log10(freq) # for 0.1 <= freq <= 1 Hz
    t_star[argwhere(freq <= 0.1)[:, 0]] = 0.9-0.1 * \
        log10(freq[argwhere(freq <= 0.1)[:, 0]]) # for 0.1Hz <= freq
    t_star[argwhere(freq >= 1)[:, 0]] = 0.5-0.1 * \
        log10(freq[argwhere(freq >= 1)[:, 0]]) # for freq <= 1 Hz
    # function to integrate
    func = (freq*abs(spec))**2*exp(freq*t_star)
    # integration in m²/s

    disp_trace.integral = simpson_integration(func, (freq[-1]-freq[0])/len(freq))
    disp_trace.stats.ener = disp_trace.integral[-1]-disp_trace.integral[0]
    disp_trace.stats.freq_npts = len(freq)
    disp_trace.spec = spec

def trace_ener(argum):
    tr,tmax = argum
    en = [0]
    for time in arange(-29,tmax):
        tr_use = tr.copy()
        tr_use.trim(tr_use.stats.starttime+60-30,tr_use.stats.starttime+60+time)
        ener_flux_station(tr_use, spectra=False)
        en.append(tr_use.stats.ener)
    en=array(en)*tr_use.stats.econst
    #sys.stdout.write(f"{max(en[:2*60]):.1e} - ")
    #en-=max(en[:20])
    return en#[20:]

def smooth_curve_mean(t, eghb, dt=None):
    adjust = t.min()
    if adjust<0:
        t-=adjust

    if not dt:
        dt = int(t.max()//20)
    xi = arange(dt, t.max()-dt+1, dt).astype(int)
    yi = eghb[xi]
    for (j, i) in enumerate(xi):
        yi[j] = mean(eghb[i-dt:i+dt+1])
    yi = concatenate(([eghb[0]], yi, [mean(eghb[xi[-1]:-1])]))
    xi = concatenate(([0], xi, [xi[-1]+dt]))

    tcl = splrep(xi, yi, s=1)
    xs = arange(t.min(), t.max()+1, 1)
    ys = splev(xs, tcl)

    if adjust<0:
        xs+=adjust
    return xs,ys

# =============================================================================
#
# =============================================================================

def find_ener(ener_bb_sta,ener_hf_sta,param):
    # mean energy
    ener_bb = mean(ener_bb_sta,axis=0)
    ener_hf = mean(ener_hf_sta,axis=0)
    t = arange(0,param.time_max)

    #smooth curve
    ts,ener_bb_s = smooth_curve_mean(deepcopy(t), ener_bb)
    _ ,ener_hf_s = smooth_curve_mean(deepcopy(t), ener_hf)
    #find rupture time
    tr_raw = find_tr(param.time_eval_method,t,ener_hf)
    tr_smooth = find_tr(param.time_eval_method,ts,ener_hf_s)
    return [ener_bb,ener_hf,t,tr_raw],[ener_bb_s,ener_hf_s,ts,tr_smooth]

def calc_ener_rt(st_bb,st_hf,param):
    with Pool(14) as p:
        sys.stdout.write(f"{Fore.RED + Style.DIM}"
                          +f" > Starting BroadBand estimations"
                          +f"{Style.RESET_ALL}\n")
        argums = [(tr,param.time_max) for tr in st_bb]
        ener_bb_sta = vstack([result for result
                          in tqdm(p.imap(trace_ener,argums),
                                  total=len(argums))])

        sys.stdout.write(f"{Fore.RED + Style.DIM}"
                          +f" > Starting HighFrequencies estimations"
                          +f"{Style.RESET_ALL}\n")
        argums = [(tr,param.time_max) for tr in st_hf]
        ener_hf_sta = vstack([result for result
                          in tqdm(p.imap(trace_ener,argums),
                                  total=len(argums))])

    sys.stdout.write(f"{Fore.RED + Style.DIM}"
                      +f" > Removing pre_P-waves energy trend"
                      +f"{Style.RESET_ALL}\n")
    # remove trend of energy before-Pwaves
    for ener in ener_bb_sta :
        mean_rate = (ener[30]-ener[0])/30
        ener -= mean_rate
    for ener in ener_hf_sta :
        mean_rate = (ener[30]-ener[0])/30
        ener -= mean_rate
    # starts at 0s after Pwaves
    ener_bb_sta = ener_bb_sta[:,30:]
    ener_hf_sta = ener_hf_sta[:,30:]
    #ener estimation
    sys.stdout.write(f"{Fore.RED + Style.DIM}"
                      +f" > Mean energy estimation"
                      +f"{Style.RESET_ALL}\n")
    raw_results, smooth_results = find_ener(ener_bb_sta,ener_hf_sta,param)
    return [ener_bb_sta, raw_results, ener_hf_sta, smooth_results]

def calc_ener_corrected(ener_bb_sta, ener_hf_sta, bb_index, hf_index):
    return


