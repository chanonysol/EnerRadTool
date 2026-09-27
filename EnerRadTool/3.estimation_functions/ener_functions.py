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

from numpy import argwhere, mean, array, arange, concatenate, shape, \
    zeros_like, log10, exp, median, ones, vstack, nan, empty, pi, nanmean, nansum
from numpy.fft import rfft, rfftfreq
from numpy.linalg import lstsq

from scipy.interpolate import splrep, splev
from scipy.integrate import simpson, romb 
from obspy import Stream


from multiprocessing import Pool
#from results_correction import
from time_rupture_functions import find_tr
from data_utils import correct_regional_dist, find_vel, sqrt, interp_imp_P,\
    interp_imp_S, geometrical_spreading_s, attenuation_s, degrees2kilometers

from colorama import Back, Fore, Style
from tqdm import tqdm
import sys
from copy import deepcopy

NPOL = 14

# =============================================================================
# ---- Useful functions
# =============================================================================

def simpson_integration(data, dt):
    integral = zeros_like(data)
    integral[0] = 0
    for ind in range(1, len(data)):
        integral[ind] = (dt/6*(data[ind]
                               + 4*mean(data[ind-1:ind+1])
                               + data[ind-1])
                         + integral[ind-1])
    return integral


def smooth_curve_mean(t, eghb, dt=None):
    adjust = t.min()
    if adjust<0:
        t-=adjust

    if not dt:
        dt = int(t.max()//50)
    xi = arange(dt, len(t)-dt+1, dt).astype(int)
    yi = eghb[xi]
    for (j, i) in enumerate(xi):
        yi[j] = nanmean(eghb[i-dt:i+dt+1])
    yi = concatenate(([eghb[0]], yi, [nanmean(eghb[xi[-1]:-1])]))
    xi = concatenate(([0], xi, [xi[-1]+dt]))

    tcl = splrep(xi, yi, s=1)
    xs = arange(t.min(), t.max()+1, 1)
    ys = splev(xs, tcl)

    if adjust<0:
        xs+=adjust
    return xs,ys

def impedance(freq,depth,typ = "hardrock"):
    quarterperiods = (1/4/deepcopy(freq)).round(3)
    
    if typ == "gener":
        from data_utils import generic_rock_vel
        depths_gener, A_gener = generic_rock_vel(quarterperiods,depth, pwave=True)
        return A_gener
    if typ == "hardrock":
        from data_utils import very_hard_rock_vel
        depths_hardrock, A_hardrock = very_hard_rock_vel(quarterperiods, depth, pwave=True)
        return A_hardrock

def weight_contrib(ev,vs):
    from numpy import sin, where, delete
    if len(vs)==1:
        return array([1])
    dv=25
    vi = ev.plan1.strike
    A = abs(sin((vi-vs)*pi/180))
    B = array([sum(1-where(abs(delete(vs,k)-v)<dv,abs(delete(vs,k)-v),0)/dv) 
                for k,v in enumerate(vs)])
    
    return A/B
# =============================================================================
# ---- Ener flux P
# =============================================================================
def ener_flux_station(disp_trace,depth, spectra=False, imp_cor = False):
    # Fourier transform
    spec = rfft(disp_trace.data) * disp_trace.stats.delta #disp_trace is the VEL
    freq = rfftfreq(disp_trace.stats.npts, d=disp_trace.stats.delta)  # .round(4)
    # Limit to the Broband frequencies limit
    index = argwhere((freq >= 1/70) & (freq <= 2))[:, 0]
    spec, freq = spec[index], freq[index]

    # attenuation parameter
    t_star = 0.5-0.5*log10(freq)# for 0.1 <= freq <= 1 Hz
    t_star[argwhere(freq <= 0.1)[:, 0]] = 0.9-0.1 * \
        log10(freq[argwhere(freq <= 0.1)[:, 0]]) # for 0.1Hz <= freq
    t_star[argwhere(freq >= 1)[:, 0]] = 0.5-0.1 * \
        log10(freq[argwhere(freq >= 1)[:, 0]]) # for freq <= 1 Hz
    # function to integrate
    # freq *= 2*pi
    gamma = (0.55 if depth>=450 
             else 0.6 if depth >=300
             else 0.75 if depth >=200
             else 0.8 if depth >=80
             else 1)
    
    if imp_cor:
        imp = interp_imp_P(freq,depth,typ = "gener")**-1
        func = (abs(spec))**2*exp(freq*2*pi*t_star*gamma)*imp
    else :
        func = (abs(spec))**2*exp(freq*2*pi*t_star*gamma)
    # integration in m²/s

    #disp_trace.stats.ener = simpson(y=func,x=freq)
    disp_trace.stats.ener = simpson(y=func,x=freq*2*pi)
    disp_trace.stats.freq_npts = len(freq)
    disp_trace.spec = spec

# from scipy.integrate import cumulative_simpson
# figure = plt.figure(figsize=(12,8))
# freqs,specs,eners=[],[],[]
# for j,disp_trace in enumerate(st+st_hf):
#     c="silver"
#     if j>len(st):c="pink"
#     sta = inv.select(station=disp_trace.stats.station)[0][0].extra.correction
#     spec = rfft(disp_trace.data) * disp_trace.stats.delta #disp_trace is the VEL
#     freq = rfftfreq(disp_trace.stats.npts, d=disp_trace.stats.delta)  # .round(4)
#     # Limit to the Broband frequencies limit
#     index = argwhere((freq >= 1/70) & (freq <= 2))[:, 0]
#     spec, freq = spec[index], freq[index]
#     t_star = 0.5-0.5*log10(freq)# for 0.1 <= freq <= 1 Hz
#     t_star[argwhere(freq <= 0.1)[:, 0]] = 0.9-0.1 * \
#         log10(freq[argwhere(freq <= 0.1)[:, 0]]) # for 0.1Hz <= freq
#     t_star[argwhere(freq >= 1)[:, 0]] = 0.5-0.1 * \
#         log10(freq[argwhere(freq >= 1)[:, 0]]) # for freq <= 1 Hz

#     gamma = (0.55 if depth>=450 
#               else 0.6 if depth >=300
#               else 0.75 if depth >=200
#               else 0.8 if depth >=80
#               else 1)
    
#     func = (abs(spec))**2*exp(freq*2*pi*t_star*gamma)
#     ener = cumulative_simpson(y=func,x=freq*2*pi,initial=0)*sta.ener
#     # integration in m²/s
#     plt.loglog(freq,abs(spec),c=c,alpha=0.1,zorder=-10)
#     freqs.append(freq);eners.append(ener);specs.append(abs(spec))
# plt.loglog(mean(freqs,axis=0),median(specs[:len(st)],axis=0),c="k",alpha=0.5)
# plt.loglog(mean(freqs,axis=0),median(specs[len(st):],axis=0),c="r",alpha=0.5)
# plt.grid()
# plt.axvline(0.5,c="firebrick",lw=2,zorder=-20)

# def inv_mw(Mw):
#     return 10**((Mw+10.73)*3/2)
# mags = arange(5,9)
# mags=ev.mag
# M0 = inv_mw(mags)*1e-7
# E = 5e-5*M0
# k=0.372; a=(mags/4)**3; rho,alp,bet=find_vel(15); mu = 3.3e4
# fc = k*bet/a ; gamma=2
# spec_brune = [M/(1+(freq/f)**gamma) for M,f in zip(M0,fc)]
# =============================================================================
# ---- Ener flux S
# =============================================================================
def ener_S_station(disp_trace,depth,stainfo,cross_dist,default="A"):
    stalat,stalon = [stainfo.latitude,stainfo.longitude]
    stainfo = stainfo.extra
    site,k,beta = stainfo.site_class.values()
    if site == None : site=default
    dist = degrees2kilometers(stainfo.station_param.distance)
    # Fourier transform
    spec = rfft(disp_trace.data) * disp_trace.stats.delta #disp_trace is the VEL
    freq = rfftfreq(disp_trace.stats.npts, d=disp_trace.stats.delta)  # .round(4)
    # Limit to the Broband frequencies limit
    index = argwhere((freq>=1/disp_trace.stats.npts/disp_trace.stats.delta)&(freq<=10))[:,0]
    spec = spec[index] ; freq = freq[index]
    
    C = 2*sqrt(0.4/1.125)
    # attenuation parameter
    g = sqrt(geometrical_spreading_s(freq, sqrt(dist**2+depth**2),
             cross_dist=cross_dist)) #km
    q = attenuation_s(freq,stalat,stalon)
     # for freq <= 1 Hz
    # function to integrate
    # freq *= 2*pi
    
    beta=3.2
    rho_0, alpha_0, beta_0 = find_vel(depth)
    rho_0*=1e3;beta_0*=1e3;beta*=1e3 #m/s and kg/m3
    dist*=1e3
    
    imp = interp_imp_S(stalat,stalon,freq,site,depth)
    func = (abs(spec) * g*1e3/C * exp(freq*2*pi*dist/2/beta/q)
               *(imp*exp(-freq*pi*k))**-1) #m2/s
    
    # integration in m²/s

    #disp_trace.stats.ener = simpson(y=func,x=freq)
    
    I = concatenate(([0],[simpson(y=func[:i]**2,x=freq[:i]*2*pi) 
                          for i in range(1,len(func))]))*2/5*4*rho_0*beta_0
    #disp_trace.stats.ener = simpson(y=func**2,x=freq*2*pi)
    disp_trace.stats.ener = I[-1]
    disp_trace.stats.int = I
    disp_trace.stats.freq = freq
    disp_trace.stats.az = stainfo.station_param.az
# =============================================================================
# ---- multiproc
# =============================================================================

def trace_ener(argum):
    tr,econst,tmax,dt,de,p = argum
    t0 = -30
    en = [0]
    for time in arange(t0,tmax,dt)[1:]:
        tr_use = tr.copy()
        tr_use.trim(tr_use.stats.starttime+60+t0,tr_use.stats.starttime+60+time)
        ener_flux_station(tr_use,de, spectra=False, imp_cor = p)
        en.append(tr_use.stats.ener)
    en=array(en)*econst
    # tr_en = tr.copy()
    # tr_en.stats.delta = 1.;tr_en.stats.npts = len(en);tr_en.stats.sampling_rate = 1.
    # tr_en.data = en; tr_en.stats.channel = "ENE"
    #sys.stdout.write(f"{max(en[:2*60]):.1e} - ")
    #en-=max(en[:20])
    return en#[20:]


# def trace_ener_S(argum):
#     tr,stainfo,tmax,dt,de,p = argum
#     coords = [stainfo.latitude,stainfo.longitude]
#     stainfo = stainfo.extra
#     stainfo.coords = coords
    
#     t0 = -30
#     en = [0]
    
#     cross_dist = de*3
    
#     for time in arange(t0,tmax,dt)[1:]:
#         tr_use = tr.copy()
#         tr_use.trim(tr_use.stats.starttime+30+t0,tr_use.stats.starttime+30+time)
#         ener_S_station(tr_use,de,stainfo,cross_dist)
#         en.append(tr_use.stats.ener)
#     #sys.stdout.write(f"{max(en[:2*60]):.1e} - ")
#     #en-=max(en[:20])
#     return en#[20:]

# =============================================================================
# ----final estimate
# =============================================================================
def temp_save(t,ener_bb_sta,ener_hf_sta,st_bb,st_hf,resultpath):
    from results_writing import write_raw_results_sta
    ener_bb = nanmean(ener_bb_sta,axis=0)
    ener_hf = nanmean(ener_hf_sta,axis=0)
    ts,ener_bb_s = smooth_curve_mean(deepcopy(t), ener_bb)
    _ ,ener_hf_s = smooth_curve_mean(deepcopy(t), ener_hf)
    write_raw_results_sta(f"{resultpath}/temp_save_BB.dat",ener_bb_sta,
                          ener_bb,ener_bb_s,[tr.id for tr in st_bb],rt=0)
    write_raw_results_sta(f"{resultpath}/temp_save_HF.dat",ener_bb_sta,
                          ener_hf,ener_hf_s,[tr.id for tr in st_hf],rt=0)
 
def find_ener(ener_bb_sta,ener_hf_sta,param,dt=1):
    # mean energy
    ener_hf = nanmean(ener_hf_sta,axis=0)
    t = arange(0,param.time_max,dt)

    if ener_bb_sta.size == 0 :
        ener_bb = array([nan for i in t])
    else :
        ener_bb = nanmean(ener_bb_sta,axis=0)
            
    #smooth curve
    ts,ener_bb_s = smooth_curve_mean(deepcopy(t), ener_bb)
    _ ,ener_hf_s = smooth_curve_mean(deepcopy(t), ener_hf)
    #find rupture time
    tr_raw = find_tr(param.time_eval_method,t,ener_hf)
    tr_smooth = find_tr(param.time_eval_method,ts,ener_hf_s)
    return [ener_bb,ener_hf,t,tr_raw],[ener_bb_s,ener_hf_s,ts,tr_smooth]
    
    
def calc_ener_S(st_bb,st_hf,inv,param,ev):
    cross_dist = ev.depth*3
    # if ev.sreg in ["Kermadec","Puysegur"] :
    #     cross_dist = 200.
    for tr in tqdm(st_bb,total=len(st_bb),
                   desc="Calc close S ener"):
        ener_S_station(tr,ev.depth,
                       inv.select(network = tr.stats.network, 
                        station=tr.stats.station)[0][0],
                       cross_dist,
                       default = param.deviation_correction.unknown_site_class 
                       if param.deviation_correction.unknown_site_class else "A")
    for tr in tqdm(st_hf,total=len(st_hf),
                   desc="Calc close S ener HF"):
        ener_S_station(tr,ev.depth,
                       inv.select(network = tr.stats.network, 
                        station=tr.stats.station)[0][0],
                       cross_dist,
                       default = param.deviation_correction.unknown_site_class 
                       if param.deviation_correction.unknown_site_class else "A")
    
    ener_bb = array([tr.stats.ener for tr in st_bb])
    ener_hf = array([tr.stats.ener for tr in st_hf])
    return ener_bb,ener_hf

def calc_ener_rt(st_bb,st_hf,inv,param,ev,resultpath,ener_sta=None,dt=1):
    ener_bb_sta = empty((0,1)); ener_hf_sta = empty((0,1))
    f = trace_ener
    with Pool(NPOL) as p:
        sys.stdout.write(f"{Fore.RED + Style.DIM}"
                          +f" > Starting BroadBand estimations"
                          +f"{Style.RESET_ALL}\n")
        argums = [(tr,
                   inv.select(network = tr.stats.network, 
                    station=tr.stats.station)[0][0].extra.correction.ener ,
                   param.time_max,dt,ev.depth,
                   True if param.waves.S_near and param.waves.P_teleseismic else False) 
                  for tr in st_bb]
        if len(argums)!=0:
            ener_bb_sta = vstack([result for result
                              in tqdm(p.imap(f,argums),
                                      total=len(argums))])
            

        sys.stdout.write(f"{Fore.RED + Style.DIM}"
                          +f" > Starting HighFrequencies estimations"
                          +f"{Style.RESET_ALL}\n")
        argums = [(tr,
                   inv.select(network = tr.stats.network, 
                    station=tr.stats.station)[0][0].extra.correction.ener ,
                   param.time_max,dt,ev.depth,
                   True if param.waves.S_near and param.waves.P_teleseismic else False)
                   for tr in st_hf]
        if len(argums)!=0:
            ener_hf_sta = vstack([result for result
                              in tqdm(p.imap(f,argums),
                                      total=len(argums))])

    sys.stdout.write(f"{Fore.RED + Style.DIM}"
                      +f" > Removing pre_P-waves energy trend"
                      +f"{Style.RESET_ALL}\n")
    #remove trend of energy before-Pwaves
    if ener_bb_sta.size!=0:
        for ener in ener_bb_sta :
            mean_rate = (ener[30]-ener[0])/30
            ener -= mean_rate
        # starts at 0s after Pwaves
        ener_bb_sta = ener_bb_sta[:,30:]
    if ener_hf_sta.size!=0:
        for ener in ener_hf_sta :
            mean_rate = (ener[30]-ener[0])/30
            ener -= mean_rate
        # starts at 0s after Pwaves
        ener_hf_sta = ener_hf_sta[:,30:]
    #correct for regional distance
    # temp_save(arange(0,shape(ener_bb_sta)[1]),ener_bb_sta,ener_hf_sta,st_bb,st_hf)
    if param.waves.P_regional:
        ener_bb_sta, ener_hf_sta = calc_ener_corrected_reg(ener_bb_sta, 
                                                           ener_hf_sta, 
                                                           st_bb,st_hf,inv)
    ener_bb_sta, ener_hf_sta = calc_ener_corrected_sta(ener_bb_sta, 
                                                       ener_hf_sta, 
                                                       st_bb, st_hf, inv)
    #add to previous iterations
    if ener_sta:
        if ener_bb_sta.size!=0:
            ener_bb_sta = vstack((ener_sta[0],ener_bb_sta))
        else:
            ener_bb_sta = ener_sta[0]

        if ener_hf_sta.size!=0:
            ener_hf_sta = vstack((ener_sta[1],ener_hf_sta))
        else:
            ener_hf_sta = ener_sta[1]
    #ener estimation
    
    sys.stdout.write(f"{Fore.RED + Style.DIM}"
                      +f" > Mean energy estimation"
                      +f"{Style.RESET_ALL}\n")
    # from do_estimation import folder_name
    # temp_save(arange(0,len(ener_bb_sta[0]),1),ener_bb_sta,ener_hf_sta,st_bb,st_hf,
    #           resultpath)
    raw_results, smooth_results = find_ener(ener_bb_sta,ener_hf_sta,param,dt=dt)
    return [ener_bb_sta, raw_results, ener_hf_sta, smooth_results]

# =============================================================================
# ---- correction
# =============================================================================

def calc_ener_corrected_reg(ener_bb_sta, ener_hf_sta, st_bb, st_hf, inv):
    dist_bb = array([inv.select(station = 
                                tr.stats.station)[0][0].extra.station_param.distance 
                     for tr in st_bb])
    dist_hf = array([inv.select(station = 
                                tr.stats.station)[0][0].extra.station_param.distance
                     for tr in st_hf])
    
    ener_sta_bb = correct_regional_dist(dist_bb,ener_bb_sta)
    ener_sta_hf = correct_regional_dist(dist_hf,ener_hf_sta)

    #ener_sta_hf /= 5
    return ener_sta_bb, ener_sta_hf

def calc_ener_corrected_sta(ener_bb_sta, ener_hf_sta, st_bb, st_hf, inv):
    from  numpy import dtype, loadtxt, where,sign,vstack
    file = 'EnerRadTool/2.data_functions/sta_cor_perma0.5.dat'
    datatype=open(file,"r").readlines()[0][1:].split()
    vals = ["f8"]*len(datatype)
    vals[0] = "U25";
    dty = dtype([(d,v) for d,v in zip (datatype, vals)])
    mat = loadtxt(file,comments="#",dtype=dty)
    
    correction = where((abs(mat["av_dev"])>1/3) & 
                       (sign(mat["av_dev"])==sign(mat["med_dev"])),
                       10**-mat["av_dev"],1)
    
    sta2cor= array([correction[argwhere(mat["station"] == tr.id)[0,0]]
                    if tr.id in mat["station"] else 1.
                    for tr in st_bb])
    sta2corhf= array([correction[argwhere(mat["station"] == tr.id)[0,0]]
                      if tr.id in mat["station"] else 1.
                      for tr in st_hf])

    sta2cor = vstack([sta2cor]*shape(ener_bb_sta)[1]).T
    sta2corhf = vstack([sta2corhf]*shape(ener_hf_sta)[1]).T
    return ener_bb_sta*sta2cor, ener_hf_sta*sta2corhf
