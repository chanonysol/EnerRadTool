# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:50:28 2024

@author: solen

Description :
    Preprocessing the traces to remove the site effects, instrumental
    response etc.
"""

import os
import random
import colorama
import sys
import warnings
import logging

from numpy import array, vstack, argwhere,\
    sin, cos, pi, arcsin, linspace, arange
from numpy.fft import rfft, irfft

from obspy import read, read_inventory, Inventory, Stream
from obspy.taup.taup_geo import calc_dist_azi

from data_utils import request_val_p_train, waves_parameters, snr_mag,\
    verif_size, signaltonoise

from glob import glob
from time import time
from colorama import Back, Fore, Style

def precalc(inv,ev):
    dist_list, az_list, _, lat_list, lon_list = array([
        calc_dist_azi(ev.lat, ev.lon, chan.latitude,
                      chan.longitude, 6371., 0)
        +(sta.latitude,sta.longitude)
        for net in inv for sta in net for chan in sta
        ]).round(3).T

    p_times,p_incid= request_val_p_train(["P","time","incident"],
                                          ev.depth, dist_list, model_name="iasp91")[0]

    ray_const, ener_const = waves_parameters(dist_list,p_incid)

    return dist_list, az_list, lat_list, lon_list, ray_const,\
        ener_const, p_times, p_incid

def add_parameters_stream(inv,ev,st,distlist,infodict):
    di,az,la,lo,rc,ec,_, _ = infodict
    for dist,azi,lat,lon,C,Econst,tr in zip(di,az,la,lo,rc,ec,st) :
        tr.stats.distance = dist
        tr.stats.az = azi
        tr.stats.ray_correction = C
        tr.stats.econst = Econst
        tr.stats.latitude = lat
        tr.stats.longitude = lon
    return st

def filter_st(inv,ev,st,channels,times,infodict,freqband):
    _,_,_,_,_,_,p_times, _ = infodict
    (origin_time,max_duration) = times
    snrlim = snr_mag(ev.mag)

    if freqband == "BB":
        pre_filt = [0.014,0.2,1.8,2]
        channels = [value for value in channels if value in ["BH","SH"]]
    if freqband == "HF":
        pre_filt = [0.5,0.6,1.9,2]
        channels = [value for value in channels if value in ["BH","HH"]]

    sys.stdout.write(f"{Style.RESET_ALL}\n")
    for (i,tr) in enumerate(st):
        tr.data = irfft(rfft(tr.data)/tr.stats.ray_correction)
        # if tr.stats.network in ['JP'] or tr.stats.station in ['AS31']:
        #     st.remove(tr)
        #     continue
        if tr.stats.channel[:-1] not in channels: #remove unwanted channels
            st.remove(tr)
            continue
        if all(tr.data==tr.data[0]): #remove flat data
            st.remove(tr)
            continue

    st.remove_response(inventory=inv, pre_filt=pre_filt, output="DISP",
                        water_level = None, zero_mean = True,
                        hide_sensitivity_mismatch_warning = True)
    for (p_t,tr) in zip(p_times,st):
        tr.detrend("spline",order=4,dspline=tr.stats.npts//tr.stats.sampling_rate)
        tr.trim(tr.stats.starttime+(p_t-origin_time),
                tr.stats.starttime+(p_t-origin_time) + max_duration + 60)
    st.sort(['distance'])
    snr = signaltonoise(st.copy()).round(2)
    for i in argwhere(snr < snrlim)[::-1, 0]:
        st.remove(st[i])
    return st

def preprocessing(datapath,ev,channels,distlist):
    start = time()
    origin_time,[_,max_duration] = request_val_p_train(['time','P',"duration"],ev.depth,
                                                           distlist,model_name="iasp91")[0]
    origin_time = origin_time[0]
    times = (origin_time,max_duration)

    inv = read_inventory(f"{datapath}/stations/*.xml")
    infodict = precalc(inv, ev)

    st = read(f"{datapath}/*.mseed")
    st.sort(["network","station","location","channel"])
    st = add_parameters_stream(inv,ev,st,distlist,infodict)

    sys.stdout.write(f"\n{Fore.MAGENTA + Style.DIM}"
                      +f" > BroadBand")
    st_bb = filter_st(inv,ev,st.copy(),channels,times,infodict,"BB")

    sys.stdout.write(f"{Fore.MAGENTA + Style.DIM}"
                      +f" > HighFrequency filtered")
    st_hf = filter_st(inv,ev,st.copy(),channels,times,infodict,"HF")

    sys.stdout.write(f"{Fore.MAGENTA + Style.DIM}"
                    + f" > Data filtered "
                    + f"{Style.RESET_ALL}"
                    + f" (lasted{time()-start: 3.0f}s) \n")

    return st_bb,st_hf




# path = '../../../random/Kermadec/'
# ROOT = f"{path}newDATA/"
# st = read(f"{ROOT}*.mseed")
# names = [f"{tr.stats.network}.{tr.stats.station}.{tr.stats.location}.{tr.stats.channel}"
#          for tr in st]

