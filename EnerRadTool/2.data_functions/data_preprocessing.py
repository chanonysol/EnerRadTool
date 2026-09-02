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
import shutil

from numpy import array, vstack, argwhere,shape, reshape,mean,log10,\
    sin, cos, pi, arcsin, linspace, arange, unique, concatenate, median, isnan,\
    argsort,empty_like
from numpy.fft import rfft, irfft

from obspy import read, read_inventory, Inventory, Stream
from obspy.taup.taup_geo import calc_dist_azi

from data_utils import request_val_p_train, request_val_s_train, \
        waves_parameters, snr_mag, verif_size, signaltonoise, \
        modify_log, write_log, add_attribute_xml,\
        save_spectrum, kilometers2degrees, degrees2kilometers

from glob import glob
from time import time
from colorama import Back, Fore, Style

import warnings
warnings.filterwarnings('ignore')

def precalc(inv,ev,close=False):
    dist_list, az_list, baz_list = array([
        calc_dist_azi(ev.lat, ev.lon, chan.latitude,
                      chan.longitude, 6371., 0)
        for net in inv for sta in net for chan in sta
        ]).round(2).T
    
    if not close:
        dist_list=dist_list.round(1)
        p_times,p_incid,p_takeo= request_val_p_train(["P","time","incident","takeoff"],
                                              ev.depth, dist_list, model_name="iasp91")[0]
    
        ray_const, ener_const = waves_parameters(dist_list,p_incid,ev.depth)

        return array([dist_list, az_list, baz_list, ray_const,
            ener_const, p_times, p_incid, p_takeo])
    if close :
        s_times,s_incid,s_takeo= request_val_s_train(["S","time","incident","takeoff"],
                                              ev.depth, dist_list, 
                                              model_name="iasp91")[0]
    
        #ray_constSV, ray_constSH = waves_parameters(dist_list,s_incid,ev.depth)

        return array([dist_list, az_list, baz_list, empty_like(dist_list), empty_like(dist_list),
                      s_times, s_incid, s_takeo])

# =============================================================================
# ---- verification functions
# =============================================================================
def test_inventory(inv,st,datapath,invpath):
    st_remove = Stream()

    for tr in st[::-1] :
        inv2 = inv.select(channel=tr.stats.channel, station=tr.stats.station, 
                          location = tr.stats.location)
        if len(inv2)==0:
            st_remove += tr
            st.remove(tr)
            for file in glob(f"{datapath}/{tr.id}*"):
                os.remove(file)

    write_log(f"{datapath}/../",f"log_cleardata_{datapath.split('/')[-1]}",
              st_remove, None, update = "no xml file")
    
    inv_remove = Inventory()
    names_inv = [f"{net.code}.{sta.code}.{cha.location_code}.{cha.code}" 
                 for net in inv for sta in net for cha in sta]
    
    
    for i in [i for i in names_inv if i not in [tr.id for tr in st]]:
        jsonfile = os.path.exists(f"{invpath}/{i}.json")
        net,sta,loc,cha = i.split('.')
        if not jsonfile : 
            inv_remove += inv.select(network = net,station= sta,
                                     location = loc,channel = cha)
        inv = inv.remove(network = net, station = sta,
                        location = loc, channel = cha)
        if not jsonfile : os.remove(f'{invpath}/{i}.xml')
            
    write_log(f"{datapath}/../",f"log_cleardata_{datapath.split('/')[-1]}",
              None, inv_remove, update = "no mseed file")
    
    return inv, st

def test_response(inv,st,datapath,invpath):
    test = array([f"{net.code}.{sta.code}.{chan.location_code}.{chan.code}"
                  for net in inv for sta in net for chan in sta
                  if len(chan.response.response_stages)==0])
    test2 = array([f"{net.code}.{sta.code}.{chan.location_code}.{chan.code}"
                    for net in inv for sta in net for chan in sta
                    if (f"{net.code}.{sta.code}.{chan.location_code}.{chan.code}"not in test
                        and chan.response.response_stages[-1].output_units not in ['counts',"COUNTS","count"])
                    ])
    if test.size!=0 and test2.size!=0 :
        liste = concatenate((test,test2))
    elif test.size!=0 :
        liste = test
    else : liste = test2

    st_remove = Stream()
    inv_remove = Inventory()
    
    for name in liste :
        net,sta,loc,cha = name.split('.')
        tr = st.select(id=name)[0]
        inv_remove += inv.select(network = net,station= sta,
                                 location = loc,channel = cha)
        st_remove += tr
        st.remove(tr)
        inv = inv.remove(network = net,station= sta,
                         location = loc,channel = cha)
        os.remove(f'{invpath}/{name}.xml')
        os.remove(f'{datapath}/{name}.mseed')
        
    write_log(f"{datapath}/../",f"log_cleardata_{datapath.split('/')[-1]}", 
              None, inv_remove, update = "no resp stage")
    return inv, st

def test_data_size(inv,st,datapath,invpath):
    st_remove = Stream()
    inv_remove = Inventory()
    for (i,tr) in enumerate(st):
        if all(tr.data==tr.data[0]): #remove flat data
            net,sta,loc,cha = tr.id.split('.')
            inv_remove += inv.select(network = net,station= sta,
                                     location = loc,channel = cha)
            st_remove += tr
            st.remove(tr)
            inv = inv.remove(network = net,station= sta,
                             location = loc,channel = cha)
            os.remove(f'{invpath}/{tr.id}.xml')
            os.remove(f'{datapath}/{tr.id}.mseed')
            continue

    write_log(f"{datapath}/../",f"log_cleardata_{datapath.split('/')[-1]}",
              st_remove, None, update = "data record flat")
    return inv,st

def remove_nb_sta(inv,st,datapath,invpath,preprocess=False):
    st_remove = Stream()

    for tr in st:
        if tr.stats.station[-1] in arange(0,10,1).astype(str):
            net,sta,loc,cha = tr.id.split('.')
            
            st_remove += tr
            st.remove(tr)
            inv = inv.remove(network = net,station= sta,
                             location = loc,channel = cha)
            for p in glob(f'{datapath}/../*/{tr.id}*'):
               os.remove(p)
            continue

    write_log(f"{datapath}/../",f"log_cleardata_{datapath.split('/')[-1]}",
              st_remove, None, update = "unsure sta _")
    if preprocess :
        write_log(f"{datapath}/../",f"log_keptdata_{datapath.split('/')[-1]}",
                  st_remove, None, update = "unsure sta _")
        for freq in ["BB","HF"]:
            if os.path.exists(f"{preprocess}/log_keptdata_{datapath.split('/')[-1]}_{freq}"):
                write_log(f"{preprocess}",f"log_keptdata_{datapath.split('/')[-1]}_{freq}",
                          st_remove, None, update = "unsure sta _")
    return inv,st

def test_snr(inv,st,st_hf,datapath,invpath,timewindow = 60, snrlim = 5):
    st_remove = Stream()
    inv_remove = Inventory()
    st_delete = Stream()
    inv_delete = Inventory()
    
    snr = signaltonoise(st_hf.copy(),timewindow).round(2)
    indmax = array([argwhere(tr.data==tr.data.max())[0,0]*tr.stats.delta for tr in st_hf]).round(0)-timewindow
    delete_liste = unique(concatenate((argwhere(snr < 5)[:,0],argwhere(abs(indmax)>=median(indmax)+50)[:, 0])))
    
    for i in delete_liste[::-1]:
        tr = st_hf[i]
        name = st_hf[i].id
        net,sta,loc,cha = name.split('.')
        
        inv_delete += inv.select(network = net,station= sta,
                                 location = loc,channel = cha)
        st_delete += tr
        
        inv = inv.remove(network = net,station= sta,
                         location = loc,channel = cha)
        st_hf.remove(tr)
        st.remove(st.select(id=name)[0])
    
    if len(st_delete)!=0:
        write_log(f"{datapath}/../",f"log_cleardata_{datapath.split('/')[-1]}", 
                  st_delete, inv_delete, update = "noise above 5db")
    
    if len(st)!=0:
        write_log(f"{datapath}/../",f"log_keptdata_{datapath.split('/')[-1]}", 
                  st, inv)
    
    snr = signaltonoise(st_hf.copy(),timewindow).round(2)
    remove_liste = argwhere(snr < snrlim)[:,0]
    for i in remove_liste[::-1]:
        tr = st_hf[i]
        name = st_hf[i].id
        net,sta,loc,cha = name.split('.')
        
        inv_remove += inv.select(network = net,station= sta,
                                         location = loc,channel = cha)
        st_remove += tr
        inv = inv.remove(network = net,station= sta,
                         location = loc,channel = cha)
        st_hf.remove(tr)
        st.remove(st.select(id=name)[0])
    
    if len(st)!=0:
        modify_log(f"{datapath}/../",f"log_keptdata_{datapath.split('/')[-1]}", 
                  [tr.id for tr in st_remove], f"noise over {snrlim:2d}db")
        
    sys.stdout.write(f"{Fore.MAGENTA + Style.BRIGHT} ... filter snr")
    return inv,st,st_hf

def fixed_network_selection(inv,st,logpath,logfile):
    fixed_stalist = ['ADK', 'AFI', 'AUTOO', 'AUUHS', 'BELA', 'BJT', 'CASY', 'CCD',
           'COCO', 'COR', 'COYC', 'CTAO', 'DAV', 'DLV', 'DRV', 'EFI', 'FAKI',
           'FITZ', 'FORT', 'FUTU', 'GENI', 'GIRL', 'GUMO', 'HIA', 'HNR',
           'HOPE', 'HPIG', 'INCN', 'INU', 'JUBA', 'KAPI', 'KKM', 'KMI', 'KUM',
           'LHI', 'LHMI', 'LSA', 'MANU', 'MAW', 'MCQ', 'MDJ', 'MEEK', 'MIDW',
           'MTN', 'NFK', 'NOUC', 'NV31', 'NWAO', 'OPA', 'ORCD', 'PAF', 'PET',
           'PFO', 'PLAI', 'PLCA', 'PMBI', 'PMG', 'PMPB', 'PMSA', 'POHA',
           'PPTF', 'PSAD3', 'PTCN', 'QIZ', 'QSPA', 'RABL', 'RAR', 'SAUI',
           'SBA', 'SBM', 'SDPT', 'SNAA', 'SNZO', 'SOEI', 'SSE', 'TARA',
           'TATO', 'TAU', 'TMU', 'TNTI', 'TOLI2', 'TPH', 'TSK', 'TTA', 'TUC',
           'UGM', 'ULN', 'VNA3', 'WAKE', 'WRAB', 'WRKA', 'WUAZ', 'WVOR',
           'XAN', 'XMAS', 'YSS']

    inv_fixed = Inventory()
    for sta in fixed_stalist:
        inv_fixed += inv.select(station=sta)
    names = [f"{net.code}.{sta.code}.{chan.location_code}.{chan.code}"
             for net in inv_fixed
             for sta in net for chan in sta]
    st_fixed = st.copy()
    for tr in st_fixed[::-1]:
        if tr.id not in names:
            st_fixed.remove(tr)
    write_log(logpath,logfile+"_fixed", st_fixed, inv_fixed)

    return inv_fixed, st_fixed

def check_inv_files(invpath,inv):
    names_xml = [os.path.split(i)[-1][:-4] for i in glob(f"{invpath}/*.xml")]
    names_json = [os.path.split(i)[-1][:-5] for i in glob(f"{invpath}/*.json")]
    names_inv = [f"{net.code}.{sta.code}.{cha.location_code}.{cha.code}" 
                 for net in inv for sta in net for cha in sta]
    for i in names_inv:
        if i not in names_xml:
            net,sta = i.split('.')[:2]
            inv2 = inv.select(network=net,station=sta)
            for neti in inv2:
                for stai in neti:
                    for cha in stai:
                        loc,chan = cha.location_code,cha.code
                        inv2.select(location=loc,
                                    channel=chan).write(f"{invpath}/{net}.{sta}.{loc}.{chan}.xml",
                                                                        format = "STATIONXML")

def clear_data_file(datapath, invpath, inv = None, st_hf=None):
    for d in glob(f"{datapath}/*?H?.mseed"):
        os.remove(d)
    if inv:
        names = [j for j in glob(f"{invpath}/*") 
                if ".".join(os.path.split(j)[-1].split(".")[:-1]) 
                not in [i for i 
                    in [f"{net.code}.{sta.code}.{chan.location_code}.{chan.code}" 
                    for net in inv for sta in net  for chan in sta] 
                    if i in [tr.id for tr in st_hf]]
                and glob(f"{datapath}/{".".join(os.path.split(j)[-1].split(".")[:-1])}.d*.MSEED")==[]]
        for d in names:
            os.remove(d)
# =============================================================================
# ---- cleaning query
# =============================================================================
def clean_data(datapath,invpath,ev,save,ext = ".mseed"):
    starttime = time()
    
    sys.stdout.write(f"{Fore.MAGENTA + Style.BRIGHT} load {datapath.split('/')[-1]}...")
    inv = read_inventory(f"{invpath}/*.xml")
    st = read(f"{datapath}/*?H?{ext}")
    st.sort(["network","station","location","channel"])
    
    check_inv_files(invpath,inv)
    
    inv, st = test_inventory(inv,st, datapath,invpath)
    inv, st = test_response(inv,st, datapath,invpath)
    inv, st = test_data_size(inv,st, datapath,invpath)
    inv, st = remove_nb_sta(inv,st,datapath,invpath)
        
    check_inv_files(invpath,inv)
    
    sys.stdout.write(f"{Fore.MAGENTA + Style.BRIGHT} calc parameters...")
    inv = add_attribute_xml(inv,precalc(inv, ev),st,
                            datapath,invpath,saveinv=True,returninv=True)
    
    if ext==".mseed":
        sys.stdout.write(f"{Fore.MAGENTA + Style.BRIGHT} correct...")
        try:
            st.remove_response(inventory=inv, #pre_filt=[0.014,0.2,1.8,2.], 
                               output="VEL",
                               water_level = None, zero_mean = True,
                               hide_sensitivity_mismatch_warning = True)
        except:
            st = read(f"{datapath}/*?H?.mseed")
            for tr in st:
                try :
                    tr.remove_response(inventory=inv, #pre_filt=[0.014,0.2,1.8,2.], 
                                       output="VEL",
                                       water_level = None, zero_mean = True,
                                       hide_sensitivity_mismatch_warning = True)
                except :
                    write_log(f"{datapath}/../",f"log_cleardata_{datapath.split('/')[-1]}",
                      Stream(tr), None, update = "pb inst resp")
                    st.remove(tr)
                    inv = inv.remove(channel=tr.stats.channel, station=tr.stats.station, 
                                     location = tr.stats.location)
                else : continue
            
        st.filter("bandpass",freqmin = 1/70, freqmax = 2, 
                     corners = 4, zerophase=True)
        sys.stdout.write(f"{Fore.MAGENTA + Style.BRIGHT} cut data...")
        for tr in st:
            p_t = inv.select(channel=tr.stats.channel, station=tr.stats.station, 
                             location = tr.stats.location)[0][0].extra.ray_param.p_time
            if isnan(p_t):
                write_log(f"{datapath}/../",f"log_cleardata_{datapath.split('/')[-1]}",
                          st, None, update = "data record flat")
                net,sta,loc,cha = tr.id.split('.')
                st.remove(tr)
                inv = inv.remove(network = net,station= sta,
                                 location = loc,channel = cha)
                os.remove(f"{datapath}/{tr.id}.mseed")
                os.remove(f"{invpath}/{tr.id}.xml")         
                continue 
            tr.trim(tr.stats.starttime+(p_t) - 2*60,
                    tr.stats.starttime+(p_t) - 2*60 + 600)
            
        for tr in st:
            r_corr = inv.select(channel=tr.stats.channel, station=tr.stats.station, 
                             location = tr.stats.location)[0][0].extra.correction.ray
            tr.data = irfft(rfft(tr.data)/r_corr)
        
    sys.stdout.write(f"{Fore.MAGENTA + Style.BRIGHT} HF filter...")
    st_hf = st.copy()
    st_hf.filter("bandpass",freqmin = 0.5, freqmax = 2, 
                 corners = 4, zerophase=True)
    
    sys.stdout.write(f"{Fore.MAGENTA + Style.BRIGHT} check snr...")
    
    inv, st, st_hf = test_snr(inv,st,st_hf,datapath,invpath,timewindow=2*60,snrlim = 5)
    # inv_fixed, st_fixed_hf = fixed_network_selection(inv,st_hf,datapath,invpath)
    clear_data_file(datapath,invpath,inv,st_hf)

    sys.stdout.write(f"{Fore.MAGENTA + Style.BRIGHT} save data...")
    for name in [tr.id for tr in st_hf]:
        tr = st.select(id=name)[0]
        inv2 = inv.select(channel=tr.stats.channel, station=tr.stats.station, 
                         location = tr.stats.location)
        tr.write(f"{datapath}/{name}.d{inv2[0][0].extra.station_param.distance:02.0f}.MSEED", 
                 format="MSEED")
    
    #final_verif
    inv = read_inventory(f"{invpath}/*.xml")
    st = read(f"{datapath}/*?H?*.MSEED")
    st.sort(["network","station","location","channel"])

    inv, st = test_inventory(inv,st, datapath,invpath) 

    
    # save_spectrum(datapath,f"spec_BB_{datapath.split('/')[-1]}",st.copy())
    # save_spectrum(datapath,f"spec_HF_{datapath.split('/')[-1]}",st_hf.copy())
    
    endtime = time() - starttime
    
    sys.stdout.write(f"{Fore.MAGENTA + Style.DIM}"
                    + f" > Data filtered "
                    + f"{Style.RESET_ALL}"
                    + f" (lasted{time()-starttime: 3.0f}s) \n")
    
    return st,st_hf

def RT_comp(st,inv,ev):
    from tqdm import tqdm
    from obspy.signal.rotate import rotate_ne_rt
    from obspy import Stream
    
    st_rad,st_tran = Stream(), Stream()
    for i in unique([".".join([tr.stats.network,tr.stats.station]) for tr in st.copy()]): 
        net,sta = i.split(".")
        inv2 = inv.select(network=net,station=sta)[0][0]
        for ch in ["BH","HH"]:
            st2 = st.copy().select(network=net,station=sta,channel=f"{ch}?")
            if len(st2)==1: st.remove(st2[0]); continue
            if len(st2)==0:continue
            # if len(st2[0])!=len(st2[1]):
            #     for tr in st2 : st.remove(tr)
            #     continue
            st2.sort(['channel'])
            r,t = rotate_ne_rt(st2[1].data, st2[0].data, inv2.extra.station_param.baz)
            tr_rad, tr_tran = st2[1].copy(),st2[1].copy()
            tr_rad.stats.channel = tr_rad.stats.channel[:2]+"R"
            tr_tran.stats.channel = tr_tran.stats.channel[:2]+"T"
            tr_rad.data=r;tr_tran.data=t
            st_rad +=Stream(tr_rad);st_tran+=Stream(tr_tran)
    
    return st_tran,st_rad

def clean_close_data(datapath,invpath,ev,save):
    starttime = time()
    
    sys.stdout.write(f"{Fore.MAGENTA + Style.BRIGHT} load {datapath.split('/')[-1]}...")
    inv = read_inventory(f"{invpath}/*.xml")
    st = read(f"{datapath}/*?H?.mseed")
    st.sort(["network","station","location","channel"])
    
    check_inv_files(invpath,inv)
    
    inv, st = test_inventory(inv,st, datapath,invpath)
    inv, st = test_response(inv,st, datapath,invpath)
    inv, st = test_data_size(inv,st, datapath,invpath)
    inv, st = remove_nb_sta(inv,st,datapath,invpath)
        
    check_inv_files(invpath,inv)
    
    sys.stdout.write(f"{Fore.MAGENTA + Style.BRIGHT} calc parameters...")
    inv = add_attribute_xml(inv,precalc(inv, ev, close=True),st,
                            datapath,invpath,saveinv=True,returninv=True)
      
    sys.stdout.write(f"{Fore.MAGENTA + Style.BRIGHT} correct...")
    
    st.filter("bandpass",freqmin = 0.01, freqmax = 10, 
                 corners = 4, zerophase=True)
    try:
        st.remove_response(inventory=inv, #pre_filt=[0.014,0.2,1.8,2.], 
                           output="VEL",
                           water_level = None, zero_mean = True,
                           hide_sensitivity_mismatch_warning = True)
    except:
        st = read(f"{datapath}/*?H?.mseed")
        for tr in st:
            try :
                tr.remove_response(inventory=inv, #pre_filt=[0.014,0.2,1.8,2.], 
                                   output="VEL",
                                   water_level = None, zero_mean = True,
                                   hide_sensitivity_mismatch_warning = True)
            except :
                write_log(f"{datapath}/../",f"log_cleardata_{datapath.split('/')[-1]}",
                  Stream(tr), None, update = "pb inst resp")
                st.remove(tr)
                inv = inv.remove(channel=tr.stats.channel, station=tr.stats.station, 
                                 location = tr.stats.location)
            else : continue
        
    
    sys.stdout.write(f"{Fore.MAGENTA + Style.BRIGHT} cut data...")
    dists = array([inv.select(channel=tr.stats.channel, station=tr.stats.station, 
                     location = tr.stats.location)[0][0].extra.station_param.distance for tr in st])
    times = request_val_p_train(["P","time"],ev.depth,dists)[0][0]
    for tr,p_t in zip(st,times):
        if isnan(p_t):
            write_log(f"{datapath}/../",f"log_cleardata_{datapath.split('/')[-1]}",
                      st, None, update = "data record flat")
            net,sta,loc,cha = tr.id.split('.')
            st.remove(tr)
            inv = inv.remove(network = net,station= sta,
                             location = loc,channel = cha)
            os.remove(f"{datapath}/{tr.id}.mseed")
            os.remove(f"{invpath}/{tr.id}.xml")         
            continue 
        tr.trim(ev.date - 30 +p_t ,
                ev.date + p_t + 180)
        
    # for tr in st:
    #     if tr.stats.channel[-1]=="N":
    #         r_corr = inv.select(channel=tr.stats.channel, station=tr.stats.station, 
    #                          location = tr.stats.location)[0][0].extra.correction.raySV
    #     if tr.stats.channel[-1]=="E":
    #         r_corr = inv.select(channel=tr.stats.channel, station=tr.stats.station, 
    #                          location = tr.stats.location)[0][0].extra.correction.raySH
    #     tr.data = irfft(rfft(tr.data)/r_corr)
        
    sys.stdout.write(f"{Fore.MAGENTA + Style.BRIGHT} RT calc...")
    
    st = RT_comp(st,inv,ev)[0]
    st_hf = st.copy()
    st_hf.filter("bandpass",freqmin = 0.5, freqmax = 2, 
                  corners = 4, zerophase=True)

    sys.stdout.write(f"{Fore.MAGENTA + Style.BRIGHT} check snr...")

    inv, st, st_hf = test_snr(inv,st,st_hf,datapath,invpath,timewindow=30,snrlim = 5)
    # inv_fixed, st_fixed_hf = fixed_network_selection(inv,st_hf,datapath,invpath)
    
    clear_data_file(datapath,
                    invpath = [j for j in glob(f"{invpath}/*") 
                                           if ".".join(os.path.split(j)[-1].split(".")[:-1])[:-1] 
                                           not in [i for 
                                                   i in unique([f"{net.code}.{sta.code}.{chan.location_code}.{chan.code[:-1]}" 
                                                         for net in inv  for sta in net  for chan in sta])
                                                   if i in [tr.id[:-1] for tr in st_hf]]] )
    
    # for net in inv :
    #     for sta in net:
    #         for cha in sta :
    #             cha.code = cha.code[:-1]+"T"
                
    sys.stdout.write(f"{Fore.MAGENTA + Style.BRIGHT} save data...")
    for name in [tr.id for tr in st_hf]:
        tr = st.select(id=name)[0]
        inv2 = inv.select(station=tr.stats.station, 
                         location = tr.stats.location)
        di = degrees2kilometers(inv2[0][0].extra.station_param.distance)
        tr.write(f"{datapath}/{name}.d{di:03.0f}.MSEED", format="MSEED")
        inv2 = read_inventory(f"{invpath}/{name[:-1]}N.xml")
        inv2[0][0][0].code = tr.stats.channel
        # inv2 = inv2.remove(channel = tr.stats.channel)
        inv2.write(f"{invpath}/{name}.xml", format="STATIONXML")
    
    for p in glob(f"{invpath}/*E.xml")+glob(f"{invpath}/*N.xml"):
        os.remove(p)
    
    # save_spectrum(datapath,f"spec_BB_{datapath.split('/')[-1]}",st.copy())
    # save_spectrum(datapath,f"spec_HF_{datapath.split('/')[-1]}",st_hf.copy())
    
    endtime = time() - starttime
    
    sys.stdout.write(f"{Fore.MAGENTA + Style.DIM}"
                    + f" > Data filtered "
                    + f"{Style.RESET_ALL}"
                    + f" (lasted{time()-starttime: 3.0f}s) \n")

    return st,st_hf
# =============================================================================
# ---- preprocessing functions
# =============================================================================
def read_distance_range(datapath,mindist,maxdist):
    st = Stream()
    for d in arange(mindist,maxdist+1,1):
        st+= read(f"{datapath}/*.d{d:2.0f}.mseed")

def find_data_in_distance(datapath,dmin,dmax):
    dists = array([p.split('.')[4][1:3]
           for p in glob(f"{datapath}/*MSEED")]).astype(float)
    index = argwhere((dists<dmax)&(dists>=dmin))[:,0]
    glob(f"{datapath}/*MSEED")[index]
   
def select_chan(st,channels):
    new_st = Stream()
    test = array([[tr.stats.station,tr.stats.location,
                   tr.stats.channel,tr.stats.sampling_rate]
                  for tr in st 
                  if tr.stats.channel in [f"{code}Z" for code in channels]])
    names, counts = unique(test.T[0],return_counts=True)
    for na,c in zip(names, counts):
        vals = test[argwhere(test.T[0]==na)[:,0]]
        vals = vals.T
        vals = vals[:,argsort(vals[1])]
        vals = vals[:,argsort(vals[2])]
        sta,loc,cha,sp = vals.T[0]
        new_st += st.select(station=sta,location=loc,channel=cha)
    new_st.sort(["network","station"])
    return new_st

from data_utils import read_log        
def add_2_log(datapath,resultpath,mindist,maxdist,param):
    mat_res = read_log(resultpath,f"log_keptdata_{datapath.split('/')[-1]}_BB", 
                       return_mat=True)[:,2].astype(float)
    if any(mat_res>=maxdist) and maxdist!=param.distance.max: return
    if maxdist==param.distance.max and any(mat_res>=mindist): return
    if abs(mat_res.min()-mindist)<7. : return
    for freq in ["BB",'HF']:
        lines_res = read_log(resultpath,f"log_keptdata_{datapath.split('/')[-1]}_{freq}", modif=True)
        lines = read_log(datapath+"/../",f"log_keptdata_{datapath.split('/')[-1]}", modif=True)
        
        mat = read_log(datapath+"/../",f"log_keptdata_{datapath.split('/')[-1]}", 
                       return_mat=True)[:,2].astype(float)
        ind = argwhere((mat>=mindist) & (mat<maxdist))[:,0]
        lines = array(lines)[ind]
        
        lines_res += list(lines) 
        write_log(resultpath,f"log_keptdata_{datapath.split('/')[-1]}_{freq}", None, None, modif = lines_res)
        print("added")
        
def preprocessing(datapath,invpath,resultpath,ev,tmax,tmax_s,
                  channels,mindist,maxdist,param,
                  correct,snrlim = 5.):
    
    start = time()
    #snrlim = 10.
    
    st= Stream()
    for d in arange(mindist,maxdist).astype(int):
        try:
            st+=read(f"{datapath}/*d{d:02d}*.mseed")
        except : continue
    if maxdist<=2.:
        st = read(f"{datapath}/*.mseed")
    st.sort(["network","station"])
    names = [tr.id for tr in st]
    inv = Inventory()
    for name in names:
        try:
            inv+=read_inventory(f"{invpath}/{name}.xml")
        except:
            st.remove(st.select(id = name)[0])
            continue

    add_2_log(datapath,resultpath,mindist,maxdist,param)   
    
    inv, st = remove_nb_sta(inv,st,datapath,invpath,preprocess=resultpath)
    
    names = [tr.id for tr in st]
    inv = add_attribute_xml(inv,None,st,
                        datapath,invpath,returninv=True)
    if maxdist<=2.:
        # st.filter("bandpass",freqmin = .1, freqmax = 10, 
        #               corners = 4, zerophase=True)
        # for tr in st.select(station="DSZ"):
        #     st.remove(tr)
        st_hf = st.copy()
        st_hf.filter("bandpass",freqmin = 0.5, freqmax = 10, 
                     corners = 4, zerophase=True)
        for tr,tr_hf in zip(st,st_hf):
            tr.trim(tr.stats.starttime+30,
                    tr.stats.starttime+30+tmax_s)
            tr_hf.trim(tr_hf.stats.starttime+30,
                       tr_hf.stats.starttime+30+tmax_s)
    else :
        st.filter("bandpass",freqmin = 1/70, freqmax = 2, 
                     corners = 4, zerophase=True)
        st_hf = st.copy()
        st_hf.filter("bandpass",freqmin = 0.5, freqmax = 2, 
                     corners = 4, zerophase=True)
        for tr,tr_hf in zip(st,st_hf):
            tr.trim(tr.stats.starttime+2*60-60,
                    tr.stats.starttime+2*60+tmax)
            tr_hf.trim(tr_hf.stats.starttime+2*60-60,
                       tr_hf.stats.starttime+2*60+tmax)
    
    # for sti,freq in zip([st,st_hf],["BB","HF"]):
    #     indmax = array([argwhere(abs(tr.data)==abs(tr.data).max())[0,0]*tr.stats.delta 
    #                     for tr in sti]).astype(int)-60
    #     index = argwhere(indmax>=tmax-10)[:,0]
    #     modify_log(f"{resultpath}/",
    #                f"log_keptdata_{datapath.split('/')[-1]}_{freq}",
    #                array(names)[index],"large end amp")
    #     for i in index[::-1]:
    #         sti.remove(sti[i])
        
        if not correct:
            for sti,freq in zip([st,st_hf],["BB","HF"]):
                snr = signaltonoise(sti.copy(),60).round(2)
                indmax = array([argwhere(tr.data==tr.data.max())[0,0]*tr.stats.delta 
                                for tr in sti]).astype(int)-60
                amp = log10(array([abs(tr.data).max() for tr in sti]))
                
                if freq == "BB" : chan = [f"{cha}Z" for cha in channels]
                if freq == "HF" : chan = [f"{cha}Z" for cha in channels]
                ch_list = array([tr.stats.channel not in chan for tr in sti])
                index = argwhere((snr<snrlim)|
                                 (indmax>median(indmax)+25)|
                                 (abs(amp-median(amp))>=.5)|
                                 (ch_list==True))[:,0]
                modify_log(f"{resultpath}/",
                           f"log_keptdata_{datapath.split('/')[-1]}_{freq}",
                           array(names)[index],"bad shape signal")
                for i in index[::-1]:
                    sti.remove(sti[i])
    
        new_st = select_chan(st,channels)
        removed_names = [na for na in [tr.id for tr in st] 
                             if na not in [tr.id for tr in new_st]]
        modify_log(f"{resultpath}/",
                   f"log_keptdata_{datapath.split('/')[-1]}_BB",
                   removed_names,"not chosen chan")
        st = new_st
        
        new_st_hf = select_chan(st_hf,channels)
        removed_names = [na for na in [tr.id for tr in st_hf] 
                         if na not in [tr.id for tr in new_st_hf]]
        modify_log(f"{resultpath}/",
                   f"log_keptdata_{datapath.split('/')[-1]}_HF",
                   removed_names,"not chosen chan")
        st_hf = new_st_hf
    
    sys.stdout.write(f"{Fore.MAGENTA + Style.DIM}"
                    + f" > Data filtered "
                    + f"{Style.RESET_ALL}"
                    + f" (lasted{time()-start: 3.0f}s)")

    return st,st_hf,inv

# tr2 = Trace(data=abs(spec),header =
# Stats({"network":tr.stats.network,"station":tr.stats.station,"location":tr.stats.location,
#         "channel":"BHZspec","delta":freq[1]-freq[0],"sampling_rate":1/(freq[1]-freq[0]),
#         "startime":0,"npts":len(spec)
#         }) )

# tr2.plot()