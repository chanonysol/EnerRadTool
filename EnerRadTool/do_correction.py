# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 20:47:45 2024

@author: solen
"""

import os
import random
import colorama
import sys
import shutil

from numpy import array, vstack, argwhere, log10, median,shape,\
    sin, cos, pi, arcsin, linspace, arange, mean, delete, nan, nanmean
from obspy import UTCDateTime
from ener_functions import trace_ener, smooth_curve_mean, calc_ener_rt
from results_writing import add_results, update_results, complete_from_template, read_results_sta, Me
from results_correction import redo_smooth,new_correction, write_index_cor, plot_correction_rectangle, plot_raw, plot_correction
from time_rupture_functions import find_tr, joint_tr

from data_utils import modify_log, write_log

from glob import glob
from time import time
from tqdm import tqdm
from multiprocessing import Pool
from colorama import Back, Fore, Style

def correct(ener,st,datapath,resultpath,log,rt):
    dev = abs(log10(ener/median(ener)))[-len(st):]
    ind = argwhere(dev>=2.)[:,0]
    names = array([tr.id for tr in st])
    modify_log(f"{resultpath}/",
               f"log_keptdata_{datapath.split('/')[-1]}_{log}",
               names[ind],f"dev median {rt}")
    return ind

def correct_devlim(ener,st,datapath,resultpath,log,rt,devlim=1.):
    dev = abs(log10(ener/mean(ener)))[-len(st):]
    ind = argwhere(dev>=devlim)[:,0]
    names = array([tr.id for tr in st])
    modify_log(f"{resultpath}/",
               f"log_keptdata_{datapath.split('/')[-1]}_{log}",
               names[ind],f"dev limit {rt}")
    return ind
    
def re_eval_time(ener,method):
    t = arange(0,shape(ener)[1])
    ener_s = vstack([smooth_curve_mean(t, e, dt=None)[1] for e in ener])
    (tr_tacer, tacer), (tr_lign, fst_lign, sec_lign) = find_tr(method,t,nanmean(ener_s,axis=0))
    return joint_tr(tr_tacer,tr_lign), tr_tacer, tr_lign
    
def correct_complex(ener_sta_bb,ener_sta_hf,
                    st_bb,st_hf,devlim,
                    datapath,resultpath,method,rt):
    
    for ener,st,log in zip([ener_sta_bb,ener_sta_hf],
                           [st_bb,st_hf],
                           ["BB","HF"]):
        cor = correct(ener[:,-1],st,datapath,resultpath,log,rt)
        
        if log == "BB": 
            ener_sta_bb = delete(ener_sta_bb, cor+len(ener[:-len(st)])-1, axis = 0)
        if log == "HF": 
            ener_sta_hf = delete(ener_sta_hf, cor+len(ener[:-len(st)])-1, axis = 0)
        for i in cor[::-1]:
            print(st[i])
            st.remove(st[i])
    
    new_Tr, new_Tr_tacer, new_Tr_lign = re_eval_time(ener_sta_hf,method)
    for ener,st,log in zip([ener_sta_bb,ener_sta_hf],
                           [st_bb,st_hf],
                           ["BB","HF"]):
        cor = correct_devlim(ener[:,int(new_Tr)],st,datapath,resultpath,log,rt,devlim)
        
        if log == "BB": 
            ener_sta_bb = delete(ener_sta_bb, cor+len(ener[:-len(st)]), axis = 0)
        if log == "HF": 
            ener_sta_hf = delete(ener_sta_hf, cor+len(ener[:-len(st)]), axis = 0)
        for i in cor[::-1]:
            print(st[i])
            st.remove(st[i])
    
    new_Tr, new_Tr_tacer, new_Tr_lign = re_eval_time(ener_sta_hf,method)

    return ener_sta_bb,ener_sta_hf, st_bb,st_hf, new_Tr, new_Tr_tacer, new_Tr_lign
    

def do_corrections(ev, resultpath, datapath, param,
               ener_sta_bb, ener_sta_hf,
               st_bb, st_hf,
               results, rt=False):
    devlim = param.deviation_correction.threshold
    method = param.time_eval_method
    #print(results["raw_smooth"])
    if isinstance(rt,int):
        ener_sta_bb,ener_sta_hf, \
            st_bb,st_hf, \
                new_Tr, new_Tr_tacer, new_Tr_lign = correct_complex(ener_sta_bb,ener_sta_hf,
                                                            st_bb,st_hf,devlim,
                                                            datapath,resultpath,method,rt)
        t = arange(0,shape(ener_sta_bb)[1])
        ener_sta_bb_s = vstack([smooth_curve_mean(t, ener, dt=None)[1] for ener in ener_sta_bb])
        ener_sta_hf_s = vstack([smooth_curve_mean(t, ener, dt=None)[1] for ener in ener_sta_hf])
        
        (tr_tacer, _), (tr_lign, _,_) = find_tr(method,t,mean(ener_sta_hf,axis=0))
        tr = joint_tr(tr_tacer,tr_lign)
        
        ener_bb,ener_hf = mean(ener_sta_bb,axis=0),mean(ener_sta_hf,axis=0)
        ener_bb_s,ener_hf_s = mean(ener_sta_bb_s,axis=0),mean(ener_sta_hf_s,axis=0)
        
        from numpy import nan
        t_list = [tr_tacer, tr_lign]
        t_lists = [new_Tr_tacer,new_Tr_lign]
        
        dev = [abs(ener_bb[tr] - ener_bb[t]) for t in t_list if t]
        dev= mean(dev/ener_bb[tr]*100) #percent
        
        dev_s = [abs(ener_bb_s[new_Tr] - ener_bb_s[t]) for t in t_lists if t]
        dev_s= mean(dev_s/ener_bb_s[new_Tr]*100) #percent
        
        dev_hf = [abs(ener_hf[tr] - ener_hf[t]) for t in t_list if t]
        dev_hf= mean(dev_hf/ener_hf[tr]*100) #percent
        
        dev_hf_s = [abs(ener_hf_s[new_Tr] - ener_hf_s[t]) for t in t_lists if t]
        dev_hf_s= mean(dev_hf_s/ener_hf_s[new_Tr]*100) #percent
        
        if ener_sta_bb[:,tr].size == 1:
            add_results(results, ev, corrected = [nan, 
                                                  mean(ener_sta_hf[:,tr]),
                                                  dev, dev_hf,
                                                  tr_tacer, tr_lign] )
            add_results(results, ev, corrected_smooth = [nan, 
                                                         mean(ener_sta_hf_s[:,new_Tr]),
                                                         dev_s, dev_hf_s,
                                                         new_Tr_tacer, new_Tr_lign] )
        else :
            add_results(results, ev, corrected = [mean(ener_sta_bb[:,tr]),
                                                  mean(ener_sta_hf[:,tr]),
                                                  dev, dev_hf,
                                                  tr_tacer, tr_lign] )
            add_results(results, ev, corrected_smooth = [mean(ener_sta_bb_s[:,new_Tr]), 
                                                         mean(ener_sta_hf_s[:,new_Tr]),
                                                         dev_s, dev_hf_s,
                                                         new_Tr_tacer, new_Tr_lign])
            
        # print(results["corrected_smooth"])
        return st_bb, st_hf

    else:
        sys.stdout.write(f"{Back.RED + Fore.BLACK}>> IN DEVELOPMENT "
                         +f"{Style.RESET_ALL + Fore.RED} CODE NOT DONE YET"
                         f"{Style.RESET_ALL}\n")


