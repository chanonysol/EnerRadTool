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

from data_utils import modify_log, write_log, read_log

from glob import glob
from time import time
from tqdm import tqdm
from multiprocessing import Pool
from colorama import Back, Fore, Style

from results_writing import Me, inv_Me, Mehf, inv_Mehf
    
import matplotlib.pyplot as plt
import matplotlib as mp

def show_correction(ener_sta_bb,ener_sta_hf,t,method,cor_bb,cor_hf,title):  
    ener_sta_bb_s = vstack([smooth_curve_mean(t, ener, dt=None)[1] for ener in ener_sta_bb])
    ener_sta_hf_s = vstack([smooth_curve_mean(t, ener, dt=None)[1] for ener in ener_sta_hf])
    
    Ebb = nanmean(ener_sta_bb,axis = 0)
    Ehf = nanmean(ener_sta_hf,axis = 0)
    Ebbs = nanmean(ener_sta_bb_s,axis = 0)
    Ehfs = nanmean(ener_sta_hf_s,axis = 0)
    
    (tr_tacer, tacer), (tr_lign, fst_lign, sec_lign) = find_tr(method,t,Ehfs)
    tr = nanmean([tr_tacer,tr_lign])
    
    mp.rcParams.update({'font.size': 14})
    
    fig,[ax,axhf] = plt.subplots(2,1,figsize = (8,12))
    #axm = ax.twinx(); axhfm = axhf.twinx()
    axm = ax.secondary_yaxis('right', functions=(Me,inv_Me))
    axhfm = axhf.secondary_yaxis('right', functions=(Mehf,inv_Mehf))

    ax.plot(t,Ebb,c="purple",ls="--",alpha=0.5)
    ax.plot(t,Ebbs,c="purple",ls="-")
    for i,e in enumerate(ener_sta_bb):
        ax.plot(t,e,c="plum" if i not in cor_bb else "gray",alpha=0.5,zorder=-10)

    axhf.plot(t,Ehf,c="red",ls="--",alpha=0.5)
    axhf.plot(t,Ehfs,c="red",ls="-")
    for i,e in enumerate(ener_sta_hf):
        axhf.plot(t,e,c="pink" if i not in cor_hf else "gray",alpha=0.5,zorder=-10)
        
    if tr_tacer!=-9999.0:
        axhf.plot(t,tacer/tacer.max()*Ehfs.max(),c="lightseagreen",ls="--",alpha=0.5,zorder = -10,label ="tacer")
    if tr_lign!=-9999.0:
        axhf.plot(t,fst_lign,c="sandybrown",ls="--",alpha=0.5,zorder = -10,label="trend")
        axhf.plot(t,sec_lign,c="sandybrown",ls="--",alpha=0.5,zorder = -10,label="trend")
        
    textTr = f"   Tr  = {tr:3.0f} s    "
    textBB = f"   EBB  = {Ebbs[int(tr)]:.2e} J  \n   MeBB = {Me(Ebbs[int(tr)]):.2f}   "
    textHF = f"   EHF  = {Ehfs[int(tr)]:.2e} J  \n   MeHF = {Me(Ehfs[int(tr)]/0.25):.2f}   "\
             + f"\nEhf/Tr3 = {Ehfs[int(tr)]/tr**3:.2e}"
    
    topbb = 10**(round(log10(Ebbs.max()),1)+.1)
    tophf = 10**(round(log10(Ehfs.max()),1)+.1)
    botbb = topbb/100
    bothf = tophf/100
    ax.set_ylim(botbb,topbb)
    axhf.set_ylim(bothf,tophf)
    #ax.set_ylim(bottom = topbb/1000)
    #axhf.set_ylim(bottom = tophf/1000)
    # axm.set_ylim(me(botbb),me(topbb))
    # axhfm.set_ylim(Me(bothf/0.25),Me(tophf/0.25))
    
    ax.set_ylabel(r"Cumulative $E_{BB}$ (J)")
    axhf.set_ylabel(r"Cumulative $E_{HF}$ (J)")
    axm.set_ylabel(r"Corresponding $Me_{BB}$")
    axhfm.set_ylabel(r"Corresponding $Me_{HF}$")
    axhf.set_xlabel("Time since first P-wave arrival (s)")
    
    for i,axi in enumerate([ax,axhf]): 
        axi.set_facecolor("w")
        axi.grid()
        axi.set_xlim(0,t.max())
        axi.axvline(tr,lw=2,c="k")
        
        if tr_tacer!=-9999:
            axi.axvline(tr_tacer,lw=.5,c="lightseagreen")
        if tr_lign!=-9999:
            axi.axvline(tr_lign,lw=.5,c="sandybrown")
        
        if i<1 :
            axi.text(t.max()-90, botbb*15, 
                     textBB, verticalalignment='top',
                     horizontalalignment='left',
                     bbox={'facecolor': 'plum', 'alpha': 0.6, 'pad': 10}, fontsize=12)
        else:
            axi.text(t.max()-90, bothf*20, 
                     textHF, verticalalignment='top',
                     horizontalalignment='left',
                     bbox={'facecolor': 'salmon', 'alpha': 0.6, 'pad': 10}, fontsize=12)
            axi.text(t.max()-75, bothf*32, 
                     textTr, verticalalignment='top',
                     horizontalalignment='left',
                     bbox={'facecolor': 'silver', 'alpha': 0.6, 'pad': 10}, fontsize=12)
            
            axi.text(tr_tacer-10,tophf*0.94,f"{tr_tacer:.0f}s",
                     backgroundcolor="w",c='lightseagreen')
            axi.text(tr_lign-5,tophf*0.94,f"{tr_lign:.0f}s",
                     backgroundcolor="w",c='sandybrown')
    plt.title(title)
    plt.show()

def correct(ener,st,datapath,resultpath,log,rt, plot=False):
    dev = abs(log10(ener/median(ener)))[-len(st):]
    ind = argwhere(dev>=2.)[:,0]
    # print(dev.round(1))
    if plot:
        reasons, names, dists, azs = read_log(resultpath,
            os.path.split(glob(f"{resultpath.replace(']','?').replace('[','?')}/log_keptdata_*{log}*")[-1])[-1][:-4],
            return_mat=True)[:,[0,1,2,3]].T
        ind_r = array([i for i,r in enumerate(reasons) if ((r.find("in processing _")!=-1)
                          or (r.find("REMAIN ITER")!=-1 and int(r[-1])<=rt))])
        names = names[ind_r]; reasons = reasons[ind_r]
        dists, azs = dists[ind_r].astype(float), azs[ind_r].astype(float)
    
        fig = plt.figure(figsize=(8,4))
        
        plt.scatter(azs, log10(ener/median(ener)),c=dists,cmap="BuGn",vmin=5,vmax=80,edgecolor="silver",label="kept")
        
        plt.scatter(azs[-len(st):][ind], log10(ener/median(ener))[-len(st):][ind],c=dists[-len(st):][ind],
                    cmap="BuGn",vmin=5,vmax=80,edgecolor="k",label=f"med dev {log} {rt}")
        plt.xlim(-1,361)
        #plt.ylim(-3,3)
        plt.legend()
        plt.grid()
        plt.show()
        plt.close()
    
    names = array([tr.id for tr in st])
    modify_log(f"{resultpath}/",
               f"log_keptdata_{datapath.split('/')[-1]}_{log}",
               names[ind],f"dev median {rt}")
    return ind

def correct_devlim(ener,st,datapath,resultpath,log,rt,devlim=1.,plot=False):
    dev = abs(log10(ener/mean(ener)))[-len(st):]
    ind = argwhere(dev>=devlim)[:,0]
    # print(dev.round(1))
    if plot :
        reasons, names, dists, azs = read_log(resultpath,
            os.path.split(glob(f"{resultpath.replace(']','?').replace('[','?')}/log_keptdata_*{log}*")[-1])[-1][:-4],
            return_mat=True)[:,[0,1,2,3]].T
        ind_r = array([i for i,r in enumerate(reasons) if ((r.find("in processing _")!=-1)
                          or (r.find("REMAIN ITER")!=-1 and int(r[-1])<=rt))])
        names = names[ind_r]; reasons = reasons[ind_r]
        dists, azs = dists[ind_r].astype(float), azs[ind_r].astype(float)
        
        
        
        fig = plt.figure(figsize=(8,4))
        plt.scatter(azs, log10(ener/mean(ener)),c=dists,cmap="BuGn",vmin=5,vmax=80,edgecolor="silver",label="kept")
        
        plt.scatter(azs[-len(st):][ind], log10(ener/mean(ener))[-len(st):][ind],c=dists[-len(st):][ind],
                    cmap="BuGn",vmin=5,vmax=80,edgecolor="k",label=f"limit dev {log} {rt}")
        plt.xlim(-1,361)
        plt.ylim(-2,2)
        plt.legend()
        plt.grid()
        plt.show()
        plt.close()

    names = array([tr.id for tr in st])
    # print(names[ind])
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
    
    # show_correction(ener_sta_bb,ener_sta_hf,arange(0,shape(ener_sta_bb)[1]),method,
    #                 correct(ener_sta_bb[:,-1],st_bb,datapath,resultpath,"BB",rt)+len(ener_sta_bb[:-len(st_bb)]),
    #                 correct(ener_sta_hf[:,-1],st_hf,datapath,resultpath,"HF",rt)+len(ener_sta_hf[:-len(st_hf)]),
    #                 f"median correction RT{rt}")
    
    for ener,st,log in zip([ener_sta_bb,ener_sta_hf],
                           [st_bb,st_hf],
                           ["BB","HF"]):
        cor = correct(ener[:,-1],st,datapath,resultpath,log,rt)
        
        if log == "BB": 
            ener_sta_bb = delete(ener_sta_bb, cor+len(ener[:-len(st)]), axis = 0)
        if log == "HF": 
            ener_sta_hf = delete(ener_sta_hf, cor+len(ener[:-len(st)]), axis = 0)
        for i in cor[::-1]:
            
            st.remove(st[i])
    
    new_Tr, new_Tr_tacer, new_Tr_lign = re_eval_time(ener_sta_hf,method)
    
    # show_correction(ener_sta_bb,ener_sta_hf,arange(0,shape(ener_sta_bb)[1]),method,
    #                 correct_devlim(ener_sta_bb[:,int(new_Tr)],st_bb,datapath,resultpath,"BB",rt)+len(ener_sta_bb[:-len(st_bb)]),
    #                 correct_devlim(ener_sta_hf[:,int(new_Tr)],st_hf,datapath,resultpath,"HF",rt)+len(ener_sta_hf[:-len(st_hf)]),
    #                 f"deviation correction RT{rt}")
    for ener,st,log in zip([ener_sta_bb,ener_sta_hf],
                           [st_bb,st_hf],
                           ["BB","HF"]):
        cor = correct_devlim(ener[:,int(new_Tr)],st,datapath,resultpath,log,rt,devlim)
        
        if log == "BB": 
            ener_sta_bb = delete(ener_sta_bb, cor+len(ener_sta_bb[:-len(st_bb)]), axis = 0)
        if log == "HF": 
            ener_sta_hf = delete(ener_sta_hf, cor+len(ener_sta_hf[:-len(st_hf)]), axis = 0)
        for i in cor[::-1]:
            # print(st[i])
            st.remove(st[i])
    
    new_Tr, new_Tr_tacer, new_Tr_lign = re_eval_time(ener_sta_hf,method)
    return ener_sta_bb,ener_sta_hf, st_bb,st_hf, new_Tr, new_Tr_tacer, new_Tr_lign
    

def do_corrections(ev, resultpath, datapath, param,
               ener_sta_bb, ener_sta_hf,
               st_bb, st_hf,
               results, rt=False):
    devlim = param.deviation_correction.threshold
    method = param.time_eval_method
    
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


