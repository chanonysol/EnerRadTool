# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:50:28 2024

@author: solen

Description :
    Look for events on IRIS, GCMT etc within set parameters
"""
import os
import sys

FILE_PATH = os.path.dirname(os.path.realpath(__file__)).replace('\\', '/')

DIR=f"{FILE_PATH}/../2.data_functions"
sys.path.insert(0, DIR)

from results_writing import read_results_sta, read_sta_log, read_results, Me, inv_Me, Mehf, inv_Mehf
from ener_functions import smooth_curve_mean
from data_utils import read_log
import sys
from colorama import Back,Style,Fore
from numpy import vstack, array, argwhere, mean, concatenate, isnan, log10,\
    shape, nanmean, nanmedian, arange, ndarray
from obspy.core.util import AttribDict
from glob import glob
from time_rupture_functions import find_tr

from do_correction import correct_complex
    
import matplotlib.pyplot as plt
import matplotlib as mp


def read_results_file(filepath,correct = True):
    with open(filepath,"r") as file:
        lines = file.readlines()
    ind = argwhere(array(lines) == "#smoothed\n")[:,0]
    if correct : ind = ind[-1]
    else : ind = ind[0]
    lines = lines[ind+1:ind+9]
    vals = [i for i in concatenate([i.strip().split("; ") for i in lines]) 
            if i!=""]
    dic = {}
    for i in vals:
        i = i.strip()
        if i[0]=="(":
            i = i[1:-1]
        n,v = i.split("= ")
        if v[-1] in ["s","J","%"]:
            v=v[:-1]
        dic[n.strip()]=float(v)
        
    if isnan(dic["Tr_tacer"]):
        method = "trend"
    elif isnan(dic["Tr_lines"]):
        method = "tacer"
    else :
        method = "both"
    dic['method']=method
    return AttribDict(dic)    

def plot_correct_staflux(resultpath, rt= False, correct = False, savefile= False, ext =".png", 
                         show=False,redo=False):
    if not show and not savefile :
        return
    sys.stdout.write(f"\n{Fore.GREEN + Style.BRIGHT}"
                     +f"> Showing stations energy flux - iteration {rt}")
    
    if os.path.exists(f"{resultpath}/staflux_{'raw' if not correct else 'corrected'}{rt}{ext}") and not redo:
        sys.stdout.write(f"\n{Fore.GREEN + Style.BRIGHT}"
                     +" ... Already Done ")
        return
       
    t,ener_sta_bb,names_bb = read_results_sta(f"{resultpath}/ener_all_bb.dat")
    t,ener_sta_hf,names_hf = read_results_sta(f"{resultpath}/ener_all_hf.dat")
    
    if len(shape(ener_sta_bb)) == 1 :
        ener_sta_bb = array([ener_sta_bb])
    if len(shape(ener_sta_hf)) == 1 :
        ener_sta_hf = array([ener_sta_hf])
        
    ener_sta_bb = vstack([smooth_curve_mean(t, ener, dt=None)[1] for ener in ener_sta_bb])
    ener_sta_hf = vstack([smooth_curve_mean(t, ener, dt=None)[1] for ener in ener_sta_hf])
    
    
    res = read_results_file(glob(f"{resultpath.replace("[","?").replace("]","?")}"
                                  +f"/results*{'' if isinstance(rt,bool) else rt}.txt")[-1], 
                            correct = correct) 
       
    
    reasons_bb, names_bb_log,dist_bb, az_bb = read_log(resultpath,
                                          os.path.split(glob(f"{resultpath.replace(']','?').replace('[','?')}/log_keptdata_*BB*")[-1])[-1][:-4],
                                          return_mat=True)[:,[0,1,2,3]].T
    ind = argwhere(reasons_bb!='not chosen chan')[:,0]
    names_bb_log = names_bb_log[ind]; reasons_bb = reasons_bb[ind]
    dist_bb, az_bb = dist_bb[ind].astype(float), az_bb[ind].astype(float)
    
    # indsort = array([argwhere(names_bb_log==i)[0,0] for i in names_bb])
    # names_bb_log = names_bb_log[indsort]; reasons_bb = reasons_bb[indsort]
    # dist_bb, az_bb = dist_bb[indsort], az_bb[indsort]
    
    
    ind_bb_m = array([i for i,r in enumerate(reasons_bb) 
                      if ((r.find("dev limit")!=-1 and int(r[-1])==rt)
                      or (r.find("REMAIN ITER")!=-1 and int(r[-1])<=rt))])
    ind_bb_f = array([i for i,r in enumerate(reasons_bb) 
                      if r.find("REMAIN ITER")!=-1 and int(r[-1])<=rt])
    
    rem_bb_m = array([i for i,r in enumerate(reasons_bb) 
                      if r.find("dev median")!=-1 and int(r[-1])==rt])
    rem_bb_f = array([i for i,r in enumerate(reasons_bb) 
                      if r.find("dev limit")!=-1 and int(r[-1])==rt])
    
    if rem_bb_f.size == 0 : rem_bb_f=list(rem_bb_f)
    if rem_bb_m.size == 0 : rem_bb_m=list(rem_bb_m)
    
    reasons_hf, names_hf_log, dist_hf, az_hf = read_log(resultpath,
                                          os.path.split(glob(f"{resultpath.replace(']','?').replace('[','?')}/log_keptdata_*HF*")[-1])[-1][:-4],
                                          return_mat=True)[:,[0,1,2,3]].T
    ind = argwhere(reasons_hf!='not chosen chan')[:,0]
    names_hf_log = names_hf_log[ind]; reasons_hf = reasons_hf[ind]
    dist_hf, az_hf = dist_hf[ind].astype(float), az_hf[ind].astype(float)
    
    indsort = array([argwhere(names_hf_log==i)[0,0] for i in names_hf])
    names_hf_log = names_hf_log[indsort]; reasons_hf = reasons_hf[indsort]
    dist_hf, az_hf = dist_hf[indsort], az_hf[indsort]
    
    
    ind_hf_m = array([i for i,r in enumerate(reasons_hf) 
                      if ((r.find("dev limit")!=-1 and int(r[-1])==rt)
                      or (r.find("REMAIN ITER")!=-1 and int(r[-1])<=rt))])
    ind_hf_f = array([i for i,r in enumerate(reasons_hf) 
                      if r.find("REMAIN ITER")!=-1 and int(r[-1])<=rt])
    
    rem_hf_m = array([i for i,r in enumerate(reasons_hf) 
                      if r.find("dev median")!=-1 and int(r[-1])==rt])
    
    rem_hf_f = array([i for i,r in enumerate(reasons_hf) 
                      if r.find("dev limit")!=-1 and int(r[-1])==rt])
    
    if rem_hf_f.size == 0 : rem_hf_f=list(rem_hf_f)
    if rem_hf_m.size == 0 : rem_hf_m=list(rem_hf_m)
    
    Ebb_r = nanmean(ener_sta_bb,axis = 0)
    Ebb_r_med = nanmedian(ener_sta_bb,axis = 0)
    Ebb_m = nanmean(ener_sta_bb[ind_bb_m],axis = 0)
    Ebb_f = nanmean(ener_sta_bb[ind_bb_f],axis = 0)
    
    Ehf_r = nanmean(ener_sta_hf,axis = 0)
    Ehf_r_med = nanmedian(ener_sta_hf,axis = 0)
    Ehf_m = nanmean(ener_sta_hf[ind_hf_m],axis = 0)
    Ehf_f = nanmean(ener_sta_hf[ind_hf_f],axis = 0)
    
    (ttac_r, _), (ttrend_r, _, _) = find_tr(res.method,t,Ehf_r)
    tr_r = nanmean((ttac_r,ttrend_r))
    (ttac_m, tacer_m), (ttrend_m, fstline_m, secline_m) = find_tr(res.method,t,Ehf_m)
    tr_m = nanmean((ttac_m,ttrend_m))
    (ttac_f, tacer), (ttrend_f, fstline, secline) = find_tr(res.method,t,Ehf_f)
    tr_f = nanmean((ttac_f,ttrend_f))
    
    mp.rcParams.update({'font.size': 14})
    #============================================================================
    fig,[[ax_r,ax_m,ax_f],[axhf_r,axhf_m,axhf_f]] = plt.subplots(2,3,figsize = (10*3,12))
    #axm = ax.twinx(); axhfm = axhf.twinx()
    axm_r = ax_r.secondary_yaxis('right', functions=(Me,inv_Me))
    axm_m = ax_m.secondary_yaxis('right', functions=(Me,inv_Me))
    # axm_f = ax_f.secondary_yaxis('right', functions=(Me,inv_Me))
    axhfm_r = axhf_r.secondary_yaxis('right', functions=(Mehf,inv_Mehf))
    axhfm_m = axhf_m.secondary_yaxis('right', functions=(Mehf,inv_Mehf))
    # axhfm_f = axhf_f.secondary_yaxis('right', functions=(Mehf,inv_Mehf))

    for axi,E,Emed,ind,rem in zip([ax_r,ax_m],[Ebb_r,Ebb_m],[Ebb_r_med,Ebb_f],
                     [ind_bb_m,ind_bb_f],[rem_bb_m,rem_bb_f]):
        axi.plot(t,E,c="purple",ls="-",label="BB ener")
        if isinstance(Emed,ndarray):
            axi.plot(t,Emed,c="purple",ls="--",label="BB corrected")
        for i,e in enumerate(ener_sta_bb[ind]):
            axi.plot(t,e,c="plum",ls="-",alpha=0.4,zorder =-10,label = "kept" if i==0 else None)
        for i,e in enumerate(ener_sta_bb[rem]):
            axi.plot(t,e,c="gray",ls="-",alpha=0.3,zorder =-10,label = "rejected" if i==0 else None)

    for axi,E,Emed,ind,rem in zip([axhf_r,axhf_m],[Ehf_r,Ehf_m],[Ehf_r_med,Ehf_f],
                     [ind_hf_m,ind_hf_f],[rem_hf_m,rem_hf_f]):
        axi.plot(t,E,c="r",ls="-",label="HF ener")
        if isinstance(Emed,ndarray):
            axi.plot(t,Emed,c="r",ls="--",label="HF corrected")
        for i,e in enumerate(ener_sta_hf[ind]):
            axi.plot(t,e,c="pink",ls="-",alpha=0.6,zorder =-10,label = "kept" if i==0 else None)
        for i,e in enumerate(ener_sta_hf[rem]):
            axi.plot(t,e,c="gray",ls="-",alpha=0.3,zorder =-10,label = "rejected" if i==0 else None)

    for axi,axi_hf,Ebb,Ehf in zip([ax_r,ax_m],[axhf_r,axhf_m],
                                [Ebb_r,Ebb_m],[Ehf_r,Ehf_m]):
        topbb = 10**(round(log10(Ebb.max()),1)+.1)
        tophf = 10**(round(log10(Ehf.max()),1)+.1)
        botbb = topbb/100
        bothf = tophf/100

        axi.set_ylim(botbb,topbb)
        axi_hf.set_ylim(bothf,tophf)

    ax_r.set_ylabel(r"Cumulative $E_{BB}$ (J)")
    for axi in [axhf_r,axhf_m]:
        axi.set_xlabel("Time since first P-wave arrival (s)")
    axhf_r.set_ylabel(r"Cumulative $E_{HF}$ (J)")
    axm_m.set_ylabel(r"Corresponding $Me_{BB}$")
    axhfm_m.set_ylabel(r"Corresponding $Me_{HF}$")

    for i,(axi,tr) in enumerate(zip([ax_r,ax_m,axhf_r,axhf_m],
                                    [t[-1],tr_m,t[-1],tr_m])): 
        axi.set_facecolor("w")
        axi.grid()
        axi.set_xlim(0,t.max())
        axi.axvline(tr,lw=2,c="k",label="Tr")
    axhf_m.axvline(tr_f,lw=2,c="k",ls="--",label="Tr corrected")
    ax_m.axvline(tr_f,lw=2,c="k",ls="--",label="Tr corrected")
    
    for ind,c,lab in zip([rem_bb_m,rem_bb_f,ind_bb_f],["r","k","silver"],["first corr","sec corr","kept"]):
        ax_f.scatter(az_bb[ind],log10(ener_sta_bb[ind,int(tr_m)]/Ebb_m[int(tr_m)]),c=dist_bb[ind],
                     cmap="BuGn",vmin=5,vmax=80,edgecolor=c,label=lab)
    for ind,c,lab in zip([rem_hf_m,rem_hf_f,ind_hf_f],["r","k","silver"],["first corr","sec corr","kept"]):
        sc = axhf_f.scatter(az_hf[ind],log10(ener_sta_hf[ind,int(tr_m)]/Ehf_m[int(tr_m)]),c=dist_hf[ind],
                     cmap="BuGn",vmin=5,vmax=80,edgecolor=c,label=lab)
    #axhf_f.legend();ax_f.legend()
    for axi in [ax_f,axhf_f]:
        axi.set_ylabel(r"Station epicentral distance (°)")
        axi.set_ylim(-3,3)
        axi.set_xlim(-1,361)
        axi.grid()
    axhf_f.set_xlabel(r"Station azimuth (°)")
    fig.colorbar(sc ,ax=[ax_f,axhf_f], label = r"$log_{10}(E_{station}/E)$")
    # plt.tight_layout()  
    
    for axi in [ax_m,ax_r,axhf_m,axhf_r,axhf_f,ax_f]:
        axi.legend(ncol=3,fontsize="small")
    
    if savefile :
        plt.savefig(f"{resultpath}/staflux_{'raw' if not correct else 'corrected'}{rt}{ext}",
                    dpi = 300, transparent=True)
    #plt.show()
    plt.close()
    sys.stdout.write(f"\n{Fore.GREEN + Style.BRIGHT}"
                     +" ... Done ")

def plot_growth(resultpath, correct = False, rt= False,savefile= False, ext =".png", show=False,redo=False):
    if not show and not savefile :
        return
    sys.stdout.write(f"\n{Fore.GREEN + Style.BRIGHT}"
                     +f"> Showing {'corrected ' if correct else ''}results graphics - iteration {rt}")
    
    if os.path.exists(f"{resultpath}/EnerOverTime_{'raw' if not correct else 'corrected'}{rt}{ext}") and not redo:
        sys.stdout.write(f"\n{Fore.GREEN + Style.BRIGHT}"
                     +" ... Already Done ")
        return
       
    t,ener_sta_bb,names_bb = read_results_sta(f"{resultpath}/ener_all_bb.dat")
    t,ener_sta_hf,names_hf = read_results_sta(f"{resultpath}/ener_all_hf.dat")
    
    if len(shape(ener_sta_bb)) == 1 :
        ener_sta_bb = array([ener_sta_bb])
    if len(shape(ener_sta_hf)) == 1 :
        ener_sta_hf = array([ener_sta_hf])
    
    res = read_results_file(glob(f"{resultpath.replace("[","?").replace("]","?")}"
                                 +f"/results*{'' if isinstance(rt,bool) else rt}.txt")[-1], 
                            correct = correct) 
       
    if correct or isinstance(rt,int):
        stalog_bb = read_log(resultpath,
                              os.path.split(glob(f"{resultpath.replace(']','?').replace('[','?')}/log_keptdata_*BB*")[-1])[-1][:-4],
                              return_mat=True)
        keep_names_bb=stalog_bb[:,1][argwhere(array([any(i.find(f"{'REMAIN ITER' if correct and j else ''}{' ' if isinstance(j,int) else ''}{j}")!=-1 
                                                         for j in (range(rt+1) if not isinstance(rt,bool) else " ")) 
                                                     for i in stalog_bb[:,0]]))[:,0]]
        stalog_hf = read_log(resultpath,
                              os.path.split(glob(f"{resultpath.replace(']','?').replace('[','?')}/log_keptdata_*HF*")[-1])[-1][:-4],
                              return_mat=True)
        keep_names_hf=stalog_hf[:,1][argwhere(array([any(i.find(f"{'REMAIN ITER' if correct and j else ''}{' ' if isinstance(j,int) else ''}{j}")!=-1 
                                                         for j in (range(rt+1) if not isinstance(rt,bool) else " ")) 
                                                     for i in stalog_hf[:,0]]))[:,0]]
        
        Lbb = [i for i,n in enumerate(names_bb) if n in keep_names_bb]
        Lbb.sort()
        Lhf = [i for i,n in enumerate(names_hf) if n in keep_names_hf]
        Lhf.sort()
        
        ener_sta_bb = ener_sta_bb[Lbb,:]
        names_bb = names_bb[Lbb]
        ener_sta_hf = ener_sta_hf[Lhf,:]
        names_hf = names_hf[Lhf]
         
    ener_sta_bb_s = vstack([smooth_curve_mean(t, ener, dt=None)[1] for ener in ener_sta_bb])
    ener_sta_hf_s = vstack([smooth_curve_mean(t, ener, dt=None)[1] for ener in ener_sta_hf])
    
    Ebb = nanmean(ener_sta_bb,axis = 0)
    Ehf = nanmean(ener_sta_hf,axis = 0)
    Ebbs = nanmean(ener_sta_bb_s,axis = 0)
    Ehfs = nanmean(ener_sta_hf_s,axis = 0)
    
    (ttac, tacer), (ttrend, fst_lign, sec_lign) = find_tr(res.method,t,Ehfs)
    
    mp.rcParams.update({'font.size': 14})
    
    fig,[ax,axhf] = plt.subplots(2,1,figsize = (8,12))
    #axm = ax.twinx(); axhfm = axhf.twinx()
    axm = ax.secondary_yaxis('right', functions=(Me,inv_Me))
    axhfm = axhf.secondary_yaxis('right', functions=(Mehf,inv_Mehf))

    ax.plot(t,Ebb,c="purple",ls="--",alpha=0.5)
    
    # for ener in ener_sta_bb_s:
    #     ax.plot(t,ener,c="silver",ls="--",alpha=0.2,zorder=-20)
    
    ax.plot(t,Ebbs,c="purple",ls="-")

    axhf.plot(t,Ehf,c="red",ls="--",alpha=0.5)
    axhf.plot(t,Ehfs,c="red",ls="-")
    if res.method in ["tacer","both"] and res.Tr_tacer!=-9999.0:
        axhf.plot(t,tacer/tacer.max()*Ehfs.max(),c="lightseagreen",ls="--",alpha=0.5,zorder = -10,label ="tacer")
    if res.method in ["lines","both"] and res.Tr_lines!=-9999.0:
        axhf.plot(t,fst_lign,c="sandybrown",ls="--",alpha=0.5,zorder = -10,label="trend")
        axhf.plot(t,sec_lign,c="sandybrown",ls="--",alpha=0.5,zorder = -10,label="trend")
        
    textTr = f"   Tr  = {res.Tr:3.0f} s    "
    textBB = f"   EBB  = {res.Ebb:.2e} J  \n   MeBB = {res.MeBB:.2f}   "
    textHF = f"   EHF  = {res.Ehf:.2e} J  \n   MeHF = {res.MeHF:.2f}   "\
             + f"\nEhf/Tr3 = {list(res.values())[-2]:.2e}"
    
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
        axi.axvline(res.Tr,lw=2,c="k")
        
        if res.Tr_tacer!=-9999:
            axi.axvline(res.Tr_tacer,lw=.5,c="lightseagreen")
        if res.Tr_lines!=-9999:
            axi.axvline(res.Tr_lines,lw=.5,c="sandybrown")
        
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
            
            axi.text(res.Tr_tacer-10,tophf*0.94,f"{res.Tr_tacer:.0f}s",
                     backgroundcolor="w",c='lightseagreen')
            axi.text(res.Tr_lines-5,tophf*0.94,f"{res.Tr_lines:.0f}s",
                     backgroundcolor="w",c='sandybrown')
    if savefile :
        plt.savefig(f"{resultpath}/EnerOverTime_{'raw' if not correct else 'corrected'}{rt}{ext}",
                    dpi = 300, transparent=True)
    #plt.show()
    plt.close()
    sys.stdout.write(f"\n{Fore.GREEN + Style.BRIGHT}"
                     +" ... Done ")

from numpy import dtype, loadtxt, log10, linspace
from generate_event import moment
from data_utils import name_reg_event
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import ListedColormap
from matplotlib import cm
from matplotlib import colormaps

def mw(M):
    return (2/3*log10(M)-10.73).round(1)

def result_in_cat(resultpath,ev,correct = False, savefile= False, ext =".png", show=False,redo=False):
    if not show and not savefile :
        return
    sys.stdout.write(f"\n{Fore.GREEN + Style.BRIGHT}"
                     +"> Showing other results in sector")
    if os.path.exists(f"{resultpath}/Res2Cat_{'raw' if not correct else 'corrected'}{ext}") and not redo :
        sys.stdout.write(f"\n{Fore.GREEN + Style.BRIGHT}"
                     +" ... Already Done ")
        return
    
    dty = dtype([('date',"U25"),('lon',"f8"),('lat',"f8"),("reg","U25"),("sreg","U25"),
                 ('Ebb',"f8"),('Ehf',"f8"),('M0',"f8"),
                 ('MeBB',"f8"),('MeHF',"f8"),('delta',"f8"),('Tr',"f8")])
    mat = loadtxt("EnerRadTool/5.plot_functions/newcat.dat",dtype=dty,comments="#")  
    regs,subregs = mat["reg"],mat["sreg"]
    r,sr = name_reg_event(ev.lat,ev.lon) 
    res = read_results_file(glob(f"{resultpath.replace("[","?").replace("]","?")}"
                                 +"/results*")[-1], 
                            correct = correct)  
    mp.rcParams.update({'font.size': 14})
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot()
    ax.set_facecolor("w")
    ind = argwhere((regs==r))[:,0]
    ax.scatter(log10(mat["M0"])[ind],log10(mat["Ebb"])[ind],s=10,c="gainsboro",
                edgecolor="gainsboro",label="Catalog results")
    if r and sr:
        ind = argwhere((regs==r)&(subregs==sr))[:,0]
        plt.scatter(log10(mat["M0"])[ind],log10(mat["Ebb"])[ind],s=40,c=mat["delta"][ind],
                    cmap="inferno",alpha=0.6,edgecolor="gray",vmin=-6, vmax = -3,
                    label="The region")
    
    ax.scatter(log10(moment(ev.moment,ev.exponent)),
                log10(res.Ebb),
                s=200,marker ="*",c=res.theta,
                cmap="magma",edgecolor="k",vmin=-6, vmax = -3,label="This event")
    
    x = array([log10(mat["M0"]).min()-.2,log10(mat["M0"]).max()+.2]).round(1)
    y = array([log10(mat["Ebb"]).min()-.5,log10(mat["Ebb"]).max()+.5]).round(1)
    for i in range(3,7):
        ax.plot(x,x-i,"--",color="silver",zorder = -5)
        xi = 16.5 + i
        if xi >= 21:
            xi = 20.7
        ax.text(xi+.05, xi+.05-i, r"$\theta=$"+f"{-i:.0f}",
                rotation=30,
                horizontalalignment="left", verticalalignment="bottom",
                color="gray",bbox={'facecolor': 'w', 'edgecolor':'w','alpha': 0.5},
                zorder=-10)
    ax.set_xlim(x)
    ax.set_ylim(y)
    ax.set_ylabel(r"BB Energy - $log_{10}(Ebb)$ $(J)$")
    ax.set_xlabel(r"Seismic moment - $log_{10}(M0)$ $(N.M)$")
    ax.legend()
    
    ax2 = ax.twiny()
    mx = mw(10**(x+7))
    ax2.set_xlim(mx)
    ax2.grid(c = "gainsboro")
    ax.grid(axis="y",c = "gainsboro")
    
    if savefile :
        plt.savefig(f"{resultpath}/Res2Cat_{'raw' if not correct else 'corrected'}{ext}",
                    dpi = 300, transparent=True)
    #plt.show()
    plt.close()
    sys.stdout.write(f"\n{Fore.GREEN + Style.BRIGHT}"
                     +" ... Done ")



def plot_theta_depth_apparent(resultpath,ev,correct = False,
                              savefile= False, ext =".png", show=False,redo=False):
    if not show and not savefile :
        return
    sys.stdout.write(f"\n{Fore.GREEN + Style.BRIGHT}"
                     +"> Showing apparent stress comparison")
    if os.path.exists(f"{resultpath}/apparentStress_{'raw' if not correct else 'corrected'}{ext}") and not redo:
        sys.stdout.write(f"\n{Fore.GREEN + Style.BRIGHT}"
                     +" ... Already Done ")
        return
    de = array([3.,15.,15.,24.4,24.4,40.,60.,80.])
    mu = array([266,266,441,441,682,680,671,674])*0.1*10**3 #kbar to MPa
    
    res = read_results_file(glob(f"{resultpath.replace("[","?").replace("]","?")}"
                                 +"/results*")[-1], 
                            correct = correct) 
    M0 = moment(ev.moment,ev.exponent)
    
    xdata = array([ev.depth]); ydata = array([log10(res.Ebb*(1+res.dEbb/100)/M0)]).round(2)
    xerror = array([[2.5],[2.5]])
    yerror = array([[abs(log10(1-res.dEbb/100).round(2))],[log10(1+res.dEbb/100).round(2)]])

    mp.rcParams.update({'font.size': 12})
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot()
    ax.set_facecolor("w")
    ax.yaxis.set_major_locator(MultipleLocator(.5))
    ax.yaxis.set_minor_locator(MultipleLocator(.1))
    ax.xaxis.set_minor_locator(MultipleLocator(5.))

    cmap = ListedColormap(colormaps["copper"](linspace(0., 0.8, 3)))
    cols = cmap.colors
    cols = [cols[0],cols[1],cols[1],cols[1],cols[2]]
    
    for i,(val,ls,name) in enumerate(zip([6.4,2.4,1.5,0.5,0.3],
                                         ["-","-","--",":","-"],
                                      ["intraoceanic \nS-Slip",
                                       "intraslab \nN-Slip",
                                       "cross-cutting \nN-Slip",
                                       "reactivated \nN-Slip",
                                       "interplate \nR-Slip"])):
        plt.plot(de,log10(val/mu).round(2),ls=ls,c =cols[i])
        ax.text(63,log10(val/mu).round(2)[-1], name,
                horizontalalignment="center", verticalalignment="center",
                color=cols[i],bbox={'facecolor': 'w', 'edgecolor':'w','alpha': 0.8})

    plt.plot(xdata,ydata,"*k",markersize=10,zorder=10,label="This event")
    errorboxes = [Rectangle((x - xe[0], y - ye[0]), xe.sum(), ye.sum())
                  for x, y, xe, ye in zip(xdata, ydata, xerror.T, yerror.T)]
    pc = PatchCollection(errorboxes, facecolor='silver', alpha=0.5,
                         edgecolor="w",ls = "--",zorder = 5)
    ax.add_collection(pc)
    artists = ax.errorbar(xdata, ydata, xerr=xerror, yerr=yerror,
                          fmt='none', ecolor='k',label="deviation")

    plt.legend()
    plt.grid(axis = "y",which ="major",c="gray")
    plt.grid(axis = "x",which ="major",c="silver")
    plt.grid(which = "minor",c="whitesmoke")
    plt.ylabel(r"Scaled energy - $log_{10}(Ebb/M_0)$")
    plt.xlabel("depth from PREM model")
    plt.title("Estimate of scaled energy \nbased on Choy et al (2006) apparent stress\n and PREM rigidity modulus")
    plt.xlim(0,70)
    plt.ylim(-5.8,-3.4)
    if savefile :
        plt.savefig(f"{resultpath}/apparentStress_{'raw' if not correct else 'corrected'}{ext}",
                    dpi = 300,transparent = True)
    #plt.show()
    plt.close()
    sys.stdout.write(f"\n{Fore.GREEN + Style.BRIGHT}"
                     +" ... Done ")
    
def plot_s_int(resultpath,ev,savefile= False, ext =".png",show=False,redo=False):
    if not show and not savefile :
        return
    sys.stdout.write(f"\n{Fore.GREEN + Style.BRIGHT}"
                     +"> S waves integrals")
    if os.path.exists(f"{resultpath}/integral_contribs{ext}") and not redo:
        sys.stdout.write(f"\n{Fore.GREEN + Style.BRIGHT}"
                     +" ... Already Done ")
        return
    
    from numpy import dtype
    file = f"{resultpath}/ener_int.dat"
    freq = loadtxt(file)[0];tot=loadtxt(file)[-1]
    mat = loadtxt(file)[1:-1]
    
    file_sta = f"{resultpath}/station_ener.dat"
    datatype=open(file_sta,"r").readlines()[0][1:].split()
    vals = ["f8"]*len(datatype)
    vals[0] = "U25";vals[5] = "U25"
    dty = dtype([(d,v) for d,v in zip (datatype, vals)])
    stas = loadtxt(file_sta,comments="#",dtype=dty)    
    
    cols = {"A":"firebrick","B":"chocolate","C":"teal",
            "D":"indigo","E":"indigo","gener":"olivegreen",
            "O":"firebrick"}
    symb = {"A":"^","B":"D","C":"s","D":"H","E":"H","gener":"o","O":"x"}
    
    fig,ax = plt.subplots(1,figsize=(12,8))
    ax2 = ax.twinx()
    
    if stas.size==1:
        n = stas["site"].item(); c,s = cols[n],symb[n]
        if n=="O":n="None"
        ax.semilogy(freq,mat[0],c=c,marker=s,markerfacecolor="w",
                    markevery=len(freq)//30,markersize = 5,lw=0.5,
                    alpha=0.8,label= f"{n}")
        ax2.semilogy(freq,mat[0],c=c,marker=s,alpha=0)
    else:
        for n,c,s in zip(list(cols.keys()),list(cols.values()),list(symb.values())):
            ind = argwhere(stas["site"]==n)
            if n=="O":n="None"
            for i,j in enumerate(ind):
                ax.semilogy(freq,mat[j][0],c=c,marker=s,markerfacecolor="w",
                            markevery=len(freq)//30,markersize = 5,lw=0.5,
                            alpha=0.8,label= f"{n if i==0 else ''}")
                ax2.semilogy(freq,mat[j][0],c=c,marker=s,alpha=0)
            
    ax.semilogy(freq,tot,c="k",lw=2,label="total")
    ax2.semilogy(freq,tot,c="k",alpha=0)
    
    for axi in [ax,ax2]:
        axi.set_ylim(10**(log10(mat[:,10].min()).round(1)),10**(log10(tot.max()).round(1)+.2))
        axi.set_xlim(0,10)
    ax2.yaxis.set_major_formatter(lambda x, pos: Me(x).round(1))
    
    ax.set_ylabel("Energy (J)")
    ax2.set_ylabel(r"Energy magnitude $M_E^S$")
    ax.set_xlabel("Frequency (Hz)")
    
    ax.grid(c="gainsboro")
    ax.legend(loc="upper left", title = "site class")
    
    plt.title("Energy spectral integrals - total and weighed stations")
    if savefile:
        plt.savefig(f"{resultpath}/integral_contribs{ext}",dpi=200)
    #plt.show()
    plt.close()
    sys.stdout.write(f"\n{Fore.GREEN + Style.BRIGHT}"
                     +" ... Done ")
    
def plot_s_map(resultpath,ev,savefile= False, ext =".png",show=False,redo=False):
    if not show and not savefile :
        return
    sys.stdout.write(f"\n{Fore.GREEN + Style.BRIGHT}"
                     +"> S waves map")
    if os.path.exists(f"{resultpath}/stations_map{ext}") and not redo:
        sys.stdout.write(f"\n{Fore.GREEN + Style.BRIGHT}"
                     +" ... Already Done ")   
        return
    
    import pygmt
    import io
    from obspy.geodetics.base import degrees2kilometers,kilometer2degrees
    
    file = f"{resultpath}/ener_int.dat"
    mat = loadtxt(file)[1:-1,-1]
    
    file_sta = f"{resultpath}/station_ener.dat"
    datatype=open(file_sta,"r").readlines()[0][1:].split()
    vals = ["f8"]*len(datatype)
    vals[0] = "U25";vals[5] = "U25"
    dty = dtype([(d,v) for d,v in zip (datatype, vals)])
    stas = loadtxt(file_sta,comments="#",dtype=dty)    
    
    path = "EnerRadTool/2.data_functions/spec_decay.dat"
    dty = dtype([('station',"U25"),('k',"f8"),('dk',"f8")])
    k0sta = loadtxt(path,comments="#",dtype=dty)
    
    cols = {"A":"firebrick","B":"coral","C":"cadetblue1","D":"darkorchid","E":"darkorchid",
            "gener":"darkolivegreen1","O":"gray50"}
    cols_t = {"A":"firebrick4","B":"coral4","C":"deepskyblue4",
              "D":"darkorchid4","E":"darkorchid4",
            "gener":"darkolivegreen4","O":"gray30"}
    symb = {"A":"t","B":"d","C":"s","D":"h","E":"h","gener":"c","O":"i"}
    
    fig = pygmt.Figure()
    # # projection azimuthale equidistant lon0/lat0/distance max/resolution
    fig.basemap(region="g", projection=f"E%f/%f/{int(kilometer2degrees(450))}/13c" %
                (ev.lon,ev.lat), frame=["af10", f"WSne"])
    fig.coast(shorelines="0.15p,black", land=None,resolution='h')  # coastlines
    # PLOT
    site = "None"; c2 ="white"
    if stas.size==1 : 
        n = stas["station"].item().split(".")[1]
        site = stas["site"].item(); c,s = cols[site],symb[site]
        lo,la = stas["lon"].item(),stas["lat"].item()
        if n in k0sta["station"]:
            c2 = "black"
        fig.plot(x=lo,y=la,fill=f"{c}",
                  style=f"{s}.4c", pen=f"0.3p,{c2}")
        fig.text(x=lo,y=la+0.125,
                  text=n,
                  fill = f"{c}",font="7p")
    else : 
        names = [n.split(".")[1] for n in stas["station"]]
        for n,site,lo,la in zip(names,stas["site"],stas["lon"],stas["lat"]):
            c,s = cols[site],symb[site]
            if n in k0sta["station"]:
                c2 = "black"
            fig.plot(x=lo,y=la,fill=f"{c}",style=f"{s}.5c", pen=f"0.3p,{c2}")
        for n,site,lo,la,e in zip(names,stas["site"],stas["lon"],stas["lat"],
                                  (mat/sum(mat)*100).round(2)):
            c,s,t = cols[site],symb[site],cols_t[site]
            fig.text(x=lo,y=la+0.125,text=n,fill = f"{c}",font="7p")
            fig.text(x=lo+0.3,y=la+.125,text=f"{e:.1f}%",pen=f".2p,{t}",
                     fill="white",font=f"7p,{t}")
    #fig.grdimage(grid=grid, cmap="elevgray.cpt", verbose="q")  # plot elevation
    legend = io.StringIO( "\n".join([f"S 0.4 {s} .3 {c} .5p,white .8 {k if k!='O' else 'None'} \nG .2" 
                                    for s,c,k in zip(symb.values(),cols.values(),cols.keys())
                                    if k in stas["site"]]+
                                    [f"S 0.4 t .3 white .5p,black .8 'k evaluated'"]))
    fig.legend(spec=legend,position="jBR+o.5c/.5c",box="+p.2p+gwhite")
    fig.plot(x=ev.lon,y=ev.lat,fill="black",
              style="a.5c", pen='1p,black')
    if savefile:
        fig.savefig(f"{resultpath}/stations_map{ext}",dpi=200)
    fig.show()
    sys.stdout.write(f"\n{Fore.GREEN + Style.BRIGHT}"
                     +" ... Done ")
