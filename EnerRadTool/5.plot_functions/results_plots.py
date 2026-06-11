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

from results_writing import read_results_sta, read_sta_log, read_results, Me
from ener_functions import smooth_curve_mean
from data_utils import read_log
import sys
from colorama import Back,Style,Fore
from numpy import vstack, array, argwhere, mean, concatenate, isnan, log10, shape, nanmean
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

def plot_growth(resultpath, correct = False, rt= False,savefile= False, ext =".png", redo=False):
    sys.stdout.write(f"\n{Fore.GREEN + Style.BRIGHT}"
                     +"> Showing results graphics")
    
    if os.path.exists(f"{resultpath}/EnerOverTime_{'raw' if not correct else 'corrected'}{ext}") and not redo:
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
                              glob(f"{resultpath.replace(']','?').replace('[','?')}/log_keptdata_*BB*")[-1].split("\\")[-1][:-4],
                              return_mat=True)
        keep_names_bb=stalog_bb[:,1][argwhere(array([any(i.find(f"{'REMAIN ITER' if correct and j else ''}{' ' if isinstance(j,int) else ''}{j}")!=-1 
                                                         for j in (range(rt+1) if not isinstance(rt,bool) else " ")) 
                                                     for i in stalog_bb[:,0]]))[:,0]]
        stalog_hf = read_log(resultpath,
                              glob(f"{resultpath.replace(']','?').replace('[','?')}/log_keptdata_*HF*")[-1].split("\\")[-1][:-4],
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
    Ehfs = smooth_curve_mean(t, Ehf, dt=None)[1]
    Ebbs = smooth_curve_mean(t, Ebb, dt=None)[1]
    
    (ttac, tacer), (ttrend, fst_lign, sec_lign) = find_tr(res.method,t,Ehfs)
    
    mp.rcParams.update({'font.size': 14})
    
    fig,[ax,axhf] = plt.subplots(2,1,figsize = (8,12))
    axm = ax.twinx(); axhfm = axhf.twinx()

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
    axm.set_ylim(Me(botbb),Me(topbb))
    axhfm.set_ylim(Me(bothf/0.13),Me(tophf/0.13))
    
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
            
            axi.text(res.Tr_tacer-10,-bothf*3,
                     f"{res.Tr_tacer:.0f}s",c='lightseagreen')
            axi.text(res.Tr_lines+2,-bothf*3,
                     f"{res.Tr_lines:.0f}s",c='sandybrown')
    if savefile :
        plt.savefig(f"{resultpath}/EnerOverTime_{'raw' if not correct else 'corrected'}{ext}",
                    dpi = 300, transparent=True)
    plt.show()
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

def result_in_cat(resultpath,ev,correct = False, savefile= False, ext =".png", redo=False):
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
    
    x = array([log10(mat["M0"]).min(),log10(mat["M0"]).max()]).round(1)
    y = array([log10(mat["Ebb"]).min()-.5,log10(mat["Ebb"]).max()+.5]).round(0)
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
    plt.show()    
    sys.stdout.write(f"\n{Fore.GREEN + Style.BRIGHT}"
                     +" ... Done ")



def plot_theta_depth_apparent(resultpath,ev,correct = False,
                              savefile= False, ext =".png", redo=False):
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
    plt.show()
    sys.stdout.write(f"\n{Fore.GREEN + Style.BRIGHT}"
                     +" ... Done ")
    
def plot_s_int(resultpath,ev,savefile= False, ext =".png",redo=False):
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
    plt.show()
    sys.stdout.write(f"\n{Fore.GREEN + Style.BRIGHT}"
                     +" ... Done ")
    
def plot_s_map(resultpath,ev,savefile= False, ext =".png",redo=False): 
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
    
    path = "spec_decay.dat"
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
    fig.basemap(region="g", projection=f"E%f/%f/{int(kilometer2degrees(250))}/13c" %
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
    fig.legend(spec=legend,position="jBL+o.5c/.5c",box="+p.2p+gwhite")
    fig.plot(x=ev.lon,y=ev.lat,fill="black",
              style="a.5c", pen='1p,black')
    if savefile:
        fig.savefig(f"{resultpath}/stations_map{ext}",dpi=200)
    fig.show()
    sys.stdout.write(f"\n{Fore.GREEN + Style.BRIGHT}"
                     +" ... Done ")
