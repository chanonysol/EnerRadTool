# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:50:28 2024

@author: solen

Description :
    Based on a list of events, create all needed events to constitute a
    catalog
"""
from numpy import argwhere, delete, median, log10, array, arange, mean,\
    concatenate, reshape, nan, empty
from ener_functions import smooth_curve_mean, find_ener
from time_rupture_functions import joint_tr, find_tr
from results_writing import read_sta_log, Me
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def plot_corrected_results(ener_sta_bb, ener_sta_hf, Tr, param,**kwargs):
    if "correction" not in kwargs.keys():
        index_del_bb, index_join_bb, index_del_hf, index_join_hf, \
            Tr_del, Tr_cor, ener_sta_bb_s, ener_sta_hf_s = \
                correct(ener_sta_bb, ener_sta_hf, Tr, param)
    else :
        index_del_bb, index_join_bb, index_del_hf, index_join_hf,\
           Tr_del, Tr_cor, ener_sta_bb_s, ener_sta_hf_s = kwargs['correction']

    fig,[[ax,axh],[ax2,ax2h],[ax4,ax4h]] = plt.subplots(3,2,figsize = (16,15))

    #BB
    label_ind = [j for j in arange(len(ener_sta_bb_s)).astype(int) if j not in index_join_bb][0]

    m_ener = mean(ener_sta_bb_s,axis=0)
    for i,e_contrib in enumerate(ener_sta_bb_s):
        c = 'plum'
        if i == label_ind :
            l="station energy flux"
            hdl, = ax.plot(e_contrib,color=c,alpha = 0.4, label=l)
        if i in index_join_bb:
            c= "mediumorchid"; l="deviatoric energy flux"
            hdl_cor, = ax.plot(e_contrib,color=c,alpha = 0.4, label=l)
        if i in index_del_bb:
            c= "mediumpurple"; l="over/under sized energy flux"
            hdl_del, = ax.plot(e_contrib,color=c,alpha = 0.4, label=l)
        else :
            ax.plot(e_contrib,color=c)
    ener, = ax.plot(m_ener,color="indigo", label = "mean energy", zorder = 10)
    ax.legend(handles = [hdl,hdl_del,hdl_cor,ener])
    ax.axvline(Tr,c='k',ls = ":")

    for i,e_contrib in enumerate(ener_sta_bb_s):
        c = 'plum'
        if i in index_join_bb:
            c= "mediumorchid"
        if i in index_del_bb: continue
        ax2.plot(e_contrib,color=c,alpha = 0.4)
    ax2.plot(mean(delete(ener_sta_bb_s,index_del_bb,axis=0),axis=0),color="indigo")
    ax2.axvline(Tr,c='k',ls = ":")
    ax2.axvline(Tr_del,c='k',ls = "-.")

    for e_contrib in delete(ener_sta_bb_s,index_join_bb,axis=0):
        ax4.plot(e_contrib,color='plum',alpha = 0.4)
    ax4.plot(mean(delete(ener_sta_bb_s,index_join_bb,axis=0),axis=0),color="indigo")
    tr0 = ax4.axvline(Tr,c='k',ls = ":")
    tr1 = ax4.axvline(Tr_del,c='k',ls = "-.")
    tr2 = ax4.axvline(Tr_cor,c='k')

    #HF
    m_ener = mean(ener_sta_hf_s,axis=0)
    for i,e_contrib in enumerate(ener_sta_hf_s):
        c = 'pink'
        if i == label_ind :
            l="station energy flux"
            hdl, = axh.plot(e_contrib,color=c,alpha = 0.4, label=l)
        if i in index_join_bb:
            c= "lightcoral"; l="deviatoric energy flux"
            hdl_cor, = axh.plot(e_contrib,color=c,alpha = 0.4, label=l)
        if i in index_del_bb:
            c= "crimson"; l="over/under sized energy flux"
            hdl_del, = axh.plot(e_contrib,color=c,alpha = 0.4, label=l)
        else :
            axh.plot(e_contrib,color=c)
    ener, = axh.plot(m_ener,color="darkred", label = "mean energy", zorder = 10)
    axh.legend(handles = [hdl,hdl_del,hdl_cor,ener])
    axh.axvline(Tr,c='k',ls = ":")

    for i,e_contrib in enumerate(ener_sta_hf_s):
        c = 'pink'
        if i in index_join_hf: c= "lightcoral"
        if i in index_del_hf: continue
        ax2h.plot(e_contrib,color=c,alpha = 0.4)
    ax2h.plot(mean(delete(ener_sta_hf_s,index_del_hf,axis=0),axis=0),color="darkred")
    ax2h.axvline(Tr,c='k',ls = ":")
    ax2h.axvline(Tr_del,c='k',ls = "-.")

    for e_contrib in delete(ener_sta_hf_s,index_join_hf,axis=0):
        ax4h.plot(e_contrib,color='pink',alpha = 0.4)
    ax4h.plot(mean(delete(ener_sta_hf_s,index_join_hf,axis=0),axis=0),color="darkred")
    tr0 = ax4h.axvline(Tr,c='k',ls = ":")
    tr1 = ax4h.axvline(Tr_del,c='k',ls = "-.")
    tr2 = ax4h.axvline(Tr_cor,c='k')
    ax4h.legend(handles=[tr0,tr1,tr2],labels=["first time estimate",
                                            "corrected time estimate",
                                            "final time estimate"])

    #deco
    for i,axi in enumerate([ax,ax2,ax4,axh,ax2h,ax4h]):
        #axi.grid(color = 'lightgray', ls =":")
        axi.set_xlim(0,param.time_max)
        axi.xaxis.set_major_locator(MultipleLocator(100))
        axi.xaxis.set_minor_locator(MultipleLocator(20))
        if i<3 :
            val_max = max(mean(delete(ener_sta_bb_s,index_join_bb,axis=0),axis=0))
        else :
            val_max = max(mean(delete(ener_sta_hf_s,index_join_hf,axis=0),axis=0))
        exp = int(log10(val_max))
        num = round(val_max/float(f"1e{exp}"),0)
        if i not in [0,3]:
            axi.set_ylim(0,float(f"{num+1}e{exp}"))
        else :
            axi.set_ylim(0,float(f"1e{exp+1}"))
        axi.yaxis.set_major_locator(MultipleLocator(float(f"{num}e{exp}")))
        axi.yaxis.set_minor_locator(MultipleLocator(float(f"{num/4}e{exp}")))
        axi.grid(axis="x", which="major")
        axi.grid(axis="x", which="minor", c='lightgray', ls =":")
        axi.grid(axis="y", which="minor", c='lightgray', ls =":")
        axi.grid(axis="y", which="major")

        #axi.set_yticks([])

    return index_del_bb, index_join_bb, index_del_hf, index_join_hf,\
        Tr_del, Tr_cor, ener_sta_bb_s, ener_sta_hf_s



from matplotlib.patches import Rectangle
from glob import glob

def redo_smooth(ener_sta):
    ener_sta_bb, ener_sta_hf = ener_sta
    if ener_sta_bb.size==0:
        ener_sta_bb_s = empty((0,1))
    else:
        ener_sta_bb_s = array([smooth_curve_mean(arange(len(ener)),ener, dt=10)[1]
                                for ener in ener_sta_bb])

    ener_sta_hf_s = array([smooth_curve_mean(arange(len(ener)),ener)[1]
                           for ener in ener_sta_hf])
    return (ener_sta_bb_s,ener_sta_hf_s)

def new_correction(ener_sta,Tr,param):
    devlim = param.deviation_correction.threshold
    ener_sta_bb_s, ener_sta_hf_s = redo_smooth(ener_sta)

    #first HF correction
    ener = ener_sta_hf_s[:,Tr]
    index_del_hf = argwhere(abs(log10(ener/median(ener)))>1.5)[:,0]
    #new Tr
    ener_sta_del = delete(ener_sta_hf_s,index_del_hf,axis=0)
    _, smooth_results_del = find_ener(ener_sta_del,ener_sta_del,param)
    (_,ener_hf_del,_,((tr_tacer_del,_),(tr_lign_del,_,_))) =  smooth_results_del
    Tr_del = joint_tr(tr_tacer_del,tr_lign_del)
    #new HF ener
    new_ener_hf = ener_sta_hf_s[:,Tr_del]
    new_mean_hf = mean(new_ener_hf)
    #choose to keep HF
    index_keep_hf = argwhere(abs(log10(new_ener_hf/new_mean_hf))<= devlim)[:,0]
    #final TR
    ener_sta_cor = ener_sta_hf_s[index_keep_hf,:]
    _, smooth_results_cor = find_ener(ener_sta_cor,ener_sta_cor,param)
    (_,ener_hf_cor,_,((tr_tacer_cor,_),(tr_lign_cor,_,_))) =  smooth_results_cor
    Tr_cor = joint_tr(tr_tacer_cor,tr_lign_cor)

    #new bb at Tr_cor
    ener = ener_sta_bb_s[:,Tr_cor]
    #first bb correction
    index_del_bb = argwhere(abs(log10(ener/median(ener)))>1.5)[:,0]
    #new mean estimate
    new_ener_bb = delete(ener_sta_bb_s,index_del_bb,axis=0)[:,Tr_cor]
    new_mean_bb = mean(new_ener_bb)
    #final correction BB
    index_keep_bb = argwhere(abs(log10(ener/new_mean_bb))<= devlim)[:,0]

    return index_del_hf,index_del_bb,index_keep_hf,index_keep_bb,Tr_del,Tr_cor, smooth_results_del, smooth_results_cor

def plot_correction_rectangle(path, correction, Tr, ener_sta, devlim):
    st_bb_info = []; st_hf_info = []
    foldlist = glob(path+"/../"+path.split("/")[-1][:-7]+"*")
    for pat in foldlist:
        if int(pat.split("/")[-1][-5:-3]) >= int(path.split("/")[-1][-2:]) :
            continue
        #print(pat.split("/")[-1])
        _,sta = read_sta_log(pat)
        info = [list(sta[k].values())[1:3] for k in sta.keys()]
        st_bb_info+=info
        _,sta = read_sta_log(pat,hf=True)
        info = [list(sta[k].values())[1:3] for k in sta.keys()]
        st_hf_info+=info
    st_bb_info = array(st_bb_info)
    st_hf_info = array(st_hf_info)

    ener_sta_bb_s, ener_sta_hf_s  = redo_smooth(ener_sta)
    index_del_hf,index_del_bb,index_keep_hf,index_keep_bb,Tr_del,Tr_cor,_,_ = correction

    ###################HF
    fig, axs = plt.subplots(1,3,figsize = (11,3.5))

    x,y = st_hf_info.T
    ener = ener_sta_hf_s[:,Tr]; dev =log10(ener/mean(ener))

    for ax in axs :
        ax.xaxis.set_major_locator(MultipleLocator(90))
        ax.xaxis.set_minor_locator(MultipleLocator(90/4))
        ax.grid(axis="x", which="major",c='lightgray')
        ax.set_xlim(-10,370)

    axs[0].set_title(f"Original estimation\n Tr = {Tr}s\n Ehf = {mean(ener_sta_hf_s[:,Tr]):.2e}J")


    axs[0].scatter(x,log10(ener/mean(ener)),s=30,c=dev,vmin=-1,vmax=1,cmap = "RdGy",edgecolor = 'w',alpha=0.7)
    axs[0].axhline(0,c="silver",zorder=-3, label = "original mean")

    axs[0].axhline(log10(median(ener)/mean(ener)),c="r",zorder=-3,label = "median data set")
    axs[0].axhline(log10(median(ener)/mean(ener))-1.5,ls=":",c="r",zorder=-3)
    axs[0].axhline(log10(median(ener)/mean(ener))+1.5,ls=":",c="r",zorder=-3)
    errorbox = Rectangle((-10,log10(median(ener)/mean(ener))-1.5), 384, 3, facecolor='salmon', alpha=0.1)
    axs[0].add_patch(errorbox)

    axs[0].scatter(x[index_del_hf],dev[index_del_hf],s=30,c=dev[index_del_hf],vmin=-1,vmax=1,edgecolor = 'r', cmap = "RdGy", label = "dev to median > 1.5")

    ener = ener_sta_hf_s[:,Tr_del]; dev =log10(ener/mean(ener))
    new_mean = mean(delete(ener,index_del_hf))
    axs[1].scatter(x,log10(ener/mean(ener)),s=30,c=dev,vmin=-1,vmax=1,cmap = "RdGy",edgecolor = 'w',alpha=0.7)

    axs[1].axhline(log10(new_mean/mean(ener)),c="k",zorder=-3, label = "new mean")
    axs[1].axhline(log10(new_mean/mean(ener))-devlim,ls=":",c="k",zorder=-3)
    axs[1].axhline(log10(new_mean/mean(ener))+devlim,ls=":",c="k",zorder=-3)

    errorbox = Rectangle((-10,log10(new_mean/mean(ener))-devlim), 384, devlim*2, facecolor='lime', alpha=0.1)
    axs[1].add_patch(errorbox)

    axs[1].scatter(x[index_keep_hf],dev[index_keep_hf],s=30,c=dev[index_keep_hf],vmin=-1,vmax=1,cmap = "RdGy",edgecolor = 'k',label = "keep")
    axs[1].set_title(f"Intermediate estimation\n new Tr = {Tr_del}s\n inter Ehf = {mean(delete(ener,index_del_hf)):.2e}J")


    ener = ener_sta_hf_s[:,Tr_cor]; dev =log10(ener/mean(ener))
    scat = axs[2].scatter(x[index_keep_hf],y[index_keep_hf],s=30,c=dev[index_keep_hf],vmin=-1,vmax=1,cmap = "RdGy",edgecolor = 'k',label = "keep")
    axs[2].set_title(f"Final estimation\n new Tr = {Tr_cor}s\n new Ehf = {mean(ener[index_keep_hf]):.2e}J")
    axs[2].yaxis.set_major_locator(MultipleLocator(20))
    axs[2].yaxis.set_minor_locator(MultipleLocator(5))
    axs[2].set_ylim(3,82)
    axs[2].set_ylabel("epicentral distance (°)")
    axs[2].set_ylabel("azimuth (°)")
    axs[2].grid(axis="y", which="major",c='lightgray',zorder = -5)
    axs[2].grid(axis="y", which="minor",c='whitesmoke',zorder = -5)

    for ax in axs:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        if ax == axs[2] : continue

        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  fancybox=True, shadow=True, ncol=1)
    cax = fig.add_axes([box.x0, box.y0 -0.1,box.width, 0.05])
    plt.colorbar(scat,cax=cax,orientation = "horizontal", label = "Deviation to corrected energy")
    plt.savefig(f"{path}/correct_hf.png",dpi=300)
    ###############################################BB

    fig, axs = plt.subplots(1,3,figsize = (11,3.5))

    x,y = st_bb_info.T
    ener = ener_sta_bb_s[:,Tr_cor]; dev =log10(ener/mean(ener))

    for ax in axs :
        ax.xaxis.set_major_locator(MultipleLocator(90))
        ax.xaxis.set_minor_locator(MultipleLocator(90/4))
        ax.grid(axis="x", which="major",c='lightgray')
        ax.set_xlim(-10,370)
    for ax in axs[:2]:
        scat = ax.scatter(x,log10(ener/mean(ener)),s=30,c=dev,vmin=-1,vmax=1,cmap = "RdGy",edgecolor = 'w',alpha=0.7)

    axs[1].set_title(f"Intermediate estimation\n new Tr = {Tr_cor}s\n inter Ebb = {mean(delete(ener,index_del_bb)):.2e}J")
    axs[0].set_title(f"Original estimation\n Tr = {Tr}s\n Ebb = {mean(ener_sta_bb_s[:,Tr]):.2e}J")


    axs[0].axhline(0,c="silver",zorder=-3, label = "original mean")

    axs[0].axhline(log10(median(ener)/mean(ener)),c="r",zorder=-3,label = "median data set")
    axs[0].axhline(log10(median(ener)/mean(ener))-1.5,ls=":",c="r",zorder=-3)
    axs[0].axhline(log10(median(ener)/mean(ener))+1.5,ls=":",c="r",zorder=-3)
    errorbox = Rectangle((-10,log10(median(ener)/mean(ener))-1.5), 384, 3, facecolor='salmon', alpha=0.1)
    axs[0].add_patch(errorbox)

    axs[0].scatter(x[index_del_bb],dev[index_del_bb],s=30,c=dev[index_del_bb],vmin=-1,vmax=1,edgecolor = 'r', cmap = "RdGy", label = "dev to median > 1.5")

    new_mean = mean(delete(ener,index_del_bb))
    axs[1].axhline(log10(new_mean/mean(ener)),c="k",zorder=-3, label = "new mean")
    axs[1].axhline(log10(new_mean/mean(ener))-devlim,ls=":",c="k",zorder=-3)
    axs[1].axhline(log10(new_mean/mean(ener))+devlim,ls=":",c="k",zorder=-3)

    errorbox = Rectangle((-10,log10(new_mean/mean(ener))-devlim), 384, devlim*2, facecolor='lime', alpha=0.1)
    axs[1].add_patch(errorbox)

    axs[1].scatter(x[index_keep_bb],dev[index_keep_bb],s=30,c=dev[index_keep_bb],vmin=-1,vmax=1,cmap = "RdGy",edgecolor = 'k',label = "keep")

    dev =log10(ener/new_mean)
    scat = axs[2].scatter(x[index_keep_bb],y[index_keep_bb],s=30,c=dev[index_keep_bb],vmin=-1,vmax=1,cmap = "RdGy",edgecolor = 'k',label = "keep", zorder=5)
    axs[2].set_title(f"Final estimation\n new Tr = {Tr_cor}s\n new Ebb = {mean(ener[index_keep_bb]):.2e}J")
    axs[2].yaxis.set_major_locator(MultipleLocator(20))
    axs[2].yaxis.set_minor_locator(MultipleLocator(5))
    axs[2].set_ylim(3,82)
    axs[2].set_ylabel("epicentral distance (°)")
    axs[2].set_ylabel("azimuth (°)")
    axs[2].grid(axis="y", which="major",c='lightgray',zorder = -15)
    axs[2].grid(axis="y", which="minor",c='whitesmoke',zorder = -15)

    for ax in axs:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                          box.width, box.height * 0.9])
        if ax == axs[2] : continue
        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  fancybox=True, shadow=True, ncol=1)
    cax = fig.add_axes([box.x0, box.y0 -0.1,box.width, 0.05])
    plt.colorbar(scat,cax=cax,orientation = "horizontal", label = "Deviation to corrected energy")
    plt.savefig(f"{path}/correct_bb.png",dpi=300)

def write_index_cor(file,Tr,correction,names,ener_sta):
    index_del_hf,index_del_bb,index_keep_hf,index_keep_bb,Tr_del,Tr_cor,_,_ = correction
    with open(file,"w") as f:
        f.write("# BB\n")
        #f.write(f"#index del BB : \n")
        for i in index_del_bb:
            f.write(f"{i:3.0f} {names[0][i]:15s} {ener_sta[0][i,Tr_cor]:.2e}\n")
        f.write("# HF\n")
        #f.write(f"\n#index del HF : \n")
        for i in index_del_hf:
            f.write(f"{i:3.0f} {names[1][i]:15s} {ener_sta[1][i,Tr]:.2e}\n")

    with open(file.replace("correction","keep"),"w") as f:
        f.write("# BB\n")
        #f.write(f"\n#index keep BB : \n")
        for i in index_keep_bb:
            f.write(f"{i:3.0f} {names[0][i]:15s} {ener_sta[0][i,Tr_cor]:.2e}\n")
        f.write("# HF\n")
        #f.write(f"\n#index keep HF : \n")
        for i in index_keep_hf:
            f.write(f"{i:3.0f} {names[1][i]:15s} {ener_sta[1][i,Tr_del]:.2e}\n")

# =============================================================================
# Results plots
# =============================================================================
def plot_raw(resultpath,param,ener_sta):
    ener_sta_bb_s, ener_sta_hf_s  = redo_smooth(ener_sta)
    ener_sta_bb,ener_sta_hf = ener_sta

    t = arange(0,len(ener_sta_hf_s.T))
    eghf = mean(ener_sta_hf,axis=0)
    eghf_s = mean(ener_sta_hf_s,axis=0)
    if ener_sta_bb.size==0:
        egbb = empty((0,1))
        egbb_s = empty((0,1))
    else :
        egbb = mean(ener_sta_bb,axis=0)
        egbb_s = mean(ener_sta_bb_s,axis=0)
    (tr_tacer, tacer), (tr_ligns, fst_lign, sec_lign) = find_tr(param.time_eval_method,t,eghf_s)
    tr_mean = joint_tr(tr_tacer,tr_ligns)

    fig,[ax4,ax4h] = plt.subplots(2,1,figsize = (8,12))

    smo, = ax4.plot(egbb_s,color="indigo",lw=3, label= "smoothed E growth")
    raw, = ax4.plot(egbb,"-.",color="magenta",lw=1, label = "E growth")
    ax4.legend(handles = [raw,smo])

    alpha=0.4
    smo, = ax4h.plot(eghf_s,color="darkred",lw=3, label= "smoothed E growth")
    raw, = ax4h.plot(eghf,"-.",color="red",lw=1, label = "E growth")

    if param.time_eval_method in ["both","trend"]:
        ligns, = ax4h.plot(t,fst_lign,":",color="sandybrown",alpha=alpha, label = 'trend')
        ax4h.plot(t,sec_lign,":",color="sandybrown",alpha=alpha)
        ax4h.axvline(tr_ligns,c='orange',alpha=alpha)
        exp = int(log10(max(eghf_s)))
        ax4h.text(tr_ligns-3,float(f"-0.25e{exp}"),f"{tr_ligns}s",c='orange')
    if param.time_eval_method in ["both","tacer"]:
        tac, = ax4h.plot(tacer/tacer.max()*eghf.max(),"-.",color="lightblue",alpha=alpha, label = 'tacer')
        ax4h.axvline(tr_tacer,c='lightseagreen',alpha=alpha)
        exp = int(log10(max(eghf_s)))
        ax4h.text(tr_tacer-3,float(f"-0.25e{exp}"),f"{tr_tacer}s",c='lightseagreen')

    textTr = f"   Tr  = {tr_mean:3.0f} s    "
    if ener_sta_bb.size==0:
        textBB = f"   EBB  = {nan:.2e} J  \n   MeBB = {nan:.2f}   "
    else :
        textBB = f"   EBB  = {egbb_s[tr_mean]:.2e} J  \n   MeBB = {Me(egbb_s[tr_mean]):.2f}   "
    textHF = f"   EHF  = {eghf_s[tr_mean]:.2e} J  \n   MeHF = {Me(eghf_s[tr_mean]/0.13):.2f}   "\
             + f"\nEhf/Tr3 = {eghf_s[tr_mean]/tr_mean**3:.2e}"

    if param.time_eval_method in ["tacer"]: ax4h.legend(handles = [tac,raw,smo],loc="lower center",ncol=2)
    if param.time_eval_method in ["trend"]: ax4h.legend(handles = [ligns,raw,smo],loc="lower center",ncol=2)
    if param.time_eval_method in ["both"]: ax4h.legend(handles = [ligns,tac,raw,smo],loc="lower center",ncol=2)

    for i,axi in enumerate([ax4,ax4h]):
        #axi.grid(color = 'lightgray', ls =":")
        axi.axvline(tr_mean,c='k',lw=2)
        axi.set_xlim(0,t.max())
        axi.xaxis.set_major_locator(MultipleLocator(100))
        axi.xaxis.set_minor_locator(MultipleLocator(20))
        if i<1 :
            if egbb_s.size==0 : val_max = 1
            else : val_max = max(egbb_s)
        else :
            val_max = max(eghf_s)
        exp = int(log10(val_max))
        num = round(val_max/float(f"1e{exp}"),0)
        axi.set_ylim(0,float(f"{num+1}e{exp}"))
        axi.yaxis.set_major_locator(MultipleLocator(float(f"{num}e{exp}")))
        axi.yaxis.set_minor_locator(MultipleLocator(float(f"{num/4}e{exp}")))
        axi.grid(axis="x", which="major")
        axi.grid(axis="x", which="minor", c='lightgray', ls =":")
        axi.grid(axis="y", which="minor", c='lightgray', ls =":")
        axi.grid(axis="y", which="major")

        if i<1 :
            axi.text(t.max()-90, float(f"{egbb_s[tr_mean]*1.3:.2e}"), textBB, verticalalignment='top',
                     horizontalalignment='left',
                     bbox={'facecolor': 'plum', 'alpha': 0.6, 'pad': 10}, fontsize=11)
        else:
            axi.text(t.max()-90, float(f"{eghf_s[tr_mean]:.2e}"), textHF, verticalalignment='top',
                     horizontalalignment='left',
                     bbox={'facecolor': 'salmon', 'alpha': 0.6, 'pad': 10}, fontsize=11)
            axi.text(t.max()-75, float(f"{eghf_s[tr_mean]*1.5:.2e}"), textTr, verticalalignment='top',
                     horizontalalignment='left',
                     bbox={'facecolor': 'silver', 'alpha': 0.6, 'pad': 10}, fontsize=11)

    #deco
    ax4h.set_xlabel('time since first P-waves(s)')
    ax4h.set_ylabel('HF energy (J)')
    ax4.set_ylabel('BB energy (J)')
    ax4.set_title('Energy growth uncorrected')
    plt.savefig(f"{resultpath}/E_growth.png")




def plot_correction(resultpath,param,ener_sta,correction):
    ener_sta_bb_s, ener_sta_hf_s  = redo_smooth(ener_sta)
    ener_sta_bb,ener_sta_hf = ener_sta
    index_del_hf,index_del_bb,index_keep_hf,index_keep_bb,Tr_del,Tr_cor, smooth_results_del,smooth_results_cor= correction

    t = arange(0,len(ener_sta_hf_s.T))
    eghf = mean(ener_sta_hf[index_keep_hf,:],axis=0)
    eghf_s = mean(ener_sta_hf_s[index_keep_hf,:],axis=0)
    if ener_sta_bb.size==0:
        egbb = empty((0,1))
        egbb_s = empty((0,1))
    else :
        egbb = mean(ener_sta_bb[index_keep_bb,:],axis=0)
        egbb_s = mean(ener_sta_bb_s[index_keep_bb,:],axis=0)
    (_,_,_,((tr_tacer, tacer), (tr_ligns, fst_lign, sec_lign))) = smooth_results_cor
    tr_mean = Tr_cor

    fig,[ax4,ax4h] = plt.subplots(2,1,figsize = (8,12))

    smo, = ax4.plot(egbb_s,color="indigo",lw=3, label= "smoothed E growth")
    raw, = ax4.plot(egbb,"-.",color="magenta",lw=1, label = "E growth")
    ax4.legend(handles = [raw,smo])

    alpha=0.4
    smo, = ax4h.plot(eghf_s,color="darkred",lw=3, label= "smoothed E growth")
    raw, = ax4h.plot(eghf,"-.",color="red",lw=1, label = "E growth")

    if param.time_eval_method in ["both","trend"]:
        ligns, = ax4h.plot(t,fst_lign,":",color="sandybrown",alpha=alpha, label = 'trend')
        ax4h.plot(t,sec_lign,":",color="sandybrown",alpha=alpha)
        ax4h.axvline(tr_ligns,c='orange',alpha=alpha)
        exp = int(log10(max(eghf_s)))
        ax4h.text(tr_ligns-3,float(f"-0.25e{exp}"),f"{tr_ligns:.0f}s",c='orange')
    if param.time_eval_method in ["both","tacer"]:
        tac, = ax4h.plot(tacer/tacer.max()*eghf.max(),"-.",color="lightblue",alpha=alpha, label = 'tacer')
        ax4h.axvline(tr_tacer,c='lightseagreen',alpha=alpha)
        exp = int(log10(max(eghf_s)))
        ax4h.text(tr_tacer-3,float(f"-0.25e{exp}"),f"{tr_tacer:.0f}s",c='lightseagreen')

    textTr = f"   Tr  = {tr_mean:3.0f} s    "
    if ener_sta_bb.size==0:
        textBB = f"   EBB  = {nan:.2e} J  \n   MeBB = {nan:.2f}   "
    else :
        textBB = f"   EBB  = {egbb_s[tr_mean]:.2e} J  \n   MeBB = {Me(egbb_s[tr_mean]):.2f}   "
    textHF = f"   EHF  = {eghf_s[tr_mean]:.2e} J  \n   MeHF = {Me(eghf_s[tr_mean]/0.13):.2f}   "\
             + f"\nEhf/Tr3 = {eghf_s[tr_mean]/tr_mean**3:.2e}"

    if param.time_eval_method in ["tacer"]: ax4h.legend(handles = [tac,raw,smo],loc="lower center", ncol=2)
    if param.time_eval_method in ["trend"]: ax4h.legend(handles = [ligns,raw,smo],loc="lower center", ncol=2)
    if param.time_eval_method in ["both"]: ax4h.legend(handles = [ligns,tac,raw,smo],loc="lower center", ncol=2)

    for i,axi in enumerate([ax4,ax4h]):
        #axi.grid(color = 'lightgray', ls =":")
        axi.axvline(tr_mean,c='k',lw=2)
        axi.set_xlim(0,t.max())
        axi.xaxis.set_major_locator(MultipleLocator(100))
        axi.xaxis.set_minor_locator(MultipleLocator(20))
        if i<1 :
            if egbb.size==0 : val_max = 1
            else : val_max = max(egbb)
        else :
            val_max = max(eghf)
        exp = int(log10(val_max))
        num = round(val_max/float(f"1e{exp}"),0)
        axi.set_ylim(0,float(f"{num+1}e{exp}"))
        axi.yaxis.set_major_locator(MultipleLocator(float(f"{num}e{exp}")))
        axi.yaxis.set_minor_locator(MultipleLocator(float(f"{num/4}e{exp}")))
        axi.grid(axis="x", which="major")
        axi.grid(axis="x", which="minor", c='lightgray', ls =":")
        axi.grid(axis="y", which="minor", c='lightgray', ls =":")
        axi.grid(axis="y", which="major")

        if i<1 :
            axi.text(t.max()-90, float(f"{egbb_s[tr_mean]*1.3:.2e}"), textBB, verticalalignment='top',
                     horizontalalignment='left',
                     bbox={'facecolor': 'plum', 'alpha': 0.6, 'pad': 10}, fontsize=11)
        else:
            axi.text(t.max()-90, float(f"{eghf_s[tr_mean]:.2e}"), textHF, verticalalignment='top',
                     horizontalalignment='left',
                     bbox={'facecolor': 'salmon', 'alpha': 0.6, 'pad': 10}, fontsize=11)
            axi.text(t.max()-75, float(f"{eghf_s[tr_mean]*1.5:.2e}"), textTr, verticalalignment='top',
                     horizontalalignment='left',
                     bbox={'facecolor': 'silver', 'alpha': 0.6, 'pad': 10}, fontsize=11)

    #deco
    ax4h.set_xlabel('time since first P-waves(s)')
    ax4h.set_ylabel('HF energy (J)')
    ax4.set_ylabel('BB energy (J)')
    ax4.set_title('Energy growth corrected')
    plt.savefig(f"{resultpath}/E_growth_corrected.png",dpi=400)
    

    