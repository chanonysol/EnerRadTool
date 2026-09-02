# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:50:28 2024

@author: solen

Description :
    Based on a list of events, create all needed events to constitute a
    catalog
"""
import os
import sys

FILE_PATH = os.path.dirname(os.path.realpath(__file__)).replace('\\', '/')

DIR=f"{FILE_PATH}/../2.data_functions"
sys.path.insert(0, DIR)

import matplotlib.pyplot as plt
import matplotlib as mp
from matplotlib import ticker
import numpy as np

from matplotlib import colormaps
from matplotlib import cm
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from numpy import linspace, arange, array, argsort, log10, argwhere, exp, pi, shape
from numpy.fft import rfft, rfftfreq
from colorama import Back, Fore, Style
from results_plots import read_results_file, glob
from results_writing import read_results_sta, read_sta_log, read_results, Me

# colormap angular distance
def plot_preprocessed_data_dist(resultpath,st,st_hf,inv,
                                correct = False, savefile= False, ext =".png"):
    sys.stdout.write(f"\n{Fore.GREEN + Style.BRIGHT}"
                     +"> Signals plot by distance")
    cmap_dist = ListedColormap(colormaps["tab20c_r"](linspace(0., 1., 80)))
    col_dist = cmap_dist.colors
    
    res = read_results_file(glob(f"{resultpath.replace("[","?").replace("]","?")}"
                                 +"/results*")[-1], 
                            correct = correct) 
    
    t,ener_sta_bb,names_bb = read_results_sta(f"{resultpath}/ener_all_bb.dat")
    t,ener_sta_hf,names_hf = read_results_sta(f"{resultpath}/ener_all_hf.dat")
    
    if len(shape(ener_sta_bb)) == 1 :
        ener_sta_bb = array([ener_sta_bb])
    if len(shape(ener_sta_hf)) == 1 :
        ener_sta_hf = array([ener_sta_hf])
        
    if correct:
            ind = argwhere(abs(log10(ener_sta_bb[:,int(res.Tr)]/res.Ebb))<=correct)[:,0]
            names_bb = names_bb[ind]
            
            ind = argwhere(abs(log10(ener_sta_hf[:,int(res.Tr)]/res.Ehf))<=correct)[:,0]
            names_hf = names_hf[ind]
    
    mp.rcParams.update({'font.size': 16})
    
    fig, axs = plt.subplots(2,1,figsize = (14,10))
    for j,(sti,nam) in enumerate(zip([st,st_hf],[names_bb,names_hf])):
        for tr in sti:
            if tr.id not in nam : continue
            a = inv.select(channel=tr.stats.channel, station=tr.stats.station, 
                                 location = tr.stats.location)[0][0].extra.station_param.az
            d = inv.select(channel=tr.stats.channel, station=tr.stats.station, 
                                 location = tr.stats.location)[0][0].extra.station_param.distance
            t=linspace(0, tr.stats.npts * tr.stats.delta, tr.stats.npts, endpoint=True)-60
            axs[j].plot(t, tr.data, color = col_dist[int(d)],
                        zorder=int(d)//4,lw = 2. if d%16<8 else 1.,alpha=0.4)
    
    for axi in axs:
        axi.set_facecolor("w")
        axi.grid()
        axi.set_xlim(-60,t.max())
        axi.axvline(0,c='k')
        axi.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1e}"))
        
    maxbb = array([abs(tr.data)[int((res.Tr+60)*tr.stats.sampling_rate)] for tr in st if tr.id in names_bb]).max()
    maxhf = array([abs(tr.data)[int((res.Tr+60)*tr.stats.sampling_rate)] for tr in st_hf if tr.id in names_hf]).max()
    
    axs[0].set_ylim(-10**round(log10(maxbb)+0.5,1),10**round(log10(maxbb)+0.5,1))
    axs[1].set_ylim(-10**round(log10(maxhf)+0.5,1),10**round(log10(maxhf)+0.5,1))
    
    #axs[1].set_xlabel("Time since first P-wave arrival(s)")
    axs[0].set_ylabel("Broadband velocity signals (m/s)")
    axs[1].set_ylabel("Highfrequency velocity signals (m/s)")
    fig.colorbar(cm.ScalarMappable(norm=Normalize(0,80), cmap=cmap_dist),
                  ax=axs,ticks=arange(0,81,8),label="Station epicentral distance (°)")
    if savefile :
        plt.savefig(f"{resultpath}/Signals_DistColor_{'all' if not correct else 'selected'}{ext}",
                    dpi = 300, transparent=True)
    
    plt.show()
    sys.stdout.write(f"\n{Fore.GREEN + Style.BRIGHT}"
                     +" ... Done ")

def plot_preprocessed_data_az(resultpath,st,st_hf,inv,
                              correct = False, savefile= False, ext =".png"):
    sys.stdout.write(f"\n{Fore.GREEN + Style.BRIGHT}"
                     +"> by azimuth")
    cmap_az = ListedColormap(colormaps["tab20b_r"](linspace(0.2, 1, 360))[44::45])
    col_az = cmap_az.colors
    
    res = read_results_file(glob(f"{resultpath.replace("[","?").replace("]","?")}"
                                 +"/results*")[-1], 
                            correct = correct) 
    
    t,ener_sta_bb,names_bb = read_results_sta(f"{resultpath}/ener_all_bb.dat")
    t,ener_sta_hf,names_hf = read_results_sta(f"{resultpath}/ener_all_hf.dat")
    
    if len(shape(ener_sta_bb)) == 1 :
        ener_sta_bb = array([ener_sta_bb])
    if len(shape(ener_sta_hf)) == 1 :
        ener_sta_hf = array([ener_sta_hf])
        
    if correct:
            ind = argwhere(abs(log10(ener_sta_bb[:,int(res.Tr)]/res.Ebb))<=correct)[:,0]
            names_bb = names_bb[ind]
            
            ind = argwhere(abs(log10(ener_sta_hf[:,int(res.Tr)]/res.Ehf))<=correct)[:,0]
            names_hf = names_hf[ind]
    
    mp.rcParams.update({'font.size': 16})
    
    fig, axs = plt.subplots(2,1,figsize = (14,10))
    for j,(sti,nam) in enumerate(zip([st,st_hf],[names_bb,names_hf])):
        for tr in sti:
            if tr.id not in nam : continue
            a = inv.select(channel=tr.stats.channel, station=tr.stats.station, 
                                 location = tr.stats.location)[0][0].extra.station_param.az
            d = inv.select(channel=tr.stats.channel, station=tr.stats.station, 
                                 location = tr.stats.location)[0][0].extra.station_param.distance
            t=linspace(0, tr.stats.npts * tr.stats.delta, tr.stats.npts, endpoint=True)-60
            axs[j].plot(t, tr.data, color = col_az[int(a)//45],
                        zorder=int(d)//4,lw = 1.,alpha=0.4)
    
    for axi in axs:
        axi.set_facecolor("w")
        axi.grid()
        axi.set_xlim(-60,t.max())
        axi.axvline(0,c='k')
        axi.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1e}"))
        
    maxbb = array([abs(tr.data)[int((res.Tr+60)*tr.stats.sampling_rate)] for tr in st if tr.id in names_bb]).max()
    maxhf = array([abs(tr.data)[int((res.Tr+60)*tr.stats.sampling_rate)] for tr in st_hf if tr.id in names_hf]).max()
     
    axs[0].set_ylim(-10**round(log10(maxbb)+0.5,1),10**round(log10(maxbb)+0.5,1))
    axs[1].set_ylim(-10**round(log10(maxhf)+0.5,1),10**round(log10(maxhf)+0.5,1))
    
    axs[1].set_xlabel("Time since first P-wave arrival(s)")
    axs[0].set_ylabel("Broadband velocity signals (m/s)")
    axs[1].set_ylabel("Highfrequency velocity signals (m/s)")
    fig.colorbar(cm.ScalarMappable(norm=Normalize(0,360), cmap=cmap_az),
                  ax=axs,ticks=arange(0,361,45),label="Station azimuth (°)")
    if savefile :
        plt.savefig(f"{resultpath}/Signals_AzColor_{'all' if not correct else 'selected'}{ext}",
                    dpi = 300, transparent=True)
    
    plt.show()
    sys.stdout.write(f"\n{Fore.GREEN + Style.BRIGHT}"
                     +" ... Done ")

def plot_preprocessed_spectrum(resultpath, st, st_hf, inv,
                               correct = False, savefile= False, ext =".png"):
    cmap_az = ListedColormap(colormaps["tab20b_r"](linspace(0.2, 1, 360))[44::45])
    col_az = cmap_az.colors
    
    fig, axs = plt.subplots(2,2,figsize = (16,10))
    names = [tr.id for tr in st if tr.id in [tr.id for tr in st_hf]]
    for j,sti in enumerate([st,st_hf]):
        for tr in sti:
            #if tr.id not in names : continue
            a = inv.select(channel=tr.stats.channel, station=tr.stats.station, 
                                 location = tr.stats.location)[0][0].extra.station_param.az
            spec = rfft(tr.data) * tr.stats.delta
            freq = rfftfreq(tr.stats.npts, d=tr.stats.delta)
            index = argwhere((freq >= 1/70) & (freq <= 2))[:, 0]
            spec, freq = spec[index], freq[index]
            
            t_star = 0.5-0.5*log10(freq)# for 0.1 <= freq <= 1 Hz
            t_star[argwhere(freq <= 0.1)[:, 0]] = 0.9-0.1 * \
                log10(freq[argwhere(freq <= 0.1)[:, 0]]) # for 0.1Hz <= freq
            t_star[argwhere(freq >= 1)[:, 0]] = 0.5-0.1 * \
                log10(freq[argwhere(freq >= 1)[:, 0]]) # for freq <= 1 Hz
            #print(freq[argwhere(freq >= 1)[:, 0]].min())
            # function to integrate
            func = (abs(spec))**2*exp(freq*2*pi*t_star)
            #func = (freq*abs(spec))**2
            #axs[j].semilogy(freq, abs(spec), color = col_az[int(a)//45],zorder=8-int(a)//45,lw = 0.5,alpha=0.6)
            axs[j][0].semilogy(freq, abs(spec), color = col_az[int(a)//45],zorder=8-int(a)//45,lw = 0.5,alpha=0.6)
            axs[j][1].semilogy(freq, func, color = col_az[int(a)//45],zorder=8-int(a)//45,lw = 0.5,alpha=0.6)

    for ax in axs:    
        for axi in ax:
            axi.set_facecolor("w")
            axi.grid()
            axi.set_xlim(0,2)
            axi.axvline(0,c='k')
    fig.colorbar(cm.ScalarMappable(norm=Normalize(0,360), cmap=cmap_az),
                  ax=axs,ticks=arange(0,361,45))
    if savefile :
        plt.savefig(f"{resultpath}/Spectrum_{'all' if not correct else 'selected'}{ext}",
                    dpi = 300, transparent=True)
    
    fig.show()

def test_color_personalized():
    import numpy as np
    x = np.arange(0, np.pi, 0.1)
    y = np.arange(0, 2 * np.pi, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = np.cos(X) * np.sin(Y) * 10

    colors = ["darkred", "lightseagreen", "lightblue","paleturquoise"]  # R -> G -> B
    n_bins = [4, 8, 10, 100]  # Discretizes the interpolation into bins
    cmap_name = 'my_list'
    fig, axs = plt.subplots(2, 2, figsize=(6, 9))
    fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.05)
    for n_bin, ax in zip(n_bins, axs.flat):
        # Create the colormap
        cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bin)
        # Fewer bins will result in "coarser" colomap interpolation
        im = ax.imshow(Z, origin='lower', cmap=cmap)
        ax.set_title("N bins: %s" % n_bin)
        fig.colorbar(im, ax=ax)

import pygmt
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
from numpy import where

def plot_from_psmeca():
    grid = pygmt.datasets.load_earth_relief(resolution="05m", region="g")*1e-3
    mat = loadtxt("psmeca_M55_thrust.dat", usecols = [0,1,2,3,4,5,6,7,8,9])
    
    fig = pygmt.Figure()
    fig.basemap(region=[159, 186, -57, -22],
                projection="M18c", frame=['a10f2g5'])
    fig.grdimage(grid=grid, cmap="elevgray.cpt", verbose="q",
                 projection="M18c", frame=["a10f2g2", "WSne"])
    fig.meca(spec=mat, scale=".3c",
             cmap="depth.cpt", pen=".001p", convention="mt")
    fig.coast(shorelines="0.08p,black", land=None,
              resolution='h')
    fig.colorbar(position="JBR+o-2.5c/-11c+w10c/.7c",
                 frame=["x%s" % "a10.f5", "y+l%s" % "depth"], cmap="elevgray.cpt",
                 box="+gwhite")
    fig.savefig("plotfiles/resultsMeca_M55_thrust.png",dpi=300)
    fig.show()
    
def plot_regions():
    grid = pygmt.datasets.load_earth_relief(resolution="05m", region="g")*1e-3
    pygmt.makecpt(cmap="gray", series=[-8, 0, 1],
                  continuous=True, output="elevgray.cpt")
    from numpy import mean
    fig = pygmt.Figure()
    fig.basemap(region=[150, 195, -56, -10],
                projection="M36c", frame=['a10f2g5'])
    fig.grdimage(grid=grid, cmap="elevgray.cpt", verbose="q",
                 projection="M36c", frame=["a10f2g2", "WSne"])
    fig.coast(shorelines="0.08p,black", land=None,
              resolution='h')
    for region,col in zip([[159, 186, -57, -25],
                            [155,190,-25,-10]],
                            ["lightgoldenrod","lightsalmon"]):
        polygon = Polygon([(region[0], region[2]),(region[0], region[3]),
                       (region[1], region[3]),(region[1], region[2])])
        x, y = polygon.exterior.xy
        fig.plot(x=x,y=y, pen = f"4p,{col}")
    for region,t in zip([[159, 166, -56, -45],[166, 176, -48, -41],
                    [172, 178, -42, -34],[178, 184, -42, -36],
                    [179, 186, -36, -25],],
                        ["Puysegur","NZ S.Island","NZ N.Island",
                                                "Hikurangi","Kermadec"]):
        polygon = Polygon([(region[0], region[2]),(region[0], region[3]),
                        (region[1], region[3]),(region[1], region[2])])
        x, y = polygon.exterior.xy
        fig.plot(x=x,y=y, fill = "lightgoldenrod",t = 60,pen="2p,black")
        fig.text(x=mean(abs(array(region[:2]))),y=region[2]+0.5,fill="white",
                 text=t ,font="24p,black")
    for region,t in zip([[158,170,-14,-10],[162,172,-24,-14],
                   [172,183,-17.5,-14],
                    [172,183,-24,-17.5],[183,190,-25,-10]],
                      ['Samoa-Tonga', 'S. Fiji', 'N. Fiji','N.Loyalty', 'Solomon-Isl.'][::-1]):
        polygon = Polygon([(region[0], region[2]),(region[0], region[3]),
                       (region[1], region[3]),(region[1], region[2])])
        x, y = polygon.exterior.xy
        fig.plot(x=x,y=y, fill = "lightsalmon",t = 60, pen="2p,black")
        fig.text(x=mean(abs(array(region[:2]))),y=region[2]+0.5, fill="white",
                 text=t ,font="24p,black")
    fig.savefig("Zones_map.png",dpi=400)
    # fig.colorbar(position="JBR+o-2c/-5.5c+w5c/.3c",
    #              frame=["x%s" % "a1.f.5", "y+l%s" % "depth(km)"], cmap="elevgray.cpt",
    #              box="+gwhite")
    fig.show()