# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:50:28 2024

@author: solen

Description :
    Make a folder for the event, with a cmt file, data folder
    and automatic personalized parameter file
"""
import os
import sys

FILE_PATH = os.path.dirname(os.path.realpath(__file__)).replace('\\', '/')

DIR=f"{FILE_PATH}/../2.data_functions"
sys.path.insert(0, DIR)

from data_utils import read_log
from results_plots import read_results_file
from results_writing import read_results_sta
from plotFMC import circles

from numpy import pi, cos, sin
from matplotlib.dates import (DateFormatter, MonthLocator, YearLocator)
import matplotlib.gridspec as gridspec
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
from numpy import meshgrid, empty
from scipy.interpolate import LinearNDInterpolator
import matplotlib.ticker as ticker
from os import remove
from obspy.imaging.beachball import MomentTensor, mt2plane, aux_plane, mt2axes
import pygmt
from numpy import linspace, ones, shape, arange, log, min, vstack, exp, zeros_like

from matplotlib import cm
from matplotlib import colormaps
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.ticker import MultipleLocator
from matplotlib.dates import date2num, YearLocator, MonthLocator, DateFormatter
import matplotlib.pyplot as plt

from numpy import ceil, loadtxt, delete, argwhere, concatenate, max, min, abs, array, mean, sum
from numpy import floor, zeros, log10, isnan, vstack, argsort, shape, where, lexsort
from obspy import UTCDateTime
from glob import glob
from os import listdir
import matplotlib.dates as mdates
from copy import deepcopy
from tqdm import tqdm
from time import time
from datetime import datetime
from colorama import Back, Fore, Style

# =============================================================================
# Colormap
# =============================================================================
# colormap angular distance
cmap_dist = ListedColormap(colormaps["tab20c_r"](linspace(0., 1., 80)))
col_dist = cmap_dist.colors
#fig.colorbar(cm.ScalarMappable(norm=Normalize(0,80), cmap=cmap_dist), ax=[ax1,ax2],ticks=arange(0,81,8))
cmap_az = ListedColormap(colormaps["tab20b_r"](linspace(0.2, 1, 360))[44::45])
col_az = cmap_az.colors
# fig.colorbar(cm.ScalarMappable(norm=Normalize(0,360), cmap=cmap_az), ax=[ax1,ax2],ticks=arange(0,361,45))

# colormap deviation
cmap_dev = ListedColormap(colormaps["seismic"](linspace(0, 1, 31))[::2])
cmap_freq = ListedColormap(colormaps["tab20"](linspace(0.2, 1, 16)))

pygmt.makecpt(cmap="gray", series=[-8, 0, 1],
              continuous=True, output="elevgray.cpt")
# pygmt.makecpt(cmap="gray", series=[-7, 0, 1], background="i",
#               truncate = [0.5,1], continuous=True, 
#               output="elevgray.cpt") #clearer
pygmt.makecpt(cmap="polar", series=[-1, 1, .1],
              continuous=True, output="dev.cpt")
# pygmt.makecpt(cmap="cubhelix", series=[0,360,45],
#               continuous=False, output="ax.cpt")
# pygmt.makecpt(cmap="jet", series=[20,80,20],
#               continuous=True, output="dist.cpt")

pygmt.makecpt(cmap="jet", series=[4,8.2,.2],
              continuous=True, output="ener.cpt")
# =============================================================================
# Station map
# =============================================================================
from obspy.taup.taup_geo import calc_dist, calc_dist_azi
from xarray import DataArray
import io

def distance_grid(grid, evlo, evla):
    xv, yv = meshgrid(-180+grid.lon, grid.lat, indexing='ij')
    d = zeros_like(grid); az = zeros_like(grid)
    for i in range(shape(xv)[0]):
        for j in range(shape(xv)[1]):
            d[j, i], az[j, i], _ = calc_dist_azi(evla, evlo, 
                                                  yv[i, j], xv[i, j], 
                                                  6371.0, 0.0)
    # aplatit pour avoir un array 1D pour plot
    # xv, yv = xv.flatten(), yv.flatten()
    # d = d.T; az = az.T  # ca c'est un pb des meshgrid avec les dimensions ^^'
    # d = d.flatten().round(1); az = az.flatten().round(1)
    # ind = argwhere((d <= 80.) & (d >= 10.) & (d % 16 == 0.))[:, 0]
    # d, xv, yv = d[ind], xv[ind], yv[ind]
    return (DataArray(data=d,dims = ["lat","lon"],coords=[yv[0],xv[:,0]]),
            DataArray(data=az.round(1),dims = ["lat","lon"],coords=[yv[0],xv[:,0]]))

from numpy import where, lexsort, sign
import pygmt

def plot_stations_contrib_map(resultpath, ev, inv, st_bb,st_hf, 
                              correct = False, savefile = False, ext = ".png",
                              redo=False):
    
    
    zones = {"S.Pacific" : {"Puysegur":[159, 166, -51, -46], 
                        "NZ-S.Island":[166, 176, -48, -41], 
                        "NZ-N.Island":[174, 178, -42, -36],
                        "Hikurangi":[176, 182, -40, -36],
                        "Kermadec":[180, 186, -36, -25],
                        "None" :[159,186,-57,-25]},
         "SW.Pacific" : {"Samoa-Tonga":[183,190,-25,-13],
                         'N. Fiji':[170,183,-17.5,-14],
                         'S. Fiji' :[170,183,-24,-17.5],
                         "N.Loyalty":[164,170,-24,-14],
                         "Salomon-Isl.":[160,168,-14,-10],
                         "None":[155,190,-25,-10]},
         "None":{"None":None}}
    sys.stdout.write(f"\n{Fore.GREEN + Style.BRIGHT}"
                     +"> Map stations contribution")
    if (os.path.exists(f"{resultpath}/station_contrib_map_BB.{ext}") 
        and os.path.exists(f"{resultpath}/station_contrib_map_HF.{ext}")
        and not redo):
        sys.stdout.write(f"\n{Fore.GREEN + Style.BRIGHT}"
                     +" ... Already Done ")
        return
    
    bar = tqdm(total = 3, desc = "Making map")
    
    grid = pygmt.datasets.load_earth_relief(resolution="10m", region="g")*1e-3
    
    
    elon, elat, edep = ev.lon, ev.lat, ev.depth
    d, az = distance_grid(pygmt.datasets.load_earth_relief(resolution="30m", 
                                                               region="g")*1e-3, 
                              elon, elat)#read_sta_log(resultpath,hf=False)
    
    t,ener_sta_bb,names_bb = read_results_sta(f"{resultpath}/ener_all_bb.dat")
    t,ener_sta_hf,names_hf = read_results_sta(f"{resultpath}/ener_all_hf.dat")
       
    res = read_results_file(glob(f"{resultpath.replace("[","?").replace("]","?")}/results*")[-1], 
                            correct = correct)
    bar.update(1)
    # if correct:
    #     ind = argwhere(abs(log10(ener_sta_bb[:,int(res.Tr)]/res.Ebb))<=correct)[:,0]
    #     ener_sta_bb = ener_sta_bb[ind,:]; names_bb=names_bb[ind]
        
    #     ind = argwhere(abs(log10(ener_sta_hf[:,int(res.Tr)]/res.Ehf))<=correct)[:,0]
    #     ener_sta_hf = ener_sta_hf[ind,:]; names_hf=names_hf[ind]
        
    for j,(ener_sta, st, nam, E) in enumerate(zip([ener_sta_bb,ener_sta_hf],
                                             [st_bb,st_hf], [names_bb,names_hf],
                                             [res.Ebb, res.Ehf])):
        if (os.path.exists(f"{resultpath}/station_contrib_map_{'BB' if j==0 else 'HF'}.{ext}") and not redo):
            sys.stdout.write(f"\n{Fore.GREEN + Style.BRIGHT}"
                         +f" ... {'BB' if j==0 else 'HF'} already done ")
            return
        stalog = read_log(resultpath,
                              os.path.split(glob(f"{resultpath.replace(']','?').replace('[','?')}/log_keptdata_*{'BB' if j==0 else 'HF'}*")[-1])[-1][:-4],
                              return_mat=True)
        keep_names=stalog[:,1][argwhere(array([i.find("REMAIN ITER")!=-1 for i in stalog[:,0]]))[:,0]]
        remove_names=stalog[:,1][argwhere(array([i.find("dev")!=-1 for i in stalog[:,0]]))[:,0]]
        keep_log=stalog[:,[2,3,6,7]][argwhere(array([i.find("REMAIN ITER")!=-1 for i in stalog[:,0]]))[:,0]].astype(float).T
        remove_log=stalog[:,[2,3,6,7]][argwhere(array([i.find("dev")!=-1 for i in stalog[:,0]]))[:,0]].astype(float).T
        ind = [i for i,n in enumerate(nam) if n in keep_names]
        ind_remove = [i for i,n in enumerate(nam) if n in remove_names]
        
        zone = zones[str(ev.reg)][str(ev.sreg)]
        
        _,azs,lats,lons = keep_log
        lons = where(lons<0,lons+360,lons)
        
        
        fig = pygmt.Figure()
        # projection azimuthale equidistant lon0/lat0/distance max/resolution
        fig.basemap(region="g", projection="E%f/%f/90/20c" %
                    (elon,elat), frame=["af10", f"WSne"])
        # PLOT
        fig.grdimage(grid=grid, cmap="elevgray.cpt", verbose="q")
        fig.grdimage(grid=az, cmap="EnerRadTool/5.plot_functions/ax.cpt", verbose="q", transparency = 70)
        fig.coast(shorelines="0.15p,black", land=None, resolution='h')
        
        if len(ind_remove)!=0:
            _,_,rem_lats,rem_lons = remove_log
            rem_lons = where(rem_lons<0,rem_lons+360,rem_lons)
            fig.plot(x=rem_lons,y=rem_lats,fill="white",
                     style="t.3c", pen='0.1p,red')
        
        dev = log10(ener_sta[ind,int(res.Tr)]/E)
        size = 1/abs(dev)
        size = (where(size>10,10,size)/10).round(2)
        size = sign(dev)*where(size<0.25,0.25,size)
        fig.plot(x=lons,y=lats,fill=dev.round(2),size = size,
                  cmap = "dev.cpt",style="t", pen='0.5p,black')
        
        fig.grdcontour(grid=d, annotation = "16+f13p")
        if zone:
            zone = array([array(zone)[[0,2,1,3]]])
            fig.plot(data=zone, style='r+s', pen="2p,yellow")
        
        legend = io.StringIO( "\n".join([f"S 0.4 t {1/m/10:.1f} {m:.1f} .5p 1. {m} \nG {s}" 
                                       for m,s in zip([.5,.1,-.1,-.5],[.3,.2,.3,.05])]+
                                        [f"S 0.4 t {1/.6/10:.1f} white .5p,red .7 removed"]))
        fig.legend( spec=legend,position="jTL+o0.3c/0.2c")  
        fig.colorbar(position="JBL+o-1c/-4.5c+w5c/.3c",
                     frame=["xa.5f.25+llog10(Esta/E)", "y"],
                     cmap="dev.cpt",box='+gwhite')
        fig.colorbar(position="JBR+o-1c/-4.5c+w5c/.3c",
                     frame=["x90f45+lsta_az", "y"],transparency = 40,
                     cmap="EnerRadTool/5.plot_functions/ax.cpt",box='+gwhite')
        fig.colorbar(frame=["xa1f.5+lElevation", "y+lkm"],
                      cmap="elevgray.cpt")
        bar.update(1)
        if savefile :
            fig.savefig(f"{resultpath}/station_contrib_map_{'BB' if j==0 else 'HF'}{ext}",
                        dpi = 300, transparent=True)
        fig.show()
        sys.stdout.write(f"\n{Fore.GREEN + Style.BRIGHT}"
                         +f" ... {'BB' if j==0 else 'HF'} already done ")  

# from numpy import tan, sign
# plt.figure(figsize=(8,8))
# plt.plot(elon,elat,"*k",markerfacecolor="r",markersize=20)
# for azi,col in zip(arange(0,360,45),col_az[::]):
#     x=array([elon, elon + sin(azi*pi/180)*180])
#     y=array([elat, elat + cos(azi*pi/180)*90])
#     x=linspace(x[0],x[1],30,endpoint=True)
#     y=linspace(y[0],y[1],30,endpoint=True)

#     #y = where(y<-90,y+180,y)
#     x = where(x>180,x-360,x)
#     if argwhere(y<-90).size!=0:
#         ind = argwhere(y<-90)[0,0]
#         plt.plot(x[:ind],y[:ind],
#                     'o',markerfacecolor="w",c=col,label=str(azi))
#         plt.plot(x[ind:]-180,y[ind:]-y[0],
#                     'o',markerfacecolor="w",c=col)
#     else:
#         plt.plot(x,y,'o',c=col,label=str(azi))
# plt.grid()
# plt.legend(ncol=4)
# plt.xlim(-180,180)
# plt.ylim(-90,90)