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

from numpy import pi, cos, sin
from matplotlib.dates import (DateFormatter, MonthLocator, YearLocator)
import matplotlib.gridspec as gridspec
from numpy.linalg import inv
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
from numpy import meshgrid, empty
from scipy.interpolate import LinearNDInterpolator
import matplotlib.ticker as ticker
from os import remove
from obspy.imaging.beachball import MomentTensor, mt2plane, aux_plane, mt2axes
import pygmt
from numpy import linspace, ones, shape, arange, log, min, vstack, exp

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

# =============================================================================
# Colormap
# =============================================================================
# colormap angular distance
cmap_dist = ListedColormap(colormaps["tab20c_r"](linspace(0., 1., 80)))
col_dist = cmap_dist.colors
#fig.colorbar(cm.ScalarMappable(norm=Normalize(0,80), cmap=cmap_dist), ax=[ax1,ax2],ticks=arange(0,81,8))
cmap_az = ListedColormap(colormaps["tab20b"](linspace(0, .799, 360)))
col_az = cmap_az.colors
# fig.colorbar(cm.ScalarMappable(norm=Normalize(0,360), cmap=cmap_az), ax=[ax1,ax2],ticks=arange(0,361,45))

# colormap deviation
cmap_dev = ListedColormap(colormaps["seismic"](linspace(0, 1, 31))[::2])
cmap_freq = ListedColormap(colormaps["tab20"](linspace(0.2, 1, 16)))

pygmt.makecpt(cmap="gray", series=[-8, 0, 1],
              continuous=True, output="elevgray.cpt")
pygmt.makecpt(cmap="polar", series=[-1, 1, .1],
              continuous=True, output="dev.cpt")
# =============================================================================
# Station map
# =============================================================================
def plot_station_rad_pattern(st,inv,dat,names):
    vals = empty((0,3))
    for tr in st:
        inv2 = inv.select(network=tr.stats.network,station=tr.stats.station)
        index = argwhere(names==tr.id)[0,0]
        val = dat[index]
        vals = vstack((vals,array([inv2[0][0].extra.station_param.distance,inv2[0][0].extra.station_param.az,val])))
    
    x,y,z = vals[:,1],vals[:,0],vals[:,2]
    X = linspace(0,360,360,endpoint=True)
    Y = linspace(20,90,5,endpoint=True)
    X, Y = meshgrid(X, Y)  # 2D grid for interpolation
    interp = LinearNDInterpolator(list(zip(x, y)), z)
    Z = interp(X, Y)
    
    cmap_dist = ListedColormap(colormaps["seismic"](linspace(0,1, shape(vals)[0])))
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'},figsize = (8,8))
    ax.pcolormesh(X*pi/180, Y, Z, shading='auto',cmap=cmap_dist,alpha=0.2,vmin=-1,vmax=1)
    test = ax.scatter(vals[:,1]*pi/180,vals[:,0],c=vals[:,2],edgecolor="k",s=120,marker="^",cmap=cmap_dist,vmin=-1,vmax=1)
    ax.set_rorigin(-25)
    ax.set_rmin(20)
    ax.set_rmax(90)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rticks([40,60,80])  # Less radial ticks
    ax.set_rlabel_position(120)  # Move radial labels away from plotted line
    ax.grid(True)
    plt.colorbar(test)
    #plt.show()
    plt.close()

#def plot_station_map(datalog):
    

# all_sta = read_log(f"{args.path}/Data/","station_log",
#                    return_mat=True)[:,-2:].astype(float)

# used_sta = read_log(f"{args.path}/Data/","station_log",
#                     return_mat="in processing _")[:,-2:].astype(float)
from numpy import shape, argwhere, mean, array, arange, concatenate
from numpy import meshgrid, linspace, zeros_like
from numpy import log10, exp, format_float_scientific, loadtxt
from numpy.fft import rfft, rfftfreq

from obspy.taup.taup_geo import calc_dist

def distance_grid(grid, evlo, evla):
    xv, yv = meshgrid(-180+grid.lon, grid.lat, indexing='ij')
    d = zeros_like(grid)
    for i in range(shape(xv)[0]):
        for j in range(shape(xv)[1]):
            d[j, i] = calc_dist(evla, evlo, yv[i, j], xv[i, j], 6371.0, 0.0)
    # aplatit pour avoir un array 1D pour plot
    xv, yv = xv.flatten(), yv.flatten()
    d = d.T  # ca c'est un pb des meshgrid avec les dimensions ^^'
    d = d.flatten().round(0)
    ind = argwhere((d <= 80.) & (d >= 10.) & (d % 10 == 0.))[:, 0]
    d, xv, yv = d[ind], xv[ind], yv[ind]
    return xv, yv, d

def distance_grid_dist(grid, evlo, evla,dist=30):
    xv, yv = meshgrid(-180+grid.lon, grid.lat, indexing='ij')
    d = zeros_like(grid)
    for i in range(shape(xv)[0]):
        for j in range(shape(xv)[1]):
            d[j, i] = calc_dist(evla, evlo, yv[i, j], xv[i, j], 6371.0, 0.0)
    # aplatit pour avoir un array 1D pour plot
    xv, yv = xv.flatten(), yv.flatten()
    d = d.T  # ca c'est un pb des meshgrid avec les dimensions ^^'
    d = d.flatten().round(0)
    ind = argwhere(d == dist)[:, 0]
    d, xv, yv = d[ind], xv[ind], yv[ind]
    return xv, yv, d


grid = pygmt.datasets.load_earth_relief(resolution="10m", region="g")*1e-3

fig = pygmt.Figure()
# projection azimuthale equidistant lon0/lat0/distance max/resolution
fig.basemap(region="g", projection="E%f/%f/28/20c" %
            (172, -30), frame=["af10", f"WSne"])
# PLOT
fig.grdimage(grid=grid, cmap="elevgray.cpt", verbose="q")  # plot elevation
fig.coast(shorelines="0.15p,black", land=None,
          resolution='h')  # coastlines
# Contour distance tout les 20 degres
# fig.contour(pen="1p", x=xv, y=yv, z=d, levels=20, annotation=20)
fig.plot(data=array([[160,-56,167,-46]]), style='r+s', pen="1p,red")
fig.plot(data=array([[167,-46,178,-37]]), style='r+s', pen="1p,red")
fig.plot(data=array([[178,-37,-174,-24]]), style='r+s', pen="1p,red")
fig.plot(data=array([[160,-56,-174,-24]]), style='r+s', pen="3p,yellow")

fig.plot(data=array([[162,-10,175,-23]]), style='r+s', pen="1p,red")
fig.plot(data=array([[175,-12,-171,-24]]), style='r+s', pen="1p,red")
fig.plot(data=array([[160,-10,-170,-24]]), style='r+s', pen="3p,yellow")

fig.plot(x=162, y=-50, style="c.5c", fill="red4", pen="black")
fig.plot(x=175, y=-42, style="c.5c", fill="red4", pen="black")
fig.plot(x=-177, y=-30, style="c.5c", fill="red4", pen="black")
fig.plot(x=-173.5, y=-18, style="c.5c", fill="red4", pen="black")
fig.plot(x=167, y=-12, style="c.5c", fill="red4", pen="black")

# fig.plot(x=all_sta.T[1], y=all_sta.T[0], fill=None, style="s.3c",
#           pen='0.5p,black')  # plot station en couleur selon distance
# fig.plot(x=used_sta.T[1], y=used_sta.T[0], fill="gold", style="t.6c",
#           pen='0.5p,black')  # plot station en couleur selon distance

fig.colorbar(frame=["xa1f.5+lElevation", "y+lkm"],
              cmap="elevgray.cpt")  # colorbar topo
fig.savefig(f"C:/Users/solen/Documents/Ener/project/invtest/region_map.png",
                    transparent=True, crop=True, anti_alias=True, dpi = 300)
fig.show(verbose="q")


# grid = pygmt.datasets.load_earth_relief(resolution="30m", region="g")*1e-3
# # xv, yv, d = distance_grid(grid, 172, -30)  # lon,lat,distance angulaire
# for i,(x,y) in enumerate(zip([162,175,-177,-173.5,167],[-50,-42,-30,-18,-12])):
#     # if i!=0 : continue
#     fig = pygmt.Figure()
#     # projection azimuthale equidistant lon0/lat0/distance max/resolution
#     fig.basemap(region="g", projection="E%f/%f/100/20c" %
#                 (172, -30), frame=["af10", f"WSne"])
#     # PLOT
#     fig.grdimage(grid=grid, cmap="elevgray.cpt", verbose="q")  # plot elevation
#     fig.coast(shorelines="0.15p,black", land=None,
#               resolution='h')  # coastlines
#     # Contour distance tout les 20 degres
    
#     #fig.contour(pen="1p", x=xv, y=yv, z=d, levels=10, annotation=20)

#     fig.plot(x=x, y=y, style="c.5c", fill="red4", 
#                   pen="black",label='Example location for eq')  
#     for di,c,r in zip([5,30,80],["lightgoldenrod1","lightsalmon1","indianred1"],[.1,.17,.23]):
#         xv, yv, d = distance_grid_dist(grid, x,y,dist=di)  # lon,lat,distance angulaire
#         fig.plot(x=xv, y=yv, style = f"s{r}c,{c}", fill=f"{c}",label=f'Radius {di}°')
        
    
#     # fig.plot(x=all_sta.T[1], y=all_sta.T[0], fill=None, style="s.3c",
#     #           pen='0.5p,black')  # plot station en couleur selon distance
#     # fig.plot(x=used_sta.T[1], y=used_sta.T[0], fill="gold", style="t.6c",
#     #           pen='0.5p,black')  # plot station en couleur selon distance
#     fig.legend()
#     # fig.colorbar(frame=["xa1f.5+lElevation", "y+lkm"],
#     #               cmap="elevgray.cpt")  # colorbar topo
#     fig.savefig(f"C:/Users/solen/Documents/Ener/project/invtest/dist_map_ev{i}.png",
#                     transparent=True, crop=True, anti_alias=True, dpi = 300)
#     fig.show(verbose="q")

# =============================================================================
# dev
# =============================================================================
# fig = pygmt.Figure()
# # projection azimuthale equidistant lon0/lat0/distance max/resolution
# fig.basemap(region="g", projection="E%f/%f/90/20c" %
#             (evlo, evla), frame=["af", f"WSne"])
# # PLOT
# fig.grdimage(grid=grid, cmap="elevgray.cpt",
#              frame='ag30', verbose="q")  # plot elevation
# fig.coast(shorelines="0.15p,black", land=None,
#           resolution='h')  # coastlines
# # Contour distance tout les 20 degres
# fig.contour(pen="1p", x=xv, y=yv, z=d, levels=20, annotation=20)
# fig.plot(x=all_sta.T[1], y=all_sta.T[0], fill=None, style="s.3c",
#          pen='0.5p,black')  # plot station en couleur selon distance
# fig.plot(x=lons, y=lats, fill=dev, style="t.6c", cmap = "polar",
#          pen='0.5p,black')  # plot station en couleur selon distance
# fig.colorbar(frame=["xa1f.5+lElevation", "y+lkm"],
#              cmap="elevgray.cpt")  # colorbar topo
# fig.show(verbose="q")