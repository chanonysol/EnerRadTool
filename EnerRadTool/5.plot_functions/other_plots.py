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

from numpy import empty,argwhere,vstack,linspace,meshgrid, array, pi, shape
from scipy.interpolate import LinearNDInterpolator

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colormaps
from matplotlib.colors import ListedColormap, Normalize

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
    plt.show()
    plt.close()