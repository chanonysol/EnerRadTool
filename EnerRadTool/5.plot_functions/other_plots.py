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
from colorama import Back,Style,Fore

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colormaps
from matplotlib.colors import ListedColormap, Normalize

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from numpy import append, degrees

from results_plots import read_results_file,read_results_sta, read_log
from data_utils import name_reg_event

def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` Axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding Axes.

    """
    # calculate evenly-spaced axis angles
    theta = linspace(0, 2*pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = append(x, x[0])
                y = append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

def certain_report(resultpath,ev,correct = True, rt= False,savefile= False, 
                    ext =".png", redo=False):
    sys.stdout.write(f"\n{Fore.GREEN + Style.BRIGHT}"
                     +f"> Fiability evaluation - iteration {rt}")
    
    if os.path.exists(f"{resultpath}/certain_radar_{rt}{ext}") and not redo:
        sys.stdout.write(f"\n{Fore.GREEN + Style.BRIGHT}"
                     +" ... Already Done ")
        return
    
    from numpy import median,var,std, unique, dtype, loadtxt, nan, log10, \
        mean, isnan, concatenate, where
    from glob import glob
    
    plt.rcParams.update({'font.size': 14,'axes.titlesize':16,'axes.labelsize':16,
                         'xtick.labelsize':14,'ytick.labelsize':14,'legend.fontsize':14,
                             'figure.titlesize':14})
    #----- res
    res = read_results_file(glob(f"{resultpath.replace("[","?").replace("]","?")}"
                                  +f"/results*{'' if isinstance(rt,bool) else rt}.txt")[-1], 
                            correct = correct)
    #----- stations   
    stalog_bb = read_log(resultpath,
                          os.path.split(glob(f"{resultpath.replace(']','?').replace('[','?')}/log_keptdata_*BB*")[-1])[-1][:-4],
                          return_mat=True)
    ind = argwhere(array([any(i.find(f"{'REMAIN ITER' if correct and j else ''}{' ' if isinstance(j,int) else ''}{j}")!=-1 
                                                      for j in (range(rt+1) if not isinstance(rt,bool) else " ")) 
                                                  for i in stalog_bb[:,0]]))[:,0]
    stadist,azlist = stalog_bb[:,[2,3]][argwhere(array([any(i.find(f"{'REMAIN ITER' if correct and j else ''}{' ' if isinstance(j,int) else ''}{j}")!=-1 
                                                      for j in (range(rt+1) if not isinstance(rt,bool) else " ")) 
                                                  for i in stalog_bb[:,0]]))[:,0]].astype(float).T
    
    azlist = unique(azlist)
    stadist = unique(stadist)
    
    azgap = []; distgap = []
    if azlist.size == 1 :
        if isnan(azlist[0]):
            azgap.append(nan)
            distgap.append(nan)
        
    azlist = concatenate(((azlist[1:]-azlist[:-1]),[360-azlist[-1]+azlist[0]]))
    azgap = azlist.max()
    if stadist.size == 1:
        distgap = 75.
    else :
        distgap = (stadist[1:] - stadist[:-1]).max()

    #-------- catalogue
    dty = dtype([('date',"U25"),('lon',"f8"),('lat',"f8"),("reg","U25"),("sreg","U25"),
                  ('Ebb',"f8"),('Ehf',"f8"),('M0',"f8"),
                  ('MeBB',"f8"),('MeHF',"f8"),('theta',"f8"),('Tr',"f8")])
    mat = loadtxt("EnerRadTool/5.plot_functions/newcat.dat",dtype=dty,comments="#")  
    r,sr = name_reg_event(ev.lat,ev.lon) 
    
    ind = argwhere((mat["reg"]==r) & (mat["sreg"]==sr))[:,0]
    mati = mat[ind]["theta"]
    avs = array([mean(mati),median(mati),var(mati),std(mati)]).round(3)
    avs = array(avs).round(3)
    
    #------data
    from generate_event import moment
    xi = moment(ev.moment,ev.exponent)
   
    data = array([abs(res.theta-avs[0]),distgap/75,
                  abs(res.Tr_lines-res.Tr_tacer)/2/70,
                  res.dEbb/100,res.dEhf/100,
                  abs(res.Tr_tacer/((1.2*1e-8*(xi*1e7)**(1/3)+5.8)*2))/3,
                  azgap/360])
    
    Tr_mom = array([(1.2*1e-8*(xi*1e7)**(1/3)+5.8)*2,(1.2*1e-8*(xi*1e7)**(1/3)+2.8)*2,(1.2*1e-8*(xi*1e7)**(1/3))*2])
    Tr_mom = concatenate(([Tr_mom[0]*3,Tr_mom[0]*2],Tr_mom))
    Tr_mom = ((Tr_mom/Tr_mom[2])/3).round(2)
    
    N = 7
    theta = radar_factory(N, frame='polygon')

    fig, ax = plt.subplots(1,1,figsize=(9,9),
                            subplot_kw=dict(projection='radar'))

    level_dic={ "GREAT":{"val":[0. ,  0/75, 0.  , 0.  , 0.  , 0.         ,   0/360],
                        "color":"aquamarine","alpha":0.3},
                "GOOD" :{"val":[0.3, 10/75, 0.07, 0.05, 0.05, Tr_mom[-1] ,  45/360],
                        "color":"yellowgreen","alpha":0.3},
                "OK"   :{"val":[0.5, 25/75, 0.14, 0.1 , 0.1 , Tr_mom[-2] , 105/360],
                        "color":"khaki","alpha":0.4},
                "LIMIT":{"val":[0.7, 50/75, 0.42, 0.35, 0.35, Tr_mom[-3] , 270/360],
                        "color":"orange","alpha":0.3},
                "BAD"  :{"val":[0.8 , 60/75, 0.71, 0.6 , 0.6 , Tr_mom[-4] , 315/360],
                        "color":"salmon","alpha":0.25}
                }

    vals = []
    for k in level_dic.keys():
        p=ax.fill(theta,level_dic[k]["val"], facecolor=level_dic[k]["color"], linewidth = 0, 
                alpha=level_dic[k]["alpha"])#, label=k)
        vals.append(p)

    data = where(data>1.,1.,where(data<0.,0.,data))

    test = array([level_dic[k]["val"] for k in level_dic.keys()]).round(3)
    levels = [int(concatenate((argwhere(test[:,j]>d)[:,0],[4])).min()-1)
                for j,d in enumerate(data)]


    pr = ax.plot(theta,data, "o-k",label="results")

    if res.Tr_lines<res.Tr_tacer:
        levels[2] = 4

        pj = ax.plot(theta[2],data[2], "or", markerfacecolor="w",label="error")
    ax.fill(theta,data, facecolor="w", alpha=0.7,label='_nolegend_')    
    mark = sum(levels)
    levels = [list(level_dic.keys())[l] for l in levels]
    if res.Tr_lines<res.Tr_tacer:
        levels[2]="ERROR"

    ax.set_varlabels([  r"$|\Theta-\bar\Theta_{reg}|$",
                        r"gap$\Delta$",
                        r"$T_{lines}$""\n""$-$""\n""$T_{TACER}$",
                        r"d$E_{BB}$",
                        r"d$E_{HF}$",
                        r"$T_{TACER}$""\n""$/$""\n""$T_{GCMT+delay}$",
                        r"gap $\lambda_{sta}$"])

    pos = [["center","right","right","center","center","left","left"],
            ["top","center","center","bottom","bottom","center","center"],
            [-0.2 , 0.07, 0.16, 0.11,-0.16,-0.16,-0.07],
            [ 0.14, 0.04, 0.04, 0.08, 0.06, 0.04, 0.04]]

    for j,k in enumerate(levels):
        if k == "ERROR":
            c = "w"
        else :
            c = level_dic[k]["color"]
        ax.text(theta[j]+pos[2][j],-pos[3][j], k, verticalalignment=pos[1][j],
                  horizontalalignment=pos[0][j],
                  bbox={'facecolor': c, 'alpha': 0.3, 
                        "edgecolor": ("r" if k == "ERROR" else None)}, fontsize=10)

    ax.tick_params(pad=17)
    ax.set_rlim(1,0)
    ax.set_yticklabels([])
    #ax.set_rticks(arange(1,0,-0.33/2).round(2))

    legend1 = plt.legend([v[0] for v in vals], list(level_dic.keys()), 
                          loc="upper center",ncol=1,bbox_to_anchor=(0.02, 1.14))
    plt.gca().add_artist(legend1)
    ax.legend(loc=1)

    #plt.tight_layout()
    plt.title(f"Confidence level in results : {list(level_dic.keys())[mark//7]}",
              pad=50,weight='bold')
    
    if savefile :
        plt.savefig(f"{resultpath}/certain_radar_{rt}{ext}",
                    dpi = 300, transparent=True)
    plt.show()
    sys.stdout.write(f"\n{Fore.GREEN + Style.BRIGHT}"
                     +" ... Done ")


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