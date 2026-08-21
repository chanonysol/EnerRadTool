# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:50:28 2024

@author: solen
"""
import os
from obspy import read, read_inventory
from obspy.taup.tau import TauPyModel
from obspy.core.util import AttribDict
from obspy.geodetics.base import degrees2kilometers, kilometers2degrees
from multiprocessing import Pool
import colorama
from numpy import vstack, loadtxt, array, argwhere, log10, ndarray, shape, unique, \
    reshape, savetxt, sin, cos, arcsin, pi, median, sqrt, mean, arange, nan, \
    concatenate, dtype, rec, argsort,array_split, delete, lexsort, linspace
from tqdm import tqdm
from copy import deepcopy
from time import sleep, time
import random
import sys

UTILS_PATH = os.path.dirname(os.path.realpath(__file__)).replace('\\', '/')
dty = dtype([('depth',"f8"), ('dist',"f8"), 
              ("phase","U25"),
              ("info","U25"), ("va","f8")])

def get_time_p_train(argum):
    """
    Get all time and angle theorical estimations from obspy

    Parameters
    ----------
    argum : tuple (for multiprocessing purpose)
        parameters for the estimations, automatically done via
        the functions precalc_time_p_train

    Returns
    -------
    dic : dictionnary
        arrival time, incident and takeoff angle for P,pP and sP waves
        as well as time difference between P and sP arrivals
        and P and S wave arrival

    """
    depth,distance,model = argum
    if distance>kilometers2degrees(215):
        arrivals = model.get_travel_times(source_depth_in_km=depth,
                                            distance_in_degree=distance,
                                            phase_list = ['P', 'pP', 'sP', "S"])
    else :
        arrivals = model.get_travel_times(source_depth_in_km=depth,
                                            distance_in_degree=distance,
                                            phase_list = ['P','S', 'pS','sS','SS'])
    
    try:
        dic = {}
        for arr in arrivals:
            if arr.name in dic.keys() : 
                continue
            if arr.name not in dic.keys():
                dic[arr.name]={"t0": arr.time.round(2),
                                "i0":  arr.takeoff_angle.round(2),
                                "ih":  arr.incident_angle.round(2)}
      
        if distance>kilometers2degrees(210.):
            origin_time = dic["P"]['t0']
            end_time = dic["S"]['t0'] - origin_time
            for k in [d for d in dic.keys() if d[-1]!="P"]:
                dic.pop(k)
            d2 = {"Ptrain":{'tmin': round(origin_time,2),
                             'tmax': round(end_time,2),
                             'dt':  round(array([dic[k]['t0'] 
                                        for k in dic.keys()]).max()-origin_time,2)}}
            
            
        else:
            origin_time = dic["P"]['t0']
            end_time = 3*60
            for k in [d for d in dic.keys() if d[-1]!="S" and d!="P"]:
                dic.pop(k)
            d2 = {"Strain":{'tmin': round(origin_time,2),
                             'tmax': round(end_time,2),
                             'dt':  round(array([dic[k]['t0'] 
                                        for k in dic.keys()]).max()-origin_time,2)}}
        dic = d2 | dic     
    except : 
        if distance>kilometers2degrees(215.):
            dic = {"Ptrain":{'tmin':nan, 'tmax':nan,'dt':nan}}    
        else : 
            dic = {"Strain":{'tmin':nan, 'tmax':nan,'dt':nan}}   
    
    test = []
    for info,val in list(AttribDict(dic).items()):
        if isinstance(val, AttribDict):
            phase = info
            for info,va in list(val.items()):
                test.append((depth,distance,phase,info,va))
            continue
    
    return test

def precalc_time_p_train(min_depth,max_depth,min_dist=3.,max_dist=82.,model_name = "iasp91"):
    """
    Generate pre-estimated values of arrival times and angles via obspy.TauPy

    Parameters
    ----------
    min_depth : int
        in km : Should be greater than 10km
    max_depth : int
        in km : Should be lower than 70km
    min_dist : float
        in degrees : Should be greater than 25
    max_dist : float
        in degrees : Should be lower than 80
    model_name : str, optional
        model to use for TauPy. The default is "iasp91".

    Returns
    -------
    None.

    """   
    fold_path = f"{UTILS_PATH}/travel_times/model_depth/{model_name}/"
    file_de = arange(0,700+1,20)

    max_de = file_de[argwhere(file_de>=max_depth)[0,0]]
    min_de = file_de[argwhere(file_de<=min_depth)[-1,0]]
    
    for d0,d1 in zip(arange(0,max_de,20),arange(20,max_de+20,20)):
        filepath = (f"{fold_path}/D{d0:.0f}-{d1:.0f}_"
                +f"{'tel' if min_dist > kilometers2degrees(200.) else 'near'}.dat")
        if d1%20==0: d1-=1
        if d1>max_depth: d1=max_depth
        if d0<min_depth: d0=min_depth
        depths = arange(d0,d1+1,1).astype(int)
        distances = arange(3. if min_dist > kilometers2degrees(200.) else 0.,
                           kilometers2degrees(200.)+.1 
                           if min_dist < kilometers2degrees(200.) else 82.1,
                           .1 if min_dist > kilometers2degrees(200.)
                           else 0.045).round(2)
        
        if os.path.exists(filepath):
            depth_done,dist_done = loadtxt(filepath,usecols=[0,1]).T
            depth_done,dist_done = unique(depth_done),unique(dist_done)
            depths = delete(depths,argwhere([d in depth_done for d in depths]))
        couple = [[de,di] for de in depths for di in distances]
        if len(couple)==0 : continue
    
        model = TauPyModel(model=model_name)
        argums = [(c[0],c[1],model) for c in couple]
        L = []
        with Pool(8) as p:
            for dic2 in tqdm(p.imap(get_time_p_train,argums),total=len(argums),
                           desc=f"{'making table':>10s}_D{d0:.0f}-{d1:.0f}_"
                               +f"DI{min_dist:02.0f}{max_dist:02.0f}"):
                L.append(array(dic2,dtype=dty))
        test = concatenate(L,dtype=dty)
        if os.path.exists(filepath):
            test2 = loadtxt(filepath,dtype=dty)
            test = concatenate((test,test2))
            test = test[lexsort((test['dist'],test['depth']))]
        if min_dist > kilometers2degrees(200.) :
            savetxt(filepath,test,fmt="%3.0f %04.1f %6s %4s %6.1f") 
        else :
            savetxt(filepath,test,fmt="%3.0f %04.2f %6s %4s %6.1f") 
        sleep(5)
    
if __name__ == "__main__":
    # precalc_time_p_train(0,50,0,kilometers2degrees(200.),model_name="iasp91")
    precalc_time_p_train(0,10,3.,82.,model_name="iasp91")
    # precalc_time_p_train(10,70,3.,82.,model_name="prem")
    # precalc_time_p_train(10,70,3.,82.,model_name="jb")
    # precalc_time_p_train(10,70,3.,82.,model_name="1066b")

def inv_argsort(L):
    L=array(L)
    ind = argsort(L)
    invind = deepcopy(ind)
    for j,i in enumerate(ind) : invind[i]=j
    return ind,invind

def find_depth_dist_ind(depth,distance,model_name='iasp91'):
    """
    Look for the index in tables based on depth and distance

    Parameters
    ----------
    depth : int or list
        depth of the event(s)
    distance : int or list
        distance from hypocenter of the station(s)
    model_name : TYPE, optional
        model to use for TauPy. The default is 'iasp91'.

    Returns
    -------
    ind : list
        list of 2-dimension indexes

    """
    fold_path = f"{UTILS_PATH}/travel_times/model_depth/{model_name}/"
    sortind,invsortind = inv_argsort(array(distance))
    sortdist = distance[sortind]
    
    ind=len(distance)
    depth = round(depth)
    if all(distance> kilometers2degrees(200)):
        dist = "tel"
        sortdist = sortdist.round(1)
    else:
        dist = "near"
        sortdist = (sortdist//0.045+sortdist%0.045//0.22)*0.045
        sortdist = sortdist.round(2)
    file_de = arange(0,700+1,20)
    d1 = file_de[argwhere(file_de>depth)[0,0]]
    d0 = file_de[argwhere(file_de<=depth)[-1,0]]
    
    file = f"{fold_path}/D{d0:.0f}-{d1:.0f}_{dist}.dat"
    if not os.path.exists(file) :
        precalc_time_p_train(d0,d1,distance.min(),distance.max(),
                              model_name=model_name)
    mat = loadtxt(file,usecols=[0,1])
    inds = [argwhere((mat.T[0]==depth)
                      &(mat.T[1]==j))[:,0]
             for j in sortdist]
    L = loadtxt(file,dtype=dty)[concatenate(inds)]
    if dist == "tel":
        ind_split = argwhere((L["phase"]=="Ptrain")&(L["info"]=="tmin"))[1:,0]
    if dist == "near":
        ind_split = argwhere((L["phase"]=="Strain")&(L["info"]=="tmin"))[1:,0]
    L = array_split(L,ind_split)
    # L = [loadtxt(file,dtype=dty)[index] 
    #         for file,index in zip(search_files,inds)]   
    return [L[i] for i in invsortind]

def request_val_p_train(arg,depth,distlist,model_name="iasp91"):
    """
    get values of time and angle requested into pre-estimated files

    Parameters
    ----------
    arg : list
        parameters to look for among the following :
            - "P" or/and "sP" or/and "pP"
            - "arrival_time" or/and "incident_angle" or/and "takeoff_angle" :
                if more than 1 type of wave is requested, will give an
                array for each data-type of each wave
            - "duration" :
                minimum and maximum time after first P wave arrival
                to include all sP waves and avoid all S-waves at all
                stations

    depth : int or list
        depth of the event(s)
    distlist : int or list
        distance from hypocenter of the station(s)
    model_name : TYPE, optional
        model to use for TauPy. The default is 'iasp91'.

    Returns
    -------
    requests : list
        contains np.arrays and lists of requested information
    keys : list
        Indicates the order of the results in "requests"

    """
    vals = find_depth_dist_ind(depth,distlist,model_name=model_name)    

    if isinstance(arg, str):
        arg =[arg]
    waves = [value for value in arg if value in ["P","pP","sP"]]
    element = [("t0" if value=="time" else "i0" if value=="takeoff" else "ih") 
               for value in arg if value in ["time","incident","takeoff"]]
    
    if "duration" in arg:
        waves+=["Ptrain"]
        element+=["dt","tmax"]

    dics = [{wave:{el:v 
            for el,v in zip(val["info"][argwhere(val["phase"]==wave)[:,0]],
                            val["va"][argwhere(val["phase"]==wave)[:,0]])
            if el in element} 
            for wave in val["phase"] if wave in waves}
            for val in vals]

    requests = [array([dic[wave][el] 
                        if wave in dic.keys() else nan
                        for dic in dics]) 
                for wave in waves[:(-1 if "duration" in arg else len(waves))]
                for el in element[:(-2 if "duration" in arg else len(element))]]
    keys = [f"{wave}-{'arrival_time' if el=='time' else el+'_angle'}" 
                for wave in waves[:(-1 if "duration" in arg else len(waves))]
                for el in [value for value in arg if value in ["time","incident","takeoff"]]] 
    if "duration" in arg:
        requests.append([array([dic["Ptrain"]["dt"] for dic in dics]).max(),
                          array([dic["Ptrain"]["tmax"] for dic in dics]).min()])
        keys+= ['Train-duration,tmax'] 
    
    return requests,keys


def request_val_s_train(arg,depth,distlist,model_name="iasp91"):
    from numpy import nanmin, nanmax
    """
    get values of time and angle requested into pre-estimated files

    Parameters
    ----------
    arg : list
        parameters to look for among the following :
            - "P" or/and "sP" or/and "pP"
            - "arrival_time" or/and "incident_angle" or/and "takeoff_angle" :
                if more than 1 type of wave is requested, will give an
                array for each data-type of each wave
            - "duration" :
                minimum and maximum time after first P wave arrival
                to include all sP waves and avoid all S-waves at all
                stations

    depth : int or list
        depth of the event(s)
    distlist : int or list
        distance from hypocenter of the station(s)
    model_name : TYPE, optional
        model to use for TauPy. The default is 'iasp91'.

    Returns
    -------
    requests : list
        contains np.arrays and lists of requested information
    keys : list
        Indicates the order of the results in "requests"

    """
    vals = find_depth_dist_ind(depth,distlist,model_name=model_name)    

    if isinstance(arg, str):
        arg =[arg]
    waves = [value for value in arg if value in ["S","SS"]]
    element = [("t0" if value=="time" else "i0" if value=="takeoff" else "ih") 
               for value in arg if value in ["time","incident","takeoff"]]
    
    if "duration" in arg:
        waves+=["Strain"]
        element+=["dt","tmax"]

    dics = [{wave:{el:v 
            for el,v in zip(val["info"][argwhere(val["phase"]==wave)[:,0]],
                            val["va"][argwhere(val["phase"]==wave)[:,0]])
            if el in element} 
            for wave in val["phase"] if wave in waves}
            for val in vals]

    requests = [array([dic[wave][el] 
                        if wave in dic.keys() else nan
                        for dic in dics]) 
                for wave in waves[:(-1 if "duration" in arg else len(waves))]
                for el in element[:(-2 if "duration" in arg else len(element))]]
    keys = [f"{wave}-{'arrival_time' if el=='time' else el+'_angle'}" 
                for wave in waves[:(-1 if "duration" in arg else len(waves))]
                for el in [value for value in arg if value in ["time","incident","takeoff"]]] 
    if "duration" in arg:
        requests.append([nanmax(array([dic["Strain"]["dt"] for dic in dics])),
                         nanmin(array([dic["Strain"]["tmax"] for dic in dics]))])
        keys+= ['Train-duration,tmax'] 
    
    return requests,keys
# =============================================================================
# ---- adapt parameters
# =============================================================================
def find_min_dist(ev,param):
    i=0
    for phase in ["P","pP","sP"]:
        times = request_val_p_train([phase,"time"],
                                    ev.depth,arange(5,80.1,.1))[0][0]
        ii = argwhere(isnan(times)==False)[0,0]
        if ii>i : i=ii
    return array([param.distance.min, arange(5,80.1,.1).round(1)[i]]).max()

def find_train(ev,stadist,t,dt=10):
    tp,tpp,tsp = request_val_p_train(["P","pP","sP","time"],
                                ev.depth,stadist)[0]
    tsp-=tp;tpp-=tp;tp-=tp
    tsp = vstack([tsp]*len(t)).T; tpp = vstack([tpp]*len(t)).T
    return where((tpp<=t+dt) & (tsp<t+dt),"sP",
                         where((tpp<=t+dt) & (tsp>t+dt),"pP","P")) 
    # if all(tr>tpp) and all(tr>tsp) : return ["P","pP","sP"]
    # elif all(tr>tpp) :
    #     if any(tr<tsp-10) : 
    #         d = arange(param.distance.min,80.1,.1).round(1)[argwhere(tr+10<tsp)[0,0]]
    #         if abs(param.distance.min-d)<abs(param.distance.max-d) :
    #             param.distance.min = d
    #         else :
    #             param.distance.max = d
    #             return ["P","pP","sP"]
    # # pb on precalc la constante alors que la je veux adapter en fonction de tr
    # # corrige la ou c'est necessaire? *Fest2/Fest_new2?
    # # je crois que c'est visible sur Vanuatu2026
    #     return ["P","pP"]
    # elif any(tr<tpp-10):
    #     d = arange(param.distance.min,80.1,.1).round(1)[argwhere(tr+10<tpp)[0,0]]
    #     if abs(param.distance.min-d)<abs(param.distance.max-d) :
    #         param.distance.max = d
    #     else :
    #         param.distance.min = d
    #         return ["P","pP"]
    # return ["P"]
    
# =============================================================================
# ---- G
# =============================================================================
from numpy import cumsum,tan
 
def numdif(x,y,h):
    dx = round(x[1]-x[0],5)
    if not isinstance(h,int) : h = round(h/dx)
    new_x = x[h:-h+1]
    new_y = array([])
    i=h
    for i in range(h,len(x)-h):
        xi, yi = x[i-h:i+h+1], y[i-h:i+h+1]
        #print(i,xi[h])
        numy = (yi[-1] - yi[0])/(h*2*dx)
        new_y = concatenate((new_y,[numy]))
        #i+=h
    for i in range(h-1,0-1,-1):
        xi, yi = x[i:i+h+1], y[i:i+h+1]
        # print(xi*180/pi)
        numy = (yi[-1] - yi[0])/(h*dx)
        # print(numy)
        new_y = concatenate(([numy],new_y,))
        #i+=h
    for i in range(len(x)-2*h,len(x)-h):
        xi, yi = x[i:i+h+1], y[i:i+h+1]
        #print(xi*180/pi)
        numy = (yi[-1] - yi[0])/(h*dx)
        # print(numy)
        new_y = concatenate((new_y,[numy]))
    new_x =  x
    #y = concatenate(([(yi[h-1] - yi[0])/(h*dx)],new_y,[(yi[-1] - yi[-1-h])/(h*dx)]))
    return new_x, new_y

def numdif2(x,y,h):
    dx = round(x[1]-x[0],5)
    if not isinstance(h,int) : h = round(h/dx)
    new_x = x[h:-h]
    new_y = array([])
    i=h
    for i in range(h,len(x)-h):
        xi, yi = x[i-h:i+h+1], y[i-h:i+h+1]
        #print(i,xi[h])
        numy = (yi[-1] -2*yi[h] + yi[0])/(h*dx)**2
        new_y = concatenate((new_y,[numy]))
        #i+=h
    for i in range(h-1,0-1,-1):
        xi, yi = x[i:i+h+2], y[i:i+h+2]
        # print(xi*180/pi)
        numy = (yi[-1] -2*yi[h] + yi[0])/(h*dx)**2
        # print(numy)
        new_y = concatenate(([numy],new_y,))
        #i+=h
    for i in range(len(x)-2*h,len(x)-h):
        xi, yi = x[i:i+h+2], y[i:i+h+2]
        #print(xi*180/pi)
        numy = (yi[-1] -2*yi[h] + yi[0])/(h*dx)**2
        # print(numy)
        new_y = concatenate((new_y,[numy]))
    new_x =  x
    #y = concatenate(([(yi[h-1] - yi[0])/(h*dx)],new_y,[(yi[-1] - yi[-1-h])/(h*dx)]))
    return new_x, new_y

from scipy import interpolate
from scipy.interpolate import make_smoothing_spline
from scipy.interpolate import CubicSpline
from numpy import isnan

def g(dist_list,depth=15,model_name="iasp91"):
    x = arange(3.,82.1,.1).round(1)
    ih,i0,t = request_val_p_train(["P","takeoff","incident","time"],depth,x,
                                model_name=model_name)[0]
    spl = make_smoothing_spline(x, deepcopy(ih))
    ih_s = spl(x[::10]); f = interpolate.interp1d(x[::10],ih_s)
    dist,var = numdif(x*pi/180, f(x)*pi/180,2.*pi/180) 
    f = interpolate.interp1d(dist*180/pi,var.round(10));
    ind = array([argwhere(x==i)[0,0] for i in dist_list])
    ih = ih[ind]; i0 = i0[ind];
    var = f(dist_list)
    return (2.9*6.8/2.6/5.8
            *sin(ih*pi/180)/sin(dist_list*pi/180)
            /cos(i0*pi/180)*abs(var))**0.5
   
    # dist,var = numdif2(x[::10]*pi/180,t[::10],4.*pi/180) 
    # f = interpolate.interp1d(dist*180/pi,var);
    # ind = array([argwhere(x==i)[0,0] for i in dist_list])
    # ih = ih[ind]; i0 = i0[ind];var = f(dist_list)
    # return (2.9*6.8/2.6/5.8
    #         *tan(ih*pi/180)/cos(i0*pi/180)*6.8/(6371-15)
    #         *abs(var))**0.5

# =============================================================================
# ---- Folder
# =============================================================================
from numpy.fft import rfft, rfftfreq
from obspy import UTCDateTime
from scipy import interpolate
from scipy.interpolate import make_smoothing_spline

def init_data_folder(path):
    os.mkdir(f"{path}/Data")
    for na in ["Records","Stations","FixedRecords","FixedStations",
               "CloseRecords","CloseStations"]:
        os.mkdir(f"{path}/Data/{na}")

def save_spectrum(datapath,file,spectrum):
    for tr in spectrum:
        tr.trim(tr.stats.starttime+2*60,
                tr.stats.starttime+2*60+150)
        spec = rfft(tr.data)
        freq = rfftfreq(tr.stats.npts, d=tr.stats.delta)
        index = argwhere((freq>0)&(freq<2))[:,0]
        tr.data = abs(spec[index])
        tr.stats.delta = mean(freq[index][1:] - freq[index][:-1])
        time = tr.stats.starttime
        tr.stats.starttime = UTCDateTime(time.year,time.month,time.day,0,0,0)
        
    spectrum.write(f"{datapath}/../Spectra/{file}.mseed")
    
    
# =============================================================================
# ---- PREPROCESSING FUNCTIONS
# =============================================================================
from numpy import ones
def Fest(dist_list,phase_list = "sP"):  # dimensionless
    if not isinstance(phase_list, str):
        return where(phase_list=="sP", 
                     1.171-7.271e-3*dist_list+6.009e-5*dist_list**2,
                     where(phase_list=="pP", 
                           0.2353-4.109e-3*dist_list+8.453e-6*dist_list**2, 
                           4/15))
    if phase_list == "sP":
        return 1.171-7.271e-3*dist_list+6.009e-5*dist_list**2
    if phase_list == "pP":
        return 0.2353-4.109e-3*dist_list+8.453e-6*dist_list**2
    if phase_list == "P":
        return ones(shape(dist_list))*4/15

# def g(dist_list,ih,i0,model_name="iasp91"):  # dimensionless
#     delta = dist_list
#     # ih,i0 = request_val_p_train(['P','takeoff','incident'],15,dist_list,
#     #                              model_name="iasp91")[0]
#     toff25 = request_val_p_train(['P','takeoff'],15.,array([25.]),
#                                  model_name=model_name)[0][0][0]
#     var = (ih - toff25)/(delta-25)
#     return (sin(ih*pi/180)/sin(delta*pi/180)/cos(i0*pi/180)*abs(var))**0.5

"""G above"""

def receiver_function_P(incid_angles, alpha = 7):  # dimensionless
    i0 = incid_angles*pi/180
    beta = alpha*3**-0.5
    p = sin(i0)/alpha
    j0 = arcsin(p*beta)
    num = 2/beta**2*(1/beta**2-2*p**2)*cos(i0)
    den = (1/beta**2-2*p**2)**2+(4*p**2*cos(i0)*cos(j0))/(alpha*beta)
    return num/den


def receiver_function_S(incid_angles, alpha=7, comp="N"):  # dimensionless
    from numpy import arccos
    j0 = incid_angles*pi/180
    beta = alpha/sqrt(3)
    p = sin(j0)/beta
    i0 = arcsin([complex(pi*alpha,0) for pi in p])
    num = 2/beta**2*(1/beta**2-2*p**2)*cos(j0)
    den = (1/beta**2-2*p**2)**2+(4*p**2*cos(i0)*cos(j0))/(alpha*beta)
    if comp=="N":
        return (num/den).real
    if comp=="E":
        return array([2.]*len(incid_angles))
        

def waves_parameters(dist_list,incid_angles,depth=15,model_name="iasp91"):
    # ih,i0 = request_val_p_train(['P','takeoff','incident'],15,dist_list,
    #                              model_name="iasp91")[0]
    # if all(dist_list<2):
    #     Ch = receiver_function_S(incid_angles)
    #     Cv = receiver_function_S(incid_angles,comp="E")
    #     return Ch,Cv
    # else :
    C = receiver_function_P(incid_angles)
    #constants and averages
    rho = 3e3
    alpha = 7e3  # kg/m3 ; m/s
    q = 15.6
    Fp = 16/15
    Fg = Fest(dist_list)
    R = 6371.0e3/g(dist_list,15.,model_name)  # m
    Econst = (1+q)*Fp*R**2/Fg*rho*alpha  # Ee=Econst*eflux_sta
    return C, Econst

# def snr_mag(mag):
#     # mag = 0
#     snrlim = mag/3.4
#     if snrlim < 1.47:
#         return round(20*log10(1.47**2),2)
#     return round(20*log10(snrlim**2),2)

def snr_mag(mag):
    # mag = 0
    snrlim = mag/3.2
    if snrlim < 1.778:
        return round(20*log10(1.778**2),2)
    return round(20*log10(snrlim**2),2)

def verif_size(st, freq):
    ind_max = array([argwhere(abs(tr.data) == max(abs(tr.data)))[
                            0, 0]*tr.stats.delta for tr in st])-60

    for i in argwhere(array(ind_max-mean(ind_max)).round(0) > 20.)[::-1, 0]:
        st.remove(st[i])

    maenv = array([max(abs(tr.data)) for tr in st])

    freqlim = 1.7
    if freq == "HF" : freqlim = 2.5

    ind = argwhere(abs(maenv/median(maenv)).round(2) >= freqlim)[:, 0]
    while ind.size!=0:
        for i in ind[::-1]:
            st.remove(st[i])
        maenv = array([max(abs(tr.data)) for tr in st])
        ind = argwhere(abs(maenv/median(maenv)).round(2) >= freqlim)[:, 0]

    return st

def rms(tr):
    data = abs(tr.data)
    return sqrt(mean(data**2))
    # return mean((data-mean(data))**2)

def signaltonoise(st,timewindow):
    Arr = []
    for (i,tr) in enumerate(st):
        noise = tr.copy().trim(tr.stats.starttime, tr.stats.starttime+timewindow)
        signal = tr.copy().trim(tr.stats.starttime+timewindow, tr.stats.starttime+2*timewindow)
        # Arr.append(max(abs(signal.data))/max(abs(noise.data)))
        Arr.append(rms(signal)/rms(noise))
    return 20*log10(array(Arr))

def correct_regional_dist(dist,ener,ref="us"):
    # M0 = mean(ev.moment)*float(f"1e{ev.exponent}")
    # theta = log10(ener/M0)
    
    if ref=="EO12":
        val = -array( [1.19,1.16,1.16,1.11,1.07,1.02,0.97,0.88,0.82,0.78,0.76,0.74,
                     0.67,0.64,0.6,0.68,0.72,0.77,0.72,0.56,0.35,0.18,0.11,0.09,
                     0.17,0.31,0.53,0.72,0.8,0.76,0.64,0.53,0.42,0.36,0.34,0.36,
                     0.44,0.56,0.68,0.72,0.63,0.45,0.28,0.16,0.13,0.1,0.11,0.11,
                     0.12,0.12,0.09,0.01,-0.08,-0.13,-0.14,-0.08,-0.04,-0.02,
                     -0.01,0])
    elif ref=="us":
        val = array([-1.67124865e+00, -2.19576519e+00, -2.30355631e+00, -2.25051382e+00,
               -1.99218010e+00, -2.12692694e+00, -2.15381309e+00, -1.95914870e+00,
               -1.86430258e+00, -1.63874469e+00, -1.35743681e+00, -1.15161780e+00,
               -1.20173564e+00, -1.15116598e+00, -1.10661365e+00, -1.19061574e+00,
               -5.73571777e-01, -2.05317337e-01, -1.65840954e-01, -3.89741751e-02,
                2.88198615e-02,  4.32924081e-02,  5.80084733e-02,  1.65076219e-01,
                1.97122283e-01,  1.64939345e-01,  1.26333482e-01,  9.38108371e-02,
                4.85533735e-02,  1.06405191e-01,  3.22249385e-02, -3.07833990e-01,
               -3.71750433e-01, -8.59759911e-02,  1.73934149e-01,  2.31251726e-01,
                1.87152738e-01,  1.82388342e-01,  1.83963346e-01,  2.53011514e-01,
                3.43141904e-01,  2.41601954e-01, -8.36119352e-02, -2.09082851e-01,
               -1.35363861e-01, -7.72329126e-02, -3.25328927e-02,  2.18560382e-02,
               -7.06695967e-03, -1.11700520e-01, -1.11978983e-01, -1.04901377e-01,
               -1.20123987e-01, -6.90879958e-02, -4.92639964e-02, -2.24619355e-02,
               -5.86207666e-04,  1.59955977e-02,  2.27766936e-02,  5.52300203e-02,
                ])
    corr_dict = {str(d):t for d,t in zip(arange(5.,35,.5),val)}

    corr = [corr_dict[str(arange(5.,35,.5)[argwhere(arange(5.,35.,.5)<d)[-1,0]])]
            if argwhere(arange(5.,35,.5)<d).size!=0 and d<35.
            else 0
            for d in dist]
    corr =  reshape(array(shape(ener)[1]*corr),(shape(ener)[1],shape(ener)[0])).T
    #corr = array([corr_dict[str(d)] if str(d) in corr_dict.keys() else 0 for d in dist])

    # theta += corr
    # new_ener = (10**theta)*M0

    return ener + ener*(10**corr-1)

# =============================================================================
# ---- GEOMETRY FUNCTIONS
# =============================================================================
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
from numpy import where

def name_reg_event(elat,elon,new_zone=False):
    elon = where(array([elon]) < 0, array([elon])+360, array([elon]))[0]

    zones = {"S.Pacific" : [{"Puysegur":[159, 166, -57, -46], 
                            "NZ-S.Island":[166, 176, -48, -41], 
                            "NZ-N.Island":[174, 178, -42, -36],
                            "Hikurangi":[176, 184, -42, -36],
                            "Kermadec":[180, 186, -36, -25]},
                            [159,186,-57,-25]],
             "SW.Pacific" : [{"Samoa-Tonga":[183,190,-25,-10],
                             "Fiji":[172,183,-23,-12], 
                             "New-Caledonia":[166,172,-23,-14], 
                             "Salomon-Isl.":[155,167,-25,-10]},
                             [155,190,-25,-10]]
             }
    
    new_zones = {"S.Pacific" : [{"Puysegur":[159, 166, -57, -46], 
                            "NZ-S.Island":[166, 176, -48, -41], 
                            "NZ-N.Island":[174, 178, -42, -36],
                            "Hikurangi":[176, 184, -42, -36],
                            "Kermadec":[180, 186, -36, -25]},
                            [159,186,-57,-25]],
             "SW.Pacific" : [{"Samoa-Tonga":[183,190,-25,-13],
                             'N. Fiji':[170,183,-17.5,-14],
                             'S. Fiji' :[170,183,-24,-17.5],
                             "N.Loyalty":[164,170,-24,-14],
                             "Salomon-Isl.":[160,168,-14,-10]},
                             [155,190,-25,-10]]
             }
    if new_zone==True:
        zones=new_zones
    
    for regname in list(zones.keys()):
        region = zones[regname][1]
        polygon = Polygon([(region[0], region[2]),(region[0], region[3]),
                           (region[1], region[3]),(region[1], region[2])])
        if not polygon.contains(Point(elon,elat)): continue
        for subregname, region in zip(list(zones[regname][0].keys()),
                                      list(zones[regname][0].values())) :
            polygon = Polygon([(region[0], region[2]),(region[0], region[3]),
                               (region[1], region[3]),(region[1], region[2])])
            if polygon.contains(Point(elon,elat)):
                return regname, subregname
        return regname, str(None)
    return str(None),str(None)


# =============================================================================
# ---- WRITTING FUNCTIONS
# =============================================================================
# from obspy import read_inventory
# inv = read_inventory(f"{args.path}Data/iter*/stations/*.xml")

def write_log(path,filename, st, inv, modif = False, update = False):
    reasons = "in processing _"
    col = colorama.Fore.WHITE

    if not modif and not update:
        with open(f"{path}/{filename}.dat",
                  "w" if not os.path.exists(f"{path}/{filename}.dat") else "a") as file:
            for tr in st:
                inv2 = inv.select(station=tr.stats.station, location = tr.stats.location)
                file.write(f"{reasons.center(15):17s}"
                +f" {tr.id:15s} {inv2[0][0].extra.station_param.distance:5.2f} {inv2[0][0].extra.station_param.az:6.2f}"
                +f" {tr.stats.sampling_rate:05.1f} Hz "
                +f" {inv2[0][0].latitude:+8.4f} {inv2[0][0].longitude:+9.4f}\n")

    if update:
        with open(f"{path}/{filename}.dat","a") as file:
            if not st and not inv :
                return
            if inv and not st:
                for net in inv:
                    for sta in net:
                        for cha in sta:
                            name = f"{net.code}.{sta.code}.{cha.location_code}.{cha.code}"
                            file.write(f"{update.center(15):17s}"
                            +f" {name:15s} {0:5.2f} {0:6.2f}"
                            +f" {cha.sample_rate:05.1f} Hz "
                            +f" {sta.latitude:+8.4f} {sta.longitude:+9.4f}\n") 
                return
            for tr in st:
                if inv :
                    inv2 = inv.select(channel=tr.stats.channel, station=tr.stats.station, location = tr.stats.location)
                    file.write(f"{update.center(15):17s}"
                    +f" {tr.id:15s} {inv2[0][0].extra.station_param.distance:5.2f} {inv2[0][0].extra.station_param.az:6.2f}"
                    +f" {tr.stats.sampling_rate:05.1f} Hz "
                    +f" {inv2[0][0].latitude:+8.4f} {inv2[0][0].longitude:+9.4f}\n")        
                else :
                    file.write(f"{update.center(15):17s}"
                    +f" {tr.id:15s} {0:5.2f} {0:6.2f}"
                    +f" {tr.stats.sampling_rate:05.1f} Hz "
                    +f" {0:+8.4f} {0:+9.4f}\n")                

    if modif:
        with open(f"{path}/{filename}.dat","w") as file:
            for line in modif:
                file.write(line)

def read_log(path,filename,modif = False, return_mat=False, print_log = False):
    with open(f"{path}/{filename}.dat","r") as file:
        lines = file.readlines()
    if print_log:
        reasons = ["REMAIN _ _",
                   "no station data","empty data _",
                   "wrong channel _", "late peak ampli"
                   "over sized data","too much noise",
                   "dev too high"]
        col = [colorama.Fore.GREEN,
               colorama.Fore.BLACK,colorama.Fore.BLACK,
               colorama.Fore.YELLOW,colorama.Fore.YELLOW,
               colorama.Fore.RED,colorama.Fore.RED,
               colorama.Fore.MAGENTA]
        for line in lines:
            index = argwhere(array(reasons) == " ".join(line.split()[:3]))[0,0]
            sys.stdout.write(f"{colorama.Style.BRIGHT + col[index]:9s}"
                             +line.replace("_"," "))
    #lines = loadtxt(f"{path}/{filename}.dat",dtype=str)
    if modif : return lines
    if return_mat :
        mat = loadtxt(f"{path}/{filename}.dat",dtype=str)[:,:]
        reasons = [" ".join(i).replace("_","").strip() for i in mat[:,:3]]
        #dist az sampling_rate Hz lat lon
        if isinstance(return_mat,str) :
            index = argwhere(array(reasons)==return_mat)[:,0]
            mat = mat[:,array([4,5,6,8,9])][index,:].astype(float)
        else :
            mat = vstack((reasons,mat[:,3:].T)).T
        return mat


def modify_log(path,filename,names,reason):
    lines = read_log(path,filename,modif = True)
    for name in names:
        for i,line in enumerate(lines):
            if name in line:
                line =  line.replace(line[:16],f"{reason.center(15):15s} ")
                lines[i] = line
    write_log(path,filename, None, None, modif = lines)

def rem_line_log(path,filename,dist):
    lines = read_log(path,filename,modif = True)
    dists = read_log(path,filename,return_mat=True)[:,2].astype(float)
    reasons = read_log(path,filename,return_mat=True)[:,0]
    ind = argwhere((array([d not in dist.round(1) for d in dists])==False)|
                   (reasons!="in processing"))[:,0]
    for i in ind[::-1] : lines.pop(i)
    if len(lines)!=0:
        write_log(path,filename, None, None, modif = lines)
        

# =============================================================================
# ---- Inventory / Stream update
# =============================================================================
from obspy import Stream, Inventory, read, read_inventory
import json
from glob import glob

def add_attribute_xml(inv,infodict,st,datapath,invpath,saveinv = False,returninv=False):
    inv_new = Inventory()
    ns = 'http://some-page.de/xmlns/1.0'
    if not saveinv and len(glob(f"{invpath}/*.json"))!=0:
        for tr in st:
            inv_sta = inv.select(network = tr.stats.network,
                                 station = tr.stats.station,
                                 location = tr.stats.location,
                                 channel = tr.stats.channel)[0]
            with open(f"{invpath}/{tr.id}.json", encoding="utf-8") as file : 
                data = AttribDict(json.load(file))
            inv_sta[0].extra = data
            inv_new += inv_sta
        return inv_new
    
    classes = site_class_parameters([tr.stats.station for tr in st])
    
    for tr,(d,a,ba,r,e,pt,i1,i0),(cl,k,beta) in zip(st,infodict.T,classes):
        inv_sta = inv.select(network = tr.stats.network,
                             station = tr.stats.station,
                             location = tr.stats.location,
                             channel = tr.stats.channel)[0]
        
        if d >= 5.:
            extra = {"station_param" : {"distance":d,"az":a,"baz":ba},
                     "ray_param" : {"p_time":pt,"p_incident":i1,"p_takeoff":i0,
                                    "pp_time":pt,"pp_incident":i1,"pp_takeoff":i0,
                                    "sp_time":pt,"sp_incident":i1,"sp_takeoff":i0},
                     "correction" : {"ray":r,"ener":e}} 
        else :
            extra = {"station_param" : {"distance":d,"az":a,"baz":ba},
                     "ray_param" : {"s_time":pt,"s_incident":i1,"s_takeoff":i0},
                     #"correction" : {"raySH":r,"raySV":e},
                     "site_class" : {"class":cl,"k":k,"beta":beta}} 
        inv_sta[0].extra = AttribDict(extra)
        inv_new += inv_sta
        
        if "Close" in datapath:
            fname = f"{invpath}/{tr.id[:-1]}T.json"
        else :
            fname = f"{invpath}/{tr.id}.json"
        if saveinv and not os.path.exists(fname): 
            with open(fname, 'w', encoding='utf-8') as f:
                json.dump(extra, f, ensure_ascii=False, indent=2)
    if returninv:
        return inv_new
    
def site_class_parameters(staname,default = "A"):
    path = "EnerRadTool/2.data_functions/site_class_metadata_Kaiser2017.csv"
    dty = dtype([('station',"U25"),('site',"U25"),('vs30',"f8"),('tsite',"U25"),
                 ('Zb',"f8"),('Q_Vs30',"U25"),('Q_Tsite',"U25"),
                 ('D_Tsite',"U25"),('Q_Zb',"U25"),('ref',"U25")])
    Nzclass = loadtxt(path,comments="#",delimiter=",",dtype=dty,usecols=range(10))
    Nzclass = array(Nzclass[["station","site","vs30"]])
    
    path = "EnerRadTool/2.data_functions/spec_decay.dat"
    dty = dtype([('station',"U25"),('k',"f8"),('dk',"f8")])
    k0sta = loadtxt(path,comments="#",dtype=dty)
    
    site_class = {"A":{"k":0.04,"beta":2.9},"B":{"k":0.04,"beta":1.13},
                  "C":{"k":0.048,"beta":.52},"D":{"k":0.055,"beta":.255},
                  "gener":{"k":0.04,"beta":.62}}
    L=[]
    for na in staname:
        if na in Nzclass["station"] :
            ind = argwhere(Nzclass["station"]==na)[0,0]
            _,site, beta = Nzclass[ind]
            if site=="E": site="D"
            beta*=1e-3
            k= site_class[site]["k"]
        else :
            k,beta = site_class[default].values()
            site= None
        if na in k0sta["station"]:
            ind = argwhere(k0sta["station"]==na)[0,0]
            _,k,_ = k0sta[ind]
        L.append([site,k,beta])
    return L
# =============================================================================
# ---- Certainty evaluation
# =============================================================================
def find_az_dist_gap(resultpath,rt,correct=True, inv = None):
    if not inv:
        stalog_bb = read_log(resultpath,
                              os.path.split(glob(f"{resultpath.replace(']','?').replace('[','?')}/log_keptdata_*BB*")[-1])[-1][:-4],
                              return_mat=True)
        ind = argwhere(array([any(i.find(f"{'REMAIN ITER' if correct and j else ''}{' ' if isinstance(j,int) else ''}{j}")!=-1 
                                                          for j in (range(rt+1) if not isinstance(rt,bool) else " ")) 
                                                      for i in stalog_bb[:,0]]))[:,0]
        stadist,azlist = stalog_bb[:,[2,3]][ind].astype(float).T
    else :
        azlist,stadist = array([(sta.extra.station_param.az ,sta.extra.station_param.distance)
                for net in inv for sta in net]).T
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
    #azgap, distgap = find_az_dist_gap()
    return azgap, distgap

def compar_cat(ev):
    from numpy import median,var,std
    dty = dtype([('date',"U25"),('lon',"f8"),('lat',"f8"),("reg","U25"),("sreg","U25"),
                  ('Ebb',"f8"),('Ehf',"f8"),('M0',"f8"),
                  ('MeBB',"f8"),('MeHF',"f8"),('theta',"f8"),('Tr',"f8")])
    mat = loadtxt("EnerRadTool/5.plot_functions/newcat.dat",dtype=dty,comments="#")  
    r,sr = name_reg_event(ev.lat,ev.lon) 
    
    ind = argwhere((mat["reg"]==r) & (mat["sreg"]==sr))[:,0]
    mati = mat[ind]["theta"]
    avs = array([mean(mati),median(mati),var(mati),std(mati)]).round(3)
    avs = array(avs).round(3)
    return avs

def quality_level(resultpath,rt,ev,correct=True,results=None):
    from results_plots import read_results_file
    if not results:
        res = read_results_file(glob(f"{resultpath.replace("[","?").replace("]","?")}"
                                  +f"/results*{'' if isinstance(rt,bool) else rt}.txt")[-1], 
                                correct = correct)
    else :
        res = AttribDict(results[f"{'raw' if not correct else 'corrected'}_smooth"])
        res.dEbb = res.dEBB; res.dEhf = res.dEHF
    
    MOM, LEVEL_DIC = ev.levels_conf()
    
    avs = compar_cat(ev)
    azgap, distgap = find_az_dist_gap(resultpath,rt,correct=True)
    
    data = array([abs(res.theta-avs[0]),distgap/75,
                  abs(res.Tr_lines-res.Tr_tacer)/2/70,
                  res.dEbb/100,res.dEhf/100,
                  abs(res.Tr_tacer/((1.2*1e-8*(MOM*1e7)**(1/3)+5.8)*2))/3,
                  azgap/360])
    
    data = where(data>1.,1.,where(data<0.,0.,data))

    test = array([LEVEL_DIC[k]["val"] for k in LEVEL_DIC.keys()]).round(3)
    levels = [int(concatenate((argwhere(test[:,j]>d)[:,0],[4])).min()-1)
                for j,d in enumerate(data)]
    if res.Tr_lines<res.Tr_tacer:
        levels[2] = 4
    mark = sum(levels)
    levels_name = [list(LEVEL_DIC.keys())[l] for l in levels]
    if res.Tr_lines<res.Tr_tacer:
        levels_name[2]="ERROR"
    
    return data, levels, levels_name, mark, list(LEVEL_DIC.keys())[mark//7]
# =============================================================================
# ---- Impedance
# =============================================================================
def beta2rho(vel):
    return 2.473 + 0.0938*vel

def find_vel(depth):
    d_prem,rho_prem,vp_prem,vs_prem = loadtxt("prem.txt",skiprows = 1,usecols=[1,2,3,4]).T
    index = argwhere(d_prem>=depth)[-1,0]
    if d_prem[index-1]==d_prem[index]:
        index-=1
    return rho_prem[index],vp_prem[index],vs_prem[index]

def find_generic_layer(depth,return_index=True):
    generic_layers = array([0.001,0.03,0.19,4,8])
    index = array([argwhere(generic_layers>d)[0,0]
                  for d in depth])
    if return_index :
        return  generic_layers[index], index
    return generic_layers[index]

from scipy.interpolate import splrep, splev

def generic_rock_vel(quarterperiods, de,dz=0.001, pwave = False):
    model_depth = arange(0,8,dz)
    model = array([0.245,
                    lambda depth : 2.206*depth**0.272,
                    lambda depth : 3.542*depth**0.407,
                    lambda depth : 2.505*depth**0.199,
                    lambda depth : 2.927*depth**0.086])
    
    
    _, index = find_generic_layer(model_depth, return_index=True)
    model_vel = model[index]
    model_vel = array([v if isinstance(v,float) else v(di) 
                  for v,di in zip(model_vel,model_depth)]).round(3)
    if pwave :
        _,a,b = find_vel(8.)
        model_vel*=a/b
    model_times = cumsum(dz/model_vel).round(3)

    rho_0, alpha_0, beta_0 = find_vel(de)
    # rho_0, alpha_0, beta_0 = find_vel(8.)

    layers = array([argwhere(model_times<=t)[-1,0] for t in quarterperiods])
    depths = model_depth[layers]

    vel_gener = array([mean(model_vel[:l+1]) for l in layers])
    vel_model = array([where(model_vel<=model_vel[l],model_vel,0) for l in layers])
    if pwave :
        den_gener = beta2rho(vel_gener/(a/b))
    else :
        den_gener = beta2rho(vel_gener)

    if pwave :
        A = sqrt(alpha_0*rho_0/(vel_gener*den_gener))
    else :
        A = sqrt(beta_0*rho_0/(vel_gener*den_gener))
    return depths, A


def find_hard_rock_layer(depth, return_index=False):
    hardrock_layers = concatenate((arange(0,0.8,0.05),[2.2,15]))
    index = array([argwhere(hardrock_layers>=d)[0,0]
                  for d in depth])
    if return_index :
        return  hardrock_layers[index], index
    return hardrock_layers[index]

def very_hard_rock_vel(quarterperiods, de, dz=0.001, pwave= False):
    model_depth = arange(0,15,dz)

    model = array([2.768,2.808,2.847,2.885,
                    2.922,2.958,2.993,3.026,
                    3.059,3.091,3.122,3.151,
                    3.180,3.208,3.234,3.260,
                    lambda depth : 3.324*depth**0.067,
                    lambda depth : 3.447*depth**0.0209])

    _, index = find_hard_rock_layer(model_depth, return_index=True)
    model_vel = model[index]
    model_vel = array([v if isinstance(v,float) else v(di) 
                  for v,di in zip(model_vel,model_depth)]).round(3)
    
    if pwave:
        _,a,b = find_vel(8.)
        model_vel*=a/b
    
    model_times = cumsum(dz/model_vel)

    rho_0, alpha_0, beta_0 = find_vel(de)

    layers = array([argwhere(model_times<=t)[-1,0] for t in quarterperiods])
    depths = model_depth[layers]

    vel_hard = array([mean(model_vel[:l+1]) for l in layers])
    vel_model = array([where(model_vel<=model_vel[l],model_vel,0) for l in layers])
    if pwave :
        den_hard = beta2rho(vel_hard/(a/b))
    else :
        den_hard = beta2rho(vel_hard)
    
    if pwave :
        A = sqrt(alpha_0*rho_0/(vel_hard*den_hard))
    else :
        A = sqrt(beta_0*rho_0/(vel_hard*den_hard))

    return depths, A

def interp_imp_P(freq,depth,typ = "hardrock"):
    freq0 = array([0.01,0.1,0.2,0.3,0.5,0.9,1.25,1.8,3,5.3,8,14])
    if typ == "gener":
        A0 = array([1.15, 1.15, 1.18, 1.24, 1.34, 1.46, 1.54, 1.66, 1.93, 2.29, 2.55, 2.85])
    if typ == "hardrock":
        A0 = array([1.07, 1.07, 1.08, 1.09, 1.1,  1.13, 1.14, 1.16, 1.18, 1.2,  1.2,  1.21])
        
    rho,alpha,beta = find_vel(8)    
        
    tcl = splrep(freq0,A0)
    # freq = arange(0.01,10, 0.01)
    A = splev(freq, tcl)   
    A = A**2/rho/alpha
    rho_0, alpha_0, beta_0 = find_vel(depth)
    rho_0, alpha_0 = 3.,6.8
    
    return sqrt(alpha_0*rho_0*A)

def interp_imp_S(stalat,stalon,freq,site,de):
    if site=="A":
        freq0 = array([0.01,0.1,0.2,0.3,0.5,0.9,1.25,1.8,3,5.3,8,14])
        A0 = array([1,1.02,1.03,1.05,1.07,1.09,1.11,1.12,1.13,1.14,1.15,1.15])
    if site=="B":
        freq0 = array([0.01,0.09,0.16,0.51,0.84,1.25,2.26,3.17,6.05,16.6,61.2])
        A0 = array([1,1.1,1.18,1.42,1.58,1.74,2.06,2.25,2.58,3.13,4])
    if site=="C":
        freq0 = array([0.01,0.09,0.16,0.51,0.84,1.25,2.26,3.17,6.05,16.6,61.2])
        A0 = array([1,1.21,1.32,1.59,1.77,1.96,2.25,2.42,2.7,3.25,4.15])
    if site=="D":
        freq0 = array([0.01,0.09,0.16,0.51,0.84,1.25,2.26,3.17,6.05,16.6,61.2])
        A0 = array([1,1.43,1.71,2.51,2.92,3.1,3.23,3.18,3.18,3.18,3.18])
    if site=="gener":
        freq0 = array([0.01,0.09,0.16,0.51,0.84,1.25,2.26,3.17,6.05,16.6,61.2])
        A0 = array([1,1.1,1.18,1.42,1.58,1.74,2.06,2.25,2.58,3.13,4])
        
    rho,_, beta = find_vel(8)    
        
    tcl = splrep(freq0,A0)
    # freq = arange(0.01,10, 0.01)
    A = splev(freq, tcl)   
    A = A**2/rho/beta
    rho_0, alpha_0, beta_0 = find_vel(de)
    
    
    return sqrt(beta_0*rho_0*A)


def geometrical_spreading_s(freq, dist, cross_dist=27.5):
    # dist*=1e3
    # cross_dist*=1e3
    from numpy import zeros_like
    if dist<= cross_dist :
        g_square = zeros_like(freq) + dist**2
    else :
        gamma = 0.5+2*log10(5*deepcopy(freq))
        gamma[argwhere(freq<=0.2)[:,0]]=0.5
        gamma[argwhere(freq>=0.25)[:,0]]=0.7
        g_square = cross_dist**2 * (dist/cross_dist)**(2*gamma)
    return g_square

def attenuation_s(freq,lat,lon):
    # q = 400*(deepcopy(freq)/1.5)**0.6
    # q[argwhere(freq<=1.5)[:,0]]=400
    Qs,Qlats,Qlons = Qmodel()
    dl=.2
    ind = argwhere((Qlats<=lat+dl)&(Qlats>=lat-dl)&(Qlons<=lon+dl)&(Qlons>=lon-dl))[:,0]
    
    if len(ind)==0:
        if lat>-40:
            ind = argwhere((Qlons[:3*64]<=lon+dl)&(Qlons[:3*64]>=lon-dl))[:,0]
    if len(ind)==0:
        q=209*deepcopy(freq)**0.6
    else :
        q = Qs[ind].mean()*deepcopy(freq)**0.55
        # q = Qs[ind].mean()/2*deepcopy(freq)**0.9
    return q

def Qmodel():
    from numpy import dtype, reshape, loadtxt, unique, pi, cos, sin
    from xarray import DataArray
    file = "EnerRadTool/2.data_functions/Qsnzw2p3xyzltln.tbl.txt"
    datatype=open(file,"r").readlines()[1][1:].split()
    vals = ["f8"]*len(datatype)
    dty = dtype([(d,v) for d,v in zip (datatype, vals)])
    mat = loadtxt(file,comments="#",dtype=dty)
    Qlats  = reshape(mat["Latitude"],(25,70,64))
    Qlons  = reshape(mat["Longitude"],(25,70,64))
    Qdepts = reshape(mat["Depth(km_BSL)"],(25,70,64))
    Qs  = reshape(mat["Qs"],(25,70,64))

    i= unique(argwhere(Qdepts==1.)[:,0])[0]
    Qs,Qlats,Qlons,Qdepts = [mat[i,:,:].flatten() 
                             for mat in [Qs,Qlats,Qlons,Qdepts] ]
    
    return Qs,Qlats,Qlons
    
# =============================================================================
# ---- OTHER / TEST
# =============================================================================
def mw(M0log):
    return round(2/3*M0log-6.07,1)

from numpy import exp
def tr_mag(m):
    # mat = (exp(linspace(-0.3, 4.45, 33))+45).astype(int)
    mat = (exp(linspace(1, 6, 33))+15).astype(int)
    return mat[argwhere(arange(4.9, 8.2, .1).round(1) == m)[0, 0]]


def test():
    dty = dtype([('date',"U25"),('lon',"f8"),('lat',"f8"),('depth',"f8"),
                 ('Ebb',"f8"),('Ehf',"f8"),('M0',"f8"),
                 ('MeBB',"f8"),('MeHF',"f8"),('delta',"f8"),('Tr',"f8")])
    mat = loadtxt("C:/Users/solen/Documents/Ener/project/cat1.dat",dtype=dty,comments="#")
    
    dty2 = dtype([('date',"U25"),('lat',"f8"),('lon',"f8"),('depth',"f8"),
                    ('mag',"f8"),('MeBB',"f8"),('MeHF',"f8"),
                    ('Tr',"f8"),('Trhf',"f8"),
                    ('Ebb',"f8"),('Ehf',"f8"),('M0',"f8"),
                    ('nstabb',"f8"),('nstahf',"f8")])
    mat2 = loadtxt("C:/Users/solen/Documents/Ener/project/EnerRadTool/5.plot_functions/globalresults.txt",dtype=dty2,comments="#")
    
    fig=plt.figure(figsize=(10,10))
    ax=fig.add_subplot()
    ax2=ax.twiny()
    ax.scatter(log10(mat2["M0"])-7,mat2["Tr"],s=5,c="silver",
                    edgecolor="silver",label="all results")
    scat=ax.scatter(log10(mat["M0"]),mat["Tr"],s=30,c=mat["Tr"],
                    cmap="hot",edgecolor="gray",label="Selected results",vmin=0,vmax=100)
    mags = arange(5,8.1,.1).round(1)
    tlim = array([tr_mag(m) for m in mags])
    ax2.plot(mags,tlim,":k",label="selection for the paper")
    x=array([16,22])
    ax.plot(x,array([1.2*1e-8*(10**(xi+7))**(1/3)*2 for xi in [16,22]]),
            "k",label="duration relation to M0 from Duputel et al. 2012")
    ax2.grid(c = "gainsboro")
    plt.colorbar(scat)
    ax.legend()
    ax.set_xlim(16.3,21.5)
    ax2.set_xlim(mw(16.3),mw(21.5))
    ax.set_ylim(0,145)

from numpy import gradient
def g_grad(dist_list,depth=15,model_name="iasp91"):
    x = arange(3.,82.1,.1).round(1)
    ih,i0,t = request_val_p_train(["P","takeoff","incident","time"],depth,x,
                                model_name=model_name)[0]
    #spl = make_smoothing_spline(x, deepcopy(ih))
    #ih_s = spl(dist_list); 
    f = interpolate.interp1d(x,ih)
    var = gradient(f(dist_list)*pi/180,.5,edge_order=1)*180/pi
    f = make_smoothing_spline(dist_list,var.round(10));
    ind = array([argwhere(x==i)[0,0] for i in dist_list])
    ih = ih[ind]; i0 = i0[ind];
    var = f(dist_list)
    return (2.9*6.8/2.6/5.8
            *sin(ih*pi/180)/sin(dist_list*pi/180)
            /cos(i0*pi/180)*abs(var))**0.5

def g_ref(dist_list,t,ih,i0,ref):  # dimensionless
    ref = argwhere(dist_list>=ref)[0,0]
    
    dist_list,t,ih,i0 = dist_list[ref:],t[ref:],ih[ref:],i0[ref:]
    var = (ih-ih[0])/(dist_list-dist_list[0])
    
    dist,var = numdif(dist_list, ih,dist_list[1]-dist_list[0]) 
    # f = interpolate.interp1d(dist*180/pi,var.round(10));
    # ind = array([argwhere(x==i)[0,0] for i in dist_list])
    # ih = ih[ind]; i0 = i0[ind];
    # var = f(dist_list)
    return dist_list,(2.9*6.8/2.6/5.8
                      *sin(ih*pi/180)/sin(dist_list*pi/180)
                      /cos(i0*pi/180)*abs(var))**0.5

def test_g_diff():
    # XREF = arange(15,95,1)
    # GREF = array([53.7 , 49.55, 45.8 , 42.42, 39.3 , 
    #               36.37, 33.68, 31.29, 28.56, 25.85, 23.17, 20.81, 18.75, 16.68, 15.54,
    #               16.36, 17.03, 17.26, 15.7 , 14.1 , 12.45, 11.9 , 11.39, 10.91, 11.81,
    #               12.76, 13.54, 13.53, 13.24, 12.97, 12.71, 12.46, 12.22, 12.09, 12.12, 
    #               12.14, 12.14, 12.14, 12.14, 12.1 , 11.76, 11.45, 11.14, 10.84, 10.55,
    #               10.27, 10.09,  9.94,  9.78,  9.64,  9.49,  9.35,  9.23,  9.12,  9.02,
    #                8.91,  8.8 ,  8.7 ,  8.6 ,  8.48,  8.36,  8.26,  8.15,  8.04,  7.94,
    #                7.82,  7.71,  7.6,   7.49,  7.39,  7.29,  6.9 ,  6.39,  5.86,  5.3 ,
    #                4.69,  4   ,  3.52,  3.24,  2.94])
    
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    from scipy.interpolate import splrep, splev
    from scipy import interpolate
    from scipy.interpolate import make_smoothing_spline
    from scipy.interpolate import CubicSpline
    
    start = time()
    d_delta = .1
    dist_list = arange(3., 82.5, .5).round(1); depth = 15
    x = deepcopy(dist_list)
    xnew = arange(5., 80.1, 3.).round(1)
    fig2,[ax,ax2] = plt.subplots(2,1,figsize=(10,13))
    axix = ax2.twinx(); axix2 = ax.twinx()
    #ax = fig2.add_subplot()#; ax2 = ax.twinx()
    for model_name,col in zip(["jb",
                                "iasp91","prem",#"ak135",#"pwdk","sp6"
                                ],
                                ["orange",
                                   "r","purple",#"b",#":c",":m"
                                  ]):
        
        
        
        p_times,p_takeo,p_incid = request_val_p_train(["P","time","takeoff","incident"],
                                                      depth,xnew,
                                                      model_name=model_name)[0]
        
        X = arange(5,80.1,.5);G=g(X,depth=15,model_name=model_name)
        gradG=g_grad(X,depth=15,model_name=model_name)
        # G = g_ref(deepcopy(xnew),deepcopy(p_times),
        #         deepcopy(p_incid),deepcopy(p_takeo),5)[1]
        # X = xnew
        invG=1/G; invGgrad = 1/gradG
        #f = interpolate.interp1d(X,G)
        invG[argwhere(G==0)]=nan
        
        X0,G0 = g_ref(deepcopy(xnew),deepcopy(p_times),
                deepcopy(p_takeo),deepcopy(p_incid),27)
        spl = interpolate.interp1d(X0,G0)
        X0 = arange(30,80.1,.5)
        G0 = spl(X0);
        """ 3. smooth nickel pour iasp not jb
            2. smooth et precis => pref
            1. pas smooth mais on voit discontinuité à 20°
            .5 not enough?
        """
        "ON BOUGE PLUS!! d.1 smooth .5 avec diff num 2. sans smooth"
        
        p_takeo= request_val_p_train(["P","takeoff"],
                                     depth,dist_list,
                                     model_name=model_name)[0][0]
        
        #cs = make_smoothing_spline(X, G)
        #cs = CubicSpline(X,G)
        ind = argwhere(X>30)
        ax2.plot(X[ind],G[ind],col,label=f"{model_name} EO2012")
        #ax2.plot(arange(27.,81.,1.),cs(arange(27.,81.,1.)),col,label=f"{model_name} EO2012")
        ax2.plot(X0,G0,ls=':',c=col,label=f"{model_name} KS76")
        axix.plot(dist_list,p_takeo,ls='--',c=col,label=f"{model_name} takeoff angle")
        # axix2.plot(xnew,p_takeo,'--'+col)
        # for d,gi in zip(X,G):
        #     print(f"{d:4.1f} : {gi:10.8f}")
        ax.plot(X,invG,col,alpha=0.5,label=f"{model_name}")
        ax.plot(X,invGgrad,":",c=col,label=f"{model_name} grad")
        #ax.plot(arange(5,80.1,.1),1/f(arange(5,80.1,.1)),"--k")
        #ax.plot(arange(4.,81.,1.),1/cs(arange(4.,81.,1.)),col,label=f"{model_name}")
        """changer debut à un peu moins
        smooth spline best"""
        # ax.plot(X0,1/G0,":"+col,label=f"{model_name}")
        file = f"{UTILS_PATH}/geo_spread/{model_name}.dat"
        # savetxt(file,vstack((X,G)).T,fmt = ["%4.1f","%10.8f"],header="D(°) g")
    # ax.plot(XREF,1/(GREF*6371e-5),'k',lw=2,label="REF",zorder=-10)    
    
    ax.legend(ncol=4)
    ax2.legend(ncol=3,loc = "lower left")
    axix.legend(ncol=1, loc ="upper right")
    for axi in [ax,ax2,axix]:
        axi.set_xlim(5,80)
        axi.xaxis.set_minor_locator(MultipleLocator(2))
        axi.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(.5))
    ax.set_ylim(0,4)
    ax.grid(which="major",c="silver")
    ax.grid(which="minor",c="gainsboro")
    ax.grid(axis = "y", which="minor",c="gainsboro")
    #ax.set_ylim(3,55)
    # ax2.set_ylim(15,55)
    # for l in [15,20,25] :
    #     ax.axvline(l,c="gray",zorder=-15)
    # for i in [23.5,18,16,14.5]:
    #     ax.axvline(i,ls=":",c="lightsalmon",zorder = -10,alpha=0.7)
    #     ax2.axvline(i,ls=":",c="lightsalmon",zorder = -10,alpha=0.7)
    # for i in [24,20,16.5]:
    #     ax.axvline(i,ls=":",c="thistle",zorder = -10,alpha=1.)
    #     ax2.axvline(i,ls=":",c="thistle",zorder = -10,alpha=1.)
    ax2.set_xlabel(r'$\Delta$($°$)')
    ax.set_ylabel(r'$1/g(\Delta)$')
    ax2.set_ylabel(r'$g(\Delta)$')
    axix.set_ylabel(r'$i_h$($°$)')
          
    ax2.grid(which="major",c="silver")
    ax2.grid(which="minor",c="gainsboro")
    # for axi in [ax2,axix]:
    #     axi.set_xlim(30,80)
    ax2.set_ylim(0,0.9)
    axix.set_ylim(15,60)
    
    ax.set_title("Numerical differenciation of g (ref : Ebeling & Okal 2012)")
    ax2.set_title("Comparing to g from Kanamori & Stewart (1976)")
    
    # plt.savefig(f"Compar_model_numdif.png",dpi=200)
    plt.show()
    plt.close()

#test_g_diff()