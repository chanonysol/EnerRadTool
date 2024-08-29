# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:50:28 2024

@author: solen
"""
#----- Pwaves until S waves
# 30 : 10.9 to 11.1 min
# 40 : 13.5 to 13.7 min
# 50 : 15.9 to 16.1 min
# 60 : 18.1 to 18.3 min
# 70 : 20.2 to 20.4 min
# 80 : 22.0 to 22.2 min
#----- Pwaves + 250s
# 30 : 10.2 to 10.3 min     612 to  619 s
# 40 : 11.6 to 11.7 min     698 to  705 s
# 50 : 13.0 to 13.1 min     777 to  784 s
# 60 : 14.2 to 14.3 min     849 to  857 s
# 70 : 15.2 to 15.4 min     914 to  922 s
# 80 : 16.2 to 16.3 min     972 to  980 s

### search every 2 min from 11min
import os
from obspy import read, read_inventory
from obspy.taup.tau import TauPyModel
from multiprocessing import Pool
import colorama
from numpy import vstack, loadtxt, array, argwhere, log10, ndarray, \
    reshape, savetxt, sin, cos, arcsin, pi, median, sqrt, mean, arange
from tqdm import tqdm
import random

UTILS_PATH = os.path.dirname(os.path.realpath(__file__)).replace('\\', '/')

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
    depth,distance,model_name = argum
    model = TauPyModel(model=model_name)
    arrivals = model.get_ray_paths(source_depth_in_km=depth,
                                   distance_in_degree=distance,
                                   phase_list = ['P','pP','sP',"S"])
    diff_time = arrivals[-1].time-arrivals[0].time
    dic = {}
    for arr in arrivals:
        if arr.name not in dic.keys():
            dic[arr.name]={"time": arr.time,
                           "i0": arr.takeoff_angle,
                           "ih": arr.incident_angle}
    del dic["S"]
    dic["max_duration"]=diff_time
    dic["min_duration"]=dic["sP"]['time']-dic["P"]['time']
    return dic

def precalc_time_p_train(min_depth,max_depth,min_dist,max_dist,model_name = "iasp91"):
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
    timing_list = []

    depths = arange(min_depth,max_depth+1,1).astype(int)
    distances = arange(min_dist,max_dist+.2,.1).round(1)

    argums = [(dep,dist,model_name) for dep in depths for dist in distances ]
    with Pool(13) as p:
        for dic in tqdm(p.imap(get_time_p_train,argums),total=len(argums),
                        desc=f"{'making table':>15s}"):
            timing_list.append(dic)

    argums = [(dep,dist) for dep in depths for dist in distances ]
    for name in ["P","pP","sP"]:
        arr_time = reshape(array([dic[name]["time"]
                                         for dic in timing_list]),
                              (len(depths),len(distances)))
        arr_i0 = reshape(array([dic[name]["i0"]
                                         for dic in timing_list]),
                              (len(depths),len(distances)))
        arr_ih = reshape(array([dic[name]["ih"]
                                         for dic in timing_list]),
                              (len(depths),len(distances)))
        savetxt(f"{UTILS_PATH}/travel_times/{model_name}/{name}/arrival_time.dat", arr_time, fmt = "% 7.2f")
        savetxt(f"{UTILS_PATH}/travel_times/{model_name}/{name}/incident_angle.dat", arr_ih, fmt = "% 5.1f")
        savetxt(f"{UTILS_PATH}/travel_times/{model_name}/{name}/takeoff_angle.dat", arr_i0, fmt = "% 7.2f")


    arr = reshape((array([dic["max_duration"]
                                for dic in timing_list])+.5).astype(int),
                     (len(depths),len(distances)))
    savetxt(f"{UTILS_PATH}/travel_times/{model_name}/PtoS_duration.dat", arr, fmt = "%02d")

    arr = reshape((array([dic["min_duration"]
                                for dic in timing_list])+.5).astype(int),
                     (len(depths),len(distances)))
    savetxt(f"{UTILS_PATH}/travel_times/{model_name}/Ptrain_duration.dat", arr, fmt = "%02d")

    with open(f"{UTILS_PATH}/travel_times/{model_name}/(depth_dist)table.csv","w") as file:
        for i,tup in enumerate(argums):
            if i%len(distances)==1:
                file.write(str(argums[i//len(distances)*len(distances):(i//len(distances)+1)*len(distances)])[1:-1]+"\n")

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
    with open(f"{UTILS_PATH}/travel_times/{model_name}/(depth_dist)table.csv",'r') as file:
        lines = file.readlines()
        test = array(lines[0].strip().split("), "))
        for line in lines[1:]:
            test = vstack((test,array(line.strip().split("), "))))
    if isinstance(depth, (int,float)) and isinstance(distance, float):
        depth = [depth] ; distance = [distance]
    if isinstance(depth, (int,float)) and isinstance(distance, (list,ndarray)):
        depth = [depth]*len(distance)
    search = [f"({dep:.0f}, {dist:.1f}" for (dep,dist) in zip(depth,distance)]
    ind = [list(argwhere(test==search[i])[0]) for i in range(len(search))]
    return ind

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

    index = find_depth_dist_ind(depth,distlist)

    if isinstance(arg, str):
        arg =[arg]
    waves = [value for value in arg if value in ["P","pP","sP"]]
    element = [value for value in arg if value in ["time","incident","takeoff"]]

    requests = []
    keys = []
    for wave in waves:
        for el in element:
            if el == "time" : el="arrival_time"
            else : el+="_angle"
            keys.append(f"{wave}-{el}")
            mat = loadtxt(f"{UTILS_PATH}/travel_times/{model_name}/{wave}/{el}.dat")
            requests.append(array([mat[ind[0],ind[1]] for ind in index]))

    if "duration" in arg:
        mat = loadtxt(f"{UTILS_PATH}/travel_times/{model_name}/PtoS_duration.dat")
        matmax = array([mat[ind[0],ind[1]] for ind in index])
        mat = loadtxt(f"{UTILS_PATH}/travel_times/{model_name}/Ptrain_duration.dat")
        matmin = array([mat[ind[0],ind[1]] for ind in index])

        keys.append("tmin-tmax")
        requests.append([max(matmin),min(matmax)])

    return requests,keys
# =============================================================================
# PREPROCESSING FUNCTIONS
# =============================================================================
def Fest(dist_list):  # dimensionless
    return 1.171-7.271e-3*dist_list+6.009e-5*dist_list**2

def g(dist_list,ih,i0):  # dimensionless
    delta = dist_list
    # ih,i0 = request_val_p_train(['P','takeoff','incident'],15,dist_list,
    #                              model_name="iasp91")[0]
    toff25 = request_val_p_train(['P','takeoff'],15,[25],
                                 model_name="iasp91")[0][0][0]
    var = (ih - toff25)/(delta-25)
    return (sin(ih*pi/180)/sin(delta*pi/180)/cos(i0*pi/180)*abs(var))**0.5

def receiver_function(dist_list, incid_angles, alpha = 7):  # dimensionless
    i0 = incid_angles*pi/180
    beta = alpha*3**-0.5
    p = sin(i0)/alpha
    j0 = arcsin(p*beta)
    num = 2/beta**2*(1/beta**2-2*p**2)*cos(i0)
    den = (1/beta**2-2*p**2)**2+(4*p**2*cos(i0)*cos(j0))/(alpha*beta)
    return num/den

def waves_parameters(dist_list,incid_angles):
    ih,i0 = request_val_p_train(['P','takeoff','incident'],15,dist_list,
                                 model_name="iasp91")[0]
    C = receiver_function(dist_list, incid_angles)
    #constants and averages
    rho = 3e3
    alpha = 7e3  # kg/m3 ; m/s
    q = 15.6
    Fp = 16/15
    Fg = Fest(dist_list)
    R = 6371.0e3/g(dist_list,ih,i0)  # m
    Econst = (1+q)*Fp*R**2/Fg*rho*alpha  # Ee=Econst*eflux_sta
    return C, Econst

def snr_mag(mag):
    # mag = 0
    snrlim = mag/3.4
    if snrlim < 1.47:
        return round(20*log10(1.47**2),2)
    return round(20*log10(snrlim**2),2)

def st_order_size(st):
    for i in st:
        i.stats.max = max(abs(i.data))
        i.stats.ind_max = round(argwhere(abs(i.data) == max(abs(i.data)))[
                                0, 0]/i.stats.sampling_rate, 0)
    st.sort(["max"], reverse=True)
    return st

def verif_size(st, freq):
    st = st_order_size(st)
    ind_max = [tr.stats.ind_max for tr in st]
    for i in argwhere(array(ind_max).round(0) > 150 + 60)[::-1, 0]:
        st.remove(st[i])

    maenv = array([tr.stats.max for tr in st])

    freqlim = 2.
    if freq == "HF" : freqlim = 2.

    ind = argwhere(abs(maenv/median(maenv)).round(2) >= freqlim)[:, 0]
    while ind.size!=0:
        for i in ind[::-1]:
            st.remove(st[i])
        maenv = array([tr.stats.max for tr in st])
        ind = argwhere(abs(maenv/median(maenv)).round(2) >= freqlim)[:, 0]

    return st

def rms(tr):
    data = abs(tr.data)
    return sqrt(mean(data**2))
    # return mean((data-mean(data))**2)

def signaltonoise(st):
    Arr = []
    for (i,tr) in enumerate(st):
        noise = tr.copy().trim(tr.stats.starttime, tr.stats.starttime+60)
        signal = tr.copy().trim(tr.stats.starttime+60, tr.stats.starttime+2*60)
        # Arr.append(max(abs(signal.data))/max(abs(noise.data)))
        Arr.append(rms(signal)/rms(noise))
    return 20*log10(array(Arr))

# =============================================================================
# WRITTING FUNCTIONS
# =============================================================================
def write_log(path,filename, st, modif = False):
    reasons = ["REMAIN","over sized data","too much noise","dev too high"]
    col = [colorama.Fore.RED,colorama.Fore.BLACK,colorama.Fore.GREEN,colorama.Fore.BLUE]

    if not modif:
        with open(f"{path}/{filename}.dat","w") as file:
            for tr in st:
                name = f"{tr.stats.network}.{tr.stats.station}.{tr.stats.location}.{tr.stats.channel}"
                index = random.randint(0,3)
                file.write(f"{colorama.Style.BRIGHT + col[index]:15s} ||| {name:15s} ,"
                +f"{reasons[index].center(15):15s} | {tr.stats.latitude:+9.4f} , {tr.stats.longitude:+9.4f} |"
                +f" {tr.stats.starttime} , {tr.stats.endtime} |"
                +f" {tr.stats.sampling_rate} Hz, {tr.stats.npts} samples\n")

    if modif:
        with open(f"{path}/{filename}.dat","w") as file:
            for line in modif:
                file.write(line)

def read_log(path,filename,modif = False, print_log = False):
    with open(f"{path}/{filename}.dat","r") as file:
        lines = file.readlines()
        mat = vstack([array([i.split(",") for i in line.strip().split('|')[3:]]).flatten() for line in lines])
    if print_log:
        for line in lines:
            print(line.strip().replace("      ||| ",""))

    if modif : return lines
    else : return mat


def modify_log(path,filename,names,reason):
    reasons = ["REMAIN","over sized data","too much noise","dev too high"]
    col = [colorama.Fore.RED,colorama.Fore.BLACK,colorama.Fore.YELLOW,colorama.Fore.BLUE]

    index = argwhere(array([reason in i for i in reasons]))[0,0]

    lines = read_log(path,filename,modif = True)
    for name in names:
        for i,line in enumerate(lines):
            if name in line:
                line =  line.replace(line[:15],f"{colorama.Style.BRIGHT + col[index]:15s}")
                line =  line.replace(line[37:37+15+1],f"{reasons[index].center(15):15s} ")
                lines[i] = line

    write_log(path,filename, modif = lines)