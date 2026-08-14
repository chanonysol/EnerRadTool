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

DIR=f"{FILE_PATH}/../1.events_functions"
sys.path.insert(0, DIR)
DIR=f"{FILE_PATH}/../2.data_functions"
sys.path.insert(0, DIR)


from numpy import array, all, any, mean, log10, argwhere, arange, loadtxt, isnan
from generate_event import moment
from time_rupture_functions import joint_tr

from obspy.core.util import AttribDict

def Me(E):
    return 2/3*log10(E)-2.9

def results_dictionnary_init(param):
    results = {'raw':{k:-9999 for k in ['Ebb','Ehf','MeBB','MeHF','Tr',
                                       'Tr_tacer','Tr_lines','theta',
                                       'E_HF/Tr3']},
               'raw_smooth':{k:-9999 for k in ['Ebb','Ehf','MeBB','MeHF','Tr',
                                                  'Tr_tacer','Tr_lines','theta',
                                                  'E_HF/Tr3']},
               'corrected':{k:-9999 for k in ['Ebb','Ehf','MeBB','MeHF','Tr',
                                                  'Tr_tacer','Tr_lines','theta',
                                                  'E_HF/Tr3']},
               'corrected_smooth':{k:-9999 for k in ['Ebb','Ehf','MeBB','MeHF','Tr',
                                                  'Tr_tacer','Tr_lines','theta',
                                                  'E_HF/Tr3']},
               'S_near':{k:-9999 for k in ['Ebb','Ehf','MeBB','MeHF','theta']},
               'S_near_cor':{k:-9999 for k in ['Ebb','Ehf','MeBB','MeHF','theta']},
               
               'information':{}}

    # if param.deviation_correction.type =="result":
    #     results.update({'corrected':{k:None for k in ['Ebb','Ehf','MeBB','MeHF','Tr',
    #                                                       'Tr_tacer','Tr_lines','theta',
    #                                                       'E_HF/Tr3']}})
    #     results.update({'corrected_smooth':{k:None for k in ['Ebb','Ehf','MeBB','MeHF','Tr',
    #                                                       'Tr_tacer','Tr_lines','theta',
    #                                                       'E_HF/Tr3']}})
    return results

def add_results(results, ev, **kwargs):
    M0 = moment(ev.moment,ev.exponent)
    for key in list(kwargs.keys()):
        Ebb, Ehf, dEBB,dEHF, tr_tacer, tr_lign = kwargs[key]
        Tr = joint_tr(tr_tacer, tr_lign)

        MeBB = Me(Ebb)
        MeHF = Me(Ehf/0.25)
        EhfTr = Ehf/Tr**3
        theta = log10(Ebb/M0)

        newdict = {'Ebb': Ebb, 'MeBB': MeBB, 'dEBB': dEBB,
                   'Ehf': Ehf, 'MeHF': MeHF, 'dEHF': dEHF,
                   'Tr': Tr,'theta': theta,
                   'Tr_tacer': tr_tacer if tr_tacer else -9999,
                    'Tr_lines': tr_lign if tr_lign else -9999,
                    'E_HF/Tr3': EhfTr}
        results.update({key:newdict})

def add_results_S(results, ev, **kwargs):
    M0 = moment(ev.moment,ev.exponent)
    for key in list(kwargs.keys()):
        Ebb, Ehf= kwargs[key]

        MeBB = Me(Ebb)
        MeHF = Me(Ehf)
        theta = log10(Ebb/M0)

        newdict = {'Ebb': Ebb, 'MeBB': MeBB, 
                   'Ehf': Ehf, 'MeHF': MeHF, 
                   'theta': theta,
                   }
        results.update({key:newdict})

def update_results(results,**kwargs):
    """
    Add or changes informations for the results dict

    Parameters
    ----------
    results : dict
        Holds all information about the calcultation for writting the
        final results file
    **ev : class Event
        Event information if needed to be added or updated
    **estimate : tuple or list
        updated information about the calculation
        (iteration number (int), distances (list), (BB data (Stream), HF data (Stream)), max duration)
    **time : dict
        estimated running time for each step
        possible keys : ['data_time', 'pre_time', 'ener_time', 'tot_time']

    Returns
    -------
    None.

    """

    if "ev" in kwargs.keys():
        ev = kwargs["ev"]
        reg,sreg = ev.reg,ev.sreg
        if "event" not in results["information"].keys():
            ev_info={"date":ev.date ,"mag":ev.mag , "magtype":"Mw",
                     "M0":moment(ev.moment,ev.exponent),
                     "lat":ev.lat, "lon":ev.lon, "depth":ev.depth,
                     "region":reg,"subregion":sreg}
        else :
            ev_info = results['information']["event"]
            for (element,value) in ev.items():
                ev_info.update({element:value})
        results['information'].update({"event":ev_info})

    if "estimate" in kwargs.keys():
        rt, dist, azlist, stadist, dist_list, (stbb,sthf), time = kwargs["estimate"]
        azlist = concatenate(((azlist[1:]-azlist[:-1]),[360-azlist[-1]+azlist[0]]))
        if "estimate" not in results["information"].keys() or rt==0:
            ener_info={"version":"v2.0.RT" ,"iter": rt if isinstance(rt, int) else None ,
                       "mindist": dist.min(),
                       "maxdist": dist.max(),
                       "nsta_bb":len(list(set([f"{tr.stats.network}.{tr.stats.station}.{tr.stats.location}" for tr in stbb]))),
                       "channel_list_bb":list(set([f"{tr.stats.channel}" for tr in stbb])),
                       "nsta_hf":len(list(set([f"{tr.stats.network}.{tr.stats.station}.{tr.stats.location}" for tr in sthf]))),
                       "channel_list_hf":list(set([f"{tr.stats.channel}" for tr in sthf])),
                       "max_az_gap":azlist.max(),
                       "max_dist_gap":(stadist[1:] - stadist[:-1]).max(),
                       "duration":time,"ref_ratio":25}
        elif rt!=results['information']['estimate']['iter']:
            ener_info = results['information']["estimate"]
            ener_info.update({"iter": rt, "maxdist": dist.max(), "mindist": dist.min(), 
                       "nsta_bb":len(list(set([f"{tr.stats.network}.{tr.stats.station}.{tr.stats.location}" for tr in stbb]))),
                       "channel_list_bb":list(set([f"{tr.stats.channel}" for tr in stbb])),
                       "nsta_hf":len(list(set([f"{tr.stats.network}.{tr.stats.station}.{tr.stats.location}" for tr in sthf]))),
                       "channel_list_hf":list(set([f"{tr.stats.channel}" for tr in sthf])),
                       "max_az_gap":azlist.max()})
        results['information'].update({"estimate":ener_info})

    if any([i in kwargs.keys()
            for i in ['data_time', 'pre_time', 'ener_time', 'tot_time']]):
        if "time" not in results["information"].keys():
            time_info={"data_time":-9999 ,"pre_time":-9999,
                       "ener_time":-9999, "tot_time":-9999}

        else:
            time_info = results['information']["time"]
        for element in [value for value in kwargs.keys()
                        if value in ['data_time', 'pre_time',
                                     'ener_time', 'tot_time']]:
            if time_info[element]!=-9999 and element!="tot_time":
                time_info.update({element: kwargs[element]+time_info[element]})
            else:
                time_info.update({element: kwargs[element]})
        results['information'].update({"time": time_info})

def complete_from_template(results, path = None, name = None, S=False):
    if not path :
        file = sys.stdout
    else :
        file = open(f"{path}/results_{name}.txt","w")
        
    file.write(f"#======================================\n")
    file.write(f"#		         RESULTS\n")
    file.write(f"#======================================\n")
    file.write(f"\n")
    
    if not all(array(list(results['raw'].values()))==-9999):
        Ebb, MeBB, dEBB, Ehf, MeHF, dEHF, Tr, theta, Tr_tacer, Tr_lines, EhfTr3 =\
            list(results['raw'].values())
        file.write(f"#----------- Raw estimation -----------\n")
        file.write(f"Ebb = {Ebb:7.2e}J ; MeBB = {MeBB:4.2f} ; dEbb = {dEBB:.1f}%\n")
        file.write(f"Ehf = {Ehf:7.2e}J ; MeHF = {MeHF:4.2f} ; dEhf = {dEHF:.1f}%\n")
        
        file.write(f"\n")
        file.write(f"Tr = {Tr:3.0f}s; theta = {theta:3.1f}\n")
        file.write(f"(Tr_tacer = {Tr_tacer:3.0f}s)\n")
        file.write(f"(Tr_lines = {Tr_lines:3.0f}s)\n")
        file.write(f"\n")
        file.write(f"E_HF/Tr3 = {EhfTr3:7.2e}J\n")
        file.write(f"E_HF/E_BB = {Ehf/Ebb*100:4.1e}%\n")
        file.write(f"\n")

    if not all(array(list(results['raw_smooth'].values()))==-9999):
        Ebb, MeBB, dEBB, Ehf, MeHF, dEHF, Tr, theta, Tr_tacer, Tr_lines, EhfTr3 =\
            list(results['raw_smooth'].values())
        file.write(f"#smoothed\n")
        file.write(f"Ebb = {Ebb:7.2e}J ; MeBB = {MeBB:4.2f} ; dEbb = {dEBB:.1f}%\n")
        file.write(f"Ehf = {Ehf:7.2e}J ; MeHF = {MeHF:4.2f} ; dEhf = {dEHF:.1f}%\n")
        file.write(f"\n")
        file.write(f"Tr = {Tr:3.0f}s; theta = {theta:3.1f}\n")
        file.write(f"(Tr_tacer = {Tr_tacer:3.0f}s)\n")
        file.write(f"(Tr_lines = {Tr_lines:3.0f}s)\n")
        file.write(f"\n")
        file.write(f"E_HF/Tr3 = {EhfTr3:7.2e}J\n")
        file.write(f"E_HF/E_BB = {Ehf/Ebb*100:4.1e}%\n")
        file.write(f"\n")

    if not all(array(list(results['corrected'].values()))==-9999):
        Ebb, MeBB, dEBB, Ehf, MeHF, dEHF, Tr, theta, Tr_tacer, Tr_lines, EhfTr3 =\
            list(results['corrected'].values())
        file.write(f"#-------- Corrected estimation --------\n")
        file.write(f"Ebb = {Ebb:7.2e}J ; MeBB = {MeBB:4.2f} ; dEbb = {dEBB:.1f}%\n")
        file.write(f"Ehf = {Ehf:7.2e}J ; MeHF = {MeHF:4.2f} ; dEhf = {dEHF:.1f}%\n")
        file.write(f"\n")
        file.write(f"Tr = {Tr:3.0f}s; theta = {theta:3.1f}\n")
        file.write(f"(Tr_tacer = {Tr_tacer:3.0f}s)\n")
        file.write(f"(Tr_lines = {Tr_lines:3.0f}s)\n")
        file.write(f"\n")
        file.write(f"E_HF/Tr3 = {EhfTr3:7.2e}J\n")
        file.write(f"E_HF/E_BB = {Ehf/Ebb*100:4.1e}%\n")
        file.write(f"\n")

    if not all(array(list(results['corrected_smooth'].values()))==-9999):
        Ebb, MeBB, dEBB, Ehf, MeHF, dEHF, Tr, theta, Tr_tacer, Tr_lines, EhfTr3 =\
            list(results['corrected_smooth'].values())
        file.write(f"#smoothed\n")
        file.write(f"Ebb = {Ebb:7.2e}J ; MeBB = {MeBB:4.2f} ; dEbb = {dEBB:.1f}%\n")
        file.write(f"Ehf = {Ehf:7.2e}J ; MeHF = {MeHF:4.2f} ; dEhf = {dEHF:.1f}%\n")
        file.write(f"\n")
        file.write(f"Tr = {Tr:3.0f}s; theta = {theta:3.1f}\n")
        file.write(f"(Tr_tacer = {Tr_tacer:3.0f}s)\n")
        file.write(f"(Tr_lines = {Tr_lines:3.0f}s)\n")
        file.write(f"\n")
        file.write(f"E_HF/Tr3 = {EhfTr3:7.2e}J\n")
        file.write(f"E_HF/E_BB = {Ehf/Ebb*100:4.1e}%\n")
        file.write(f"\n")
        
    if S :
        file.write(f"#------------ S waves near ------------\n")
    
        Ebb, MeBB, Ehf, MeHF, theta =\
            list(results['S_near'].values())
        file.write(f"Ebb = {Ebb:7.2e}J ; MeBB = {MeBB:4.2f} \n")
        if not all(array(list(results['corrected_smooth'].values()))==-9999):
            Ebbp, MeBBp, dEBBp, _,_,_, Tr, _, _, _, _ = list(results['corrected_smooth'].values())
            file.write(f"dEp = {log10(Ebb/Ebbp):+5.2f} ; dMep = {MeBB-MeBBp:+5.2f}\n")
        elif not all(array(list(results['raw_smooth'].values()))==-9999):
            Ebbp, MeBBp, dEBBp, _,_,_, Tr, _, _, _, _ = list(results['raw_smooth'].values())
            file.write(f"dEp = {log10(Ebb/Ebbp):+5.2f} ; dMep = {MeBB-MeBBp:+5.2f}\n")
        # file.write(f"Ehf = {Ehf:7.2e}J ; MeHF = {MeHF:4.2f} \n")
        file.write(f"\n")
        file.write(f"theta = {theta:3.1f}\n")
        file.write(f"\n")
        
        # if all(array(list(results['S_near_cor'].values()))!=-9999):
        #     Ebb, MeBB, Ehf, MeHF, theta =\
        #         list(results['S_near_cor'].values())
            
        # file.write(f"#-------- S waves near correct ---------\n")
    
        # Ebb, MeBB, Ehf, MeHF, theta =\
        #     list(results['S_near_cor'].values())
        # file.write(f"Ebb = {Ebb:7.2e}J ; MeBB = {MeBB:4.2f} \n")
        # file.write(f"Ehf = {Ehf:7.2e}J ; MeHF = {MeHF:4.2f} \n")
        # file.write(f"\n")
        # file.write(f"theta = {theta:3.1f}\n")
        # file.write(f"\n")

    file.write(f"#======================================\n")
    file.write(f"#	          INFORMATION\n")
    file.write(f"#======================================\n")
    file.write(f"\n")

    date, mag, magtype, M0, lat, lon, depth, region, subregion=\
        list(results['information']['event'].values())
    file.write(f"#---------- Event information ---------\n")
    file.write(f"# {str(date):s}\n")
    file.write(f"# Magnitude {mag:3.1f} {magtype}\n")
    file.write(f"# Moment {M0:8.2e} (N.m)\n")
    file.write(f"# {lat:+6.2f} {lon:+7.2f} {depth:2.0f}\n")
    file.write(f"# Region : {region:15s}\n")
    file.write(f"# Subregion : {subregion:15s}\n")
    file.write(f"\n")

    version, iter, mindist, maxdist, nsta_bb, channel_list_bb, nsta_hf, \
        channel_list_hf, az_gap, dist_gap, duration, ref_ratio=\
        list(results['information']['estimate'].values())
    file.write(f"#------- Estimation information -------\n")
    file.write(f"# {version}\n")
    file.write(f"# iteration n°{iter} : {mindist:2.0f}-{maxdist:2.0f}\n")
    file.write(f"# BB : {nsta_bb:2.0f} stations [{channel_list_bb}]\n")
    file.write(f"# HF : {nsta_hf:2.0f} stations [{channel_list_hf}]\n")
    file.write(f"# Time after P-waves : {duration:3.0f}s\n")
    file.write(f"# reference used for mEhf : {ref_ratio:2.0f}%\n")
    file.write(f"\n")

    data_time, pre_time, ener_time, tot_time =\
        list(results['information']['time'].values())
    file.write(f"#----------- Estimation time ----------\n")
    file.write(f"# Data collection : {data_time:3.0f}s\n")
    file.write(f"# Preprocessing : {pre_time:3.0f}s\n")
    file.write(f"# Energy estimation : {ener_time:3.0f}s\n")
    file.write(f"# TOTAL : {tot_time:3.0f}s\n")

    if path:
        file.close()

def read_results(filepath):
    with open(filepath,"r") as file:
        lines_file = file.readlines()
    indices = argwhere(array([l in ["#----------- Raw estimation -----------\n",
                                    "#smoothed\n",
                                    "#-------- Corrected estimation --------\n",
                                    "#---------- Event information ---------\n",
                                    "#------- Estimation information -------\n",
                                    "#----------- Estimation time ----------\n"
                                    ] for l in lines_file]))[:,0]
    results = {}
    for ind,name in zip(indices[:4],["raw","raw_s","corr","corr_s"]):
        lines = lines_file[ind+1:ind+9]
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
        results[name]=dic
        
    ind = indices[4]
    lines = lines_file[ind+1:ind+7]
    dic={}
    results["ev"]=dic
    
    # ind = indices[5]
    # lines = lines_file[ind+1:ind+7]
    # lines = [i.strip()[2:] for i in lines]
    # dic={}
    # dic["version"] = lines[0]
    # dic["iteration"] = int(lines[1].split()[1][-1])
    # dic["distance"] = [int(i) for i in lines[1].split()[-1].split("-")]
    # dic["version"] = 
    # dic["version"] = 
    # dic["version"] = 
    # dic["version"] = 
    # results["info"]=dic
    
    # ind = indices[6]
    # lines = lines_file[ind+1:]
    # dic={}
    # results["times"]=dic
    
    return AttribDict(results)

def write_station_log(file_sta,st_bb,st_hf, inv):
    st_hf_names = [tr.id for tr in st_hf]
    st_bb_names = [tr.id for tr in st_bb]
    test = st_hf_names+st_bb_names
    test = list(set(test))
    test.sort()
    with open(file_sta,"a") as write_file:
        write_file.write(f"#{'station':15s} {'use':4s} {'az'.rjust(6):6s} {'dist':4s} {'lon'.rjust(8):8s} {'lat'.rjust(8):8s} \n")
        for name in test :
            index_bb = argwhere(array(st_bb_names)==name)
            index_hf = argwhere(array(st_hf_names)==name)
            net,sta = name.split(".")[:2]
            info = inv.select(network = net, station = sta)[0][0]
            if index_bb.size !=0 and index_hf.size!=0:
                index = index_bb[0,0]
                use = "both"
            elif index_bb.size !=0 :
                index = index_bb[0,0]
                use = "BB"
            elif index_hf.size!=0:
                index = index_hf[0,0]
                use = "HF"
            write_file.write(f"{name:15s} {use:4s} {info.extra.station_param.az:6.2f} " 
                             +f"{info.extra.station_param.distance:5.2f} "
                             +f"{info.longitude:+8.3f} {info.latitude:+8.3f}\n")
    return st_bb_names,st_hf_names

def write_raw_results_sta(file,ener_sta,ener,ener_s,names,rt=None):
    start = None
    end = None
    try : open(file,"r")
    except :
        with open(file,"w") as write_file:
            write_file.write(f'#{"time(s)":15s} {" ".join([f"{e:9.0f}" for e in arange(0,len(ener),1)])}\n')
    else:
        if rt and rt>0:
            with open(file,"r") as write_file:
                lines = write_file.readlines()
            index = argwhere(array([l[:6]=="#mean " for l in lines]))[:,0]
            start = "".join(lines[:index[rt-1]+2])
            if len(index)>=(rt+1):
                end = "".join(lines[index[rt]+2:])
            with open(file,"w") as write_file:
                write_file.write(start)
    
    with open(file,"a") as write_file:
        for i in range(len(ener_sta)):
            test = ener_sta[i,:]
            name = names[i]
            write_file.write(f'{name:15s} {" ".join([f"{e:9.3e}" for e in test])}\n')
        write_file.write(f'#{"mean":15s} {" ".join([f"{e:9.3e}" for e in ener])}\n')
        write_file.write(f'#{"mean_smoothed":15s} {" ".join([f"{e:9.3e}" for e in ener_s])}\n')
        if end:
            write_file.write(end)
            
def read_results_sta(file,rt=None):
    time = (loadtxt(file,dtype =str,comments = None)[0,1:]).astype(int)
    ener_sta_bb = loadtxt(file,
                          usecols = list(arange(1,time.max()+2).astype(int)),
                          skiprows = 1)
    names = loadtxt(file, usecols = [0],skiprows = 1,dtype=str)
    if isinstance(rt,int):
        with open(file,"r") as file:
            lines = file.readlines()[1:]
        index = argwhere(array([l[:6]=="#mean " for l in lines]))[:,0]
        ener_sta_bb = ener_sta_bb[index[rt-1]+2 if rt!=0 else 0:index[rt]]
        names = names[index[rt-1]+2 if rt!=0 else 0:index[rt]]
    return time, ener_sta_bb, names

def read_sta_log(file,hf=False):
    with open(f"{file}/sta_log.dat",'r') as file:
        lines=file.readlines()
    stalog = {}
    for line in lines[1:]:
        name,use,az,dist,lon,lat = line.strip().split()
        az,dist,lon,lat = float(az),float(dist),float(lon),float(lat)
        stalog[name]={'use': use, 'az':az, 'dist' : dist,'lon':lon,'lat':lat}
    uses = array([sta['use'] for sta in stalog.values()])
    if hf :
        index = argwhere((uses=='HF')| (uses=='both'))[:,0]
    else:
        index = argwhere((uses=='BB')| (uses=='both'))[:,0]
    names = array(list(stalog.keys()))[index]
    info = array(list(stalog.values()))[index]
    stalog = {}
    for name,inf in zip(names,info):
        stalog[name] = inf
    return names, stalog

# =============================================================================
# Make a PDF
# =============================================================================
from fpdf import FPDF
from glob import glob
from numpy import concatenate
from tqdm import tqdm
class PDF(FPDF):
    def __init__(self):
        super().__init__()
    def header(self):
        self.set_font('Arial', '', 12)
        self.cell(0, 7, 'Energy estimation report', "B", 1, 'C')
        
    def footer(self):
        self.set_y(-10)
        self.set_font('Arial', '', 7)
        self.set_text_color(82,0,12)
        self.cell(0, 5, '''Solution provided by R-CET: Rapid Characterization of Earthquake and Tsunami
                        A GNS Science Led Research Programme''', 0, 0, 'L')
        self.image(f'EnerRadTool/4.PostProcess_functions/RCETlogo.png', 
                   x = 178, y = 285, w=20, h=10, type = 'PNG')

def read_result_info(filepath):
    with open(filepath,"r") as file:
        lines = file.readlines()
    ind = argwhere(array([i[-12:] for i in lines]) == "INFORMATION\n")[0,0]
    lines = lines[ind-1:]
    vals = [f"{' '*5}"+i for i in [i.strip()[2:] #if j not in [0,2,3,10,17]
                        #else i.strip()[2:-2]  
                        for j,i in enumerate(lines)] 
            if i!=""]
    vals[0] = vals[0][:-10]; vals[2] = vals[2][:-10]
    return vals

def make_report(resultpath,correct=False):
    res = read_result_info(glob(f"{resultpath.replace("[","?").replace("]","?")}"
                                 +"/results*")[-1]) 
    
    #img=f'{resultpath}/EnerOverTime_raw.png'
    if not correct :
        img2=f'{resultpath}/EnerOverTime_raw.png'
        img8=f'{resultpath}/Res2Cat_raw.png'
        img9=f'{resultpath}/apparentStress_raw.png'
    else : 
        img2=f'{resultpath}/EnerOverTime_corrected.png'
        img8=f'{resultpath}/Res2Cat_corrected.png'
        img9=f'{resultpath}/apparentStress_corrected.png'
    img3=f'{resultpath}/Signals_DistColor_selected.png'
    img4=f'{resultpath}/Signals_AzColor_selected.png'
    #img5=f'{resultpath}/Spectrum_selected.png'
    img6=f'{resultpath}/station_contrib_map_BB.png'
    #img7=f'{resultpath}/station_contrib_map_HF.png'
    
    p= tqdm(total = 10,desc = "Compiling image into pdf")
    
    pdf = PDF() # Instance of custom class
    # Margin
    MARGIN = 10 
    pdf.add_page()
    p.update(1)
    
    #pdf.cell(w=20, h=2, border=0, ln=1)
    pdf.set_font('Arial', '', 6)
    pdf.cell(0,5, '''Note: These results are currently evaluated by a fully automated process.
              Users should be advised to use and interpret these products with caution.''', 0, 1, 'C')
    p.update(1)
    
    pdf.set_font('Times', '', 8)
    pdf.multi_cell(60, 4, txt="\n".join(res[3:10]), align="L", border = 0)
    pdf.set_xy(x=75, y= 22)
    pdf.multi_cell(62, 4, txt="\n".join(res[10:17]), align="L", border = 0)
    pdf.set_xy(x=140, y= 22)
    pdf.multi_cell(60, 4, txt="\n".join(res[17:]), align="L", border = 0)
    p.update(1)
    
    pdf.image(img3, x = MARGIN + 80, y = 45, w=125, h=65, type = 'PNG'); p.update(1)  
    pdf.image(img4, x = MARGIN + 80, y = 45+55, w=125, h=65, type = 'PNG'); p.update(1)  
    
    pdf.image(img6, x = 0+MARGIN, y = 55, w=80, h=90, type = 'PNG'); p.update(1)  
   
    pdf.image(img8, x = -5+MARGIN, y = 142, w=90, h=90, type = 'PNG'); p.update(1)  

    pdf.image(img9, x = 0+MARGIN, y = 138+90, w=80, h=60, type = 'PNG'); p.update(1)  
    
    pdf.image(img2, x = MARGIN + 85, y = 148, w=100, h=150, type = 'PNG'); p.update(1)  

    pdf.output(f'{resultpath}/../../report_{resultpath.split('/')[-1]}.pdf', 'F')
    p.update(1)  
    p.close()
    
    # # 1. Open your images
    # img1 = Image.open(img6)
    # img2 = Image.open(img3)
    # img3 = Image.open(img4)

    # # 2. Create a new blank image (e.g., side-by-side)
    # total_width = img1.width + img2.width
    # max_height = max(img1.height, img2.height)
    # new_img = Image.new('RGB', (total_width, max_height))

    # # 3. Paste them onto the canvas
    # new_img.paste(img1, (0, 0))
    # new_img.paste(img2, (img1.width, 0))
    # new_img.paste(img3, (img1.width, img2.height))
    # new_img.save("temp_combined.jpg")

    # # 4. Convert the combined image to PDF
    # with open(f"{resultpath}/Stickers.pdf", "wb") as f:
    #     f.write(img2pdf.convert("temp_combined.jpg"))