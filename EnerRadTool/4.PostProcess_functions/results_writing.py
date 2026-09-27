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


from numpy import array, all, any, mean, log10, argwhere, arange, loadtxt, isnan, nan
from generate_event import moment
from time_rupture_functions import joint_tr

from obspy.core.util import AttribDict
from obspy import UTCDateTime

#TODO add this in data_utils
# =============================================================================
#---- MAG FUNCTIONS
# =============================================================================
def Me(E):
    return 2/3*log10(E)-2.9

def Mehf(E):
    return 2/3*log10(E/0.25)-2.9

def inv_Me(Me):
    return 10**((Me+2.9)*3/2)

def inv_Mehf(Me):
    return 10**((Me+2.9)*3/2)*0.25

# =============================================================================
#---- results storage and updating 
# =============================================================================
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
    # event information --------------------------------------------
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
        
    # calcul information -------------------------------------------
    if "estimate" in kwargs.keys():
        rt, dist, (azgap,distgap), (stbb,sthf), time = kwargs["estimate"]
        if "estimate" not in results["information"].keys() or rt==0:
            ener_info={"version":"v2.3.RT" ,"iter": rt if isinstance(rt, int) else None ,
                       "mindist": dist.min(),
                       "maxdist": dist.max(),
                       "nsta_bb":len(list(set([f"{tr.stats.network}.{tr.stats.station}.{tr.stats.location}" for tr in stbb]))),
                       "channel_list_bb":list(set([f"{tr.stats.channel}" for tr in stbb])),
                       "nsta_hf":len(list(set([f"{tr.stats.network}.{tr.stats.station}.{tr.stats.location}" for tr in sthf]))),
                       "channel_list_hf":list(set([f"{tr.stats.channel}" for tr in sthf])),
                       "max_az_gap": azgap,
                       "max_dist_gap": distgap,
                       "duration":time,"ref_ratio":25}
        elif rt!=results['information']['estimate']['iter']:
            ener_info = results['information']["estimate"]
            ener_info.update({"iter": rt, "maxdist": dist.max(), "mindist": dist.min(), 
                       "nsta_bb":len(list(set([f"{tr.stats.network}.{tr.stats.station}.{tr.stats.location}" for tr in stbb]))),
                       "channel_list_bb":list(set([f"{tr.stats.channel}" for tr in stbb])),
                       "nsta_hf":len(list(set([f"{tr.stats.network}.{tr.stats.station}.{tr.stats.location}" for tr in sthf]))),
                       "channel_list_hf":list(set([f"{tr.stats.channel}" for tr in sthf])),
                       "max_az_gap":azgap,
                       "max_dist_gap": distgap})
        results['information'].update({"estimate":ener_info})
    # certainty information -------------------------------------------    
    if "certain" in kwargs.keys():
        rt, [data, levels, levels_name, mark, general] = kwargs["certain"]
        keys = ["reg_fit","az_gap","tr_gap","dEbb","dEhf","tr_moment","dist_gap","total"]
        if "estimate" not in results["information"].keys() or rt==0:
            certain_info=dict([(k,{"val":v,"name":n}) 
                               for k,v,n in zip(keys,
                                                 concatenate((levels,[mark])),
                                                 concatenate((levels_name,[general])))])
        elif rt==results['information']['estimate']['iter']:
            certain_info = results['information']["certain"]
            certain_info.update(dict([(k,{"val":v,"name":n}) 
                               for k,v,n in zip(keys,
                                                 concatenate((levels,[mark])),
                                                 concatenate((levels_name,[general])))]))
        results['information'].update({"certain":certain_info})
    # time information -------------------------------------------      
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
# =============================================================================
#---- Write results file
# =============================================================================
def complete_from_template(results, path = None, name = None, S=False):
    if not path :
        file = sys.stdout
    else :
        file = open(f"{path}/results_{name}.txt","w")
        
    file.write(f"#======================================\n")
    file.write(f"#		         RESULTS\n")
    file.write(f"#======================================\n")
    file.write(f"\n")
    
    # RAW results
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
        file.write(f"E_HF/E_BB = {Ehf/Ebb*100:4.1f}%\n")
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
        file.write(f"E_HF/E_BB = {Ehf/Ebb*100:4.1f}%\n")
        file.write(f"\n")

    # CORRECTED results
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
        file.write(f"E_HF/E_BB = {Ehf/Ebb*100:4.1f}%\n")
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
        file.write(f"E_HF/E_BB = {Ehf/Ebb*100:4.1f}%\n")
        file.write(f"\n")
        
    if S :
        file.write(f"#------------ S waves near ------------\n")
    
        Ebb, MeBB, Ehf, MeHF, theta =\
            list(results['S_near'].values())
        file.write(f"Ebb = {Ebb:7.2e}J ; MeBB = {MeBB:4.2f} \n")
        file.write(f"#Pwaves-comparison\n")
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

    # Event
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
    
    # Param
    version, iter, mindist, maxdist, nsta_bb, channel_list_bb, nsta_hf, \
        channel_list_hf, az_gap, dist_gap, duration, ref_ratio=\
        list(results['information']['estimate'].values())
    file.write(f"#------- Estimation information -------\n")
    file.write(f"# {version}\n")
    file.write(f"# iteration n°{iter} : {mindist:2.0f}-{maxdist:2.0f}\n")
    file.write(f"# network azimuthal gap : {az_gap:3.0f}°\n")
    file.write(f"# network distance gap : {dist_gap:3.0f}°\n")
    file.write(f"# BB : {nsta_bb:2.0f} stations [{channel_list_bb}]\n")
    file.write(f"# HF : {nsta_hf:2.0f} stations [{channel_list_hf}]\n")
    file.write(f"# Time after P-waves : {duration:3.0f}s\n")
    file.write(f"# reference used for mEhf : {ref_ratio:2.0f}%\n")
    file.write(f"\n")
    
    # Time
    data_time, pre_time, ener_time, tot_time =\
        list(results['information']['time'].values())
    file.write(f"#----------- Estimation time ----------\n")
    file.write(f"# Data collection : {data_time:3.0f}s\n")
    file.write(f"# Preprocessing : {pre_time:3.0f}s\n")
    file.write(f"# Energy estimation : {ener_time:3.0f}s\n")
    file.write(f"# TOTAL : {tot_time:3.0f}s\n")
    
    # Certainty
    if "certain" in results["information"].keys():
        certain_dic = list(results['information']['certain'].values())
        text = ["Theta reg fit","station az gap","TR variation","Ebb variation",
                "Ehf variation","TR theorique","station dist gap","CONFIDENCE LEVEL"]
        file.write(f"#----------- Certainty level ----------\n")
        file.write("#----- 0 highest - 6 or 34 lowest -----\n")
        for t,d in zip(text,certain_dic):
            file.write(f"# {t:17s} : {d['val']:2.0f} {d['name']:5s}\n")
        
    if path:
        file.close()

def write_rt_cat(ev,results):
    resultsfile = f"{FILE_PATH}/../6.database/results.dat"
    if not os.path.exists(resultsfile):
        with open(resultsfile,"w") as file:
           file.write(f"#{'DateTime':19s} {'lat':7s} {'lon':7s} "
                          +f"{'region':12s} {'zone':12s} "
                           +f"{'moment':8s} {'mw':4s} "
                           +f"{'Ebb':8s} {'dbb':3s} {'MeBB':3s} "
                           +f"{'Ehf':8s} {'dhf':3s} {'Mehf':3s} "
                           +f"{'Tr':3s} {'lines':3s} {'tacer':3s} "
                           +f"{'theta':4s}\n" ) 
    
    else :
        evline = (f"{str(ev.date)[:-8]:19s} {ev.lat:+7.2f} {ev.lon:+7.2f} "
                      +f"{ev.reg:12s} {ev.sreg:12s} "
                       +f"{moment(ev.moment,ev.exponent):8.2e} {ev.mag:4.2f} ")
        with open(resultsfile,"r") as file:
            lines = file.readlines()
        
        ind = argwhere(array([l.find(evline) for l in lines])>-1)
        if ind.size!=0:
            for i in ind[::-1,0]:
                lines.pop(i)
            with open(resultsfile,"w") as file:
                for l in lines:
                    file.write(l)  

    with open(resultsfile,"a") as file:
        res = results["corrected_smooth"]
        if "dEBB"  not in res.keys():
            res["dEBB"]=-9999; res["dEHF"]=-9999
        for k,v in zip(res.keys(),res.values()):
            if v ==-9999 :
                res[k]=nan
        res = AttribDict(res)
        file.write(f"{str(ev.date)[:-8]:19s} {ev.lat:+7.2f} {ev.lon:+7.2f} "
                      +f"{ev.reg:12s} {ev.sreg:12s} "
                       +f"{moment(ev.moment,ev.exponent):8.2e} {ev.mag:4.2f} "
                       +f"{res.Ebb:8.2e} {res.dEBB:3.2f} {res.MeBB:3.2f} "
                       +f"{res.Ehf:8.2e} {res.dEHF:3.2f} {res.MeHF:3.2f} "
                       +f"{res.Tr: 3.0f} {res.Tr_lines: 3.0f} {res.Tr_tacer: 3.0f} "
                       +f"{res.theta:+4.1f}\n" ) 
# =============================================================================
# ---- READING LOG AND RESULTS
# =============================================================================

def read_results(filepath):
    with open(filepath,"r") as file:
        lines_file = file.readlines()
    indices = argwhere(array([l in ["#----------- Raw estimation -----------\n",
                                    "#smoothed\n",
                                    "#-------- Corrected estimation --------\n",
                                    "#---------- Event information ---------\n",
                                    "#------- Estimation information -------\n",
                                    "#----------- Estimation time ----------\n",
                                    "#----------- Certainty level ----------\n"
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
    
    for ind,ind_next,name in zip(indices[4:],
                                 concatenate((indices[5:],[-1])),
                                 ["ev","info","time","eval"]):   
        lines = lines_file[ind+1:ind_next if ind_next!=-1 else None]
        if ind==indices[6]:
            vals = [i for i in 
                    concatenate([i.strip().split(" : ") for i in lines])][1::2]
        elif ind==indices[7]:
            vals = [i for i in 
                    concatenate([i.strip().split(" : ")[1:] for i in lines])]
        elif ind==indices[5]:
            vals = [i 
                    for i in concatenate([[i] if i.find("[")!=-1  else i.split()
                                          for i in concatenate([i.strip().split(" : ") 
                                                                for i in lines])
                                          if i not in ['# network azimuthal gap',
                                                       '# network distance gap',
                                                       '# BB','# HF',
                                                       '# Time after P-waves',
                                                       '# reference used for mEhf']]) 
                    if i not in ["","#",'iteration','stations']]
        
        dic = {}
        if ind==indices[6]:
            for i,n in zip(vals,["data","proc","calc","total"]):
                dic[n]=float(i[:-1])
            results[name]=dic
        elif ind==indices[7]:
            for i,n in zip(vals,["reg_fit","az_gap","tr_gap","dEbb","dEhf",
                                 "tr_moment","dist_gap","total"]):
                dic[n]={"val":int(i.split()[0]),"name":i.split()[1]}
            results[name]=dic
        elif ind==indices[5]:
            for i,n in zip(vals,["version","iter","dist","azgap","distgap",
                                 "bb_sta","hf_sta","sig_length","mehf_ref"]):
                if n in ["bb_sta","hf_sta"]:
                    dic[n]={"nsta":int(i.split()[0]),
                            "channel":i[i.find("[")+2:-2].replace("'","").split(", ") }
                elif n=="dist":
                    dic[n]={"min":int(i.split("-")[0]),
                            "max":int(i.split("-")[1])}
                else:
                    i = i.replace("°","").replace("%","")
                    if i[0]=="v":
                        dic[n]=i
                    elif i[0]=="n":
                        dic[n]=int(i[-1])
                    elif n=="mehf_ref":
                        dic[n]=float(i)/100
                    else:
                        dic[n]=int(i.replace("s",""))
            results[name]=dic
        else :
            vals = [i for i in concatenate([i.strip().replace("# ","").split(" ") for i in lines]) 
                    if i!=""]
            dic = {"date":UTCDateTime(vals[0]),
                   "mw" : float(vals[2]), "M0": float(vals[5]),
                   "lat": float(vals[7]),"lon":float(vals[8]),"de":float(vals[9]),
                   "reg":vals[12],"sreg":"-".join(vals[15:])}
            results[name]=dic
            
    
    return AttribDict(results)

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
        if line[0]=="#":continue
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
# ---- WRITING LOG
# =============================================================================
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
            

# =============================================================================
# ---- Make a PDF
# =============================================================================
from matplotlib.backends.backend_pdf import PdfPages
from glob import glob
from numpy import concatenate
from tqdm import tqdm

def read_result_info(filepath):
    with open(filepath,"r") as file:
        lines = file.readlines()
    ind = argwhere(array([i[-12:] for i in lines]) == "INFORMATION\n")[0,0]
    lines = lines[ind-1:]
    vals = [f"{' '*5}"+i for i in [i.strip()[2:] #if j not in [0,2,3,10,17]
                        #else i.strip()[2:-2]  
                        for j,i in enumerate(lines)] 
            if i!=""]
    vals = vals[3:]; vals.pop(15); vals.pop(13)
    return vals

def make_report(resultpath,correct=False):
    res = read_result_info(glob(f"{resultpath.replace("[","?").replace("]","?")}"
                                 +"/results*")[-1]) 
    final = int(resultpath.split("_")[-2][2:-1].split('.')[-1])-1
    #img=f'{resultpath}/EnerOverTime_raw.png'
    if not correct :
        img2=f'{resultpath}/EnerOverTime_raw{final}.png'
        img8=f'{resultpath}/Res2Cat_raw.png'
        img9=f'{resultpath}/apparentStress_raw.png'
    else : 
        img2=f'{resultpath}/EnerOverTime_corrected{final}.png'
        img8=f'{resultpath}/Res2Cat_corrected.png'
        img9=f'{resultpath}/apparentStress_corrected.png'
    img3=f'{resultpath}/Signals_DistColor_selected.png'
    img4=f'{resultpath}/Signals_AzColor_selected.png'
    #img5=f'{resultpath}/Spectrum_selected.png'
    img6=f'{resultpath}/station_contrib_map_BB.png'
    #img7=f'{resultpath}/station_contrib_map_HF.png'
      
    
    level = int(res[-1].strip().split()[-2])
    levels, level_c, level_a = (["GREAT","GOOD","OK","LIMIT","BAD"],
                                ["aquamarine","yellowgreen","khaki","orange","salmon"], 
                                [0.3,0.3,0.4,0.3,0.25])
    
    from PIL import Image
    from numpy import asarray
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches
    from numpy.ma import masked_where
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import datetime
    
    cm = 1/2.54

    p= tqdm(total = 23,desc = "Compiling image into pdf")

    fig2 = plt.figure(2, figsize=(21*cm,29.7*cm),dpi=1000)

    gs0 = gridspec.GridSpec(7,6, figure=fig2, height_ratios=[.3,1.2,2,2,3,2,.4], wspace=0.01, hspace=-0.05)
    fig2.add_subplot(gs0[0,:]); p.update(1)
    for j in range(0,gs0.ncols,2):
        fig2.add_subplot(gs0[1,j:j+2])
        p.update(1)
    fig2.add_subplot(gs0[2:4,:3]); p.update(1)
    for i in range(1,gs0.nrows-1):
        if i in [4,5]:
            fig2.add_subplot(gs0[i,:3])
        if i in [2,3]:
            fig2.add_subplot(gs0[i,3:])
        p.update(1)
    fig2.add_subplot(gs0[4:6,3:]); p.update(1)
    for i,j in enumerate([0,3,5]):
        fig2.add_subplot(gs0[-1,j:j+3-i])
        p.update(1)        
    #format_axes(fig2)
    p.update(1)

    logo = 'EnerRadTool/4.PostProcess_functions/RCETlogo.png'
    img = asarray(Image.open(logo))
    img = masked_where(img == [255, 255, 255,   0], img)
    fig2.axes[-1].imshow(img); p.update(1)


    fig2.axes[0].text(0.5, .55, 'Energy estimation report',horizontalalignment='center', size='x-small')
    fig2.axes[0].axhline(y=0.35, linewidth=.7, color='k')
    fig2.axes[0].text(0.5, 0., '''Note: These results are currently evaluated by a fully automated process.        Users should be advised to use and interpret these products with caution.''',horizontalalignment='center', style="italic", size=5.5)

    fig2.axes[1].text(0.05, 0.5, "\n".join(res[:7]),verticalalignment ="center",horizontalalignment='left',size=6.5)
    fig2.axes[2].text(0.05, 0.5, "\n".join(res[7:14]),verticalalignment ="center",horizontalalignment='left',size=6.5)
    fig2.axes[3].text(0.05, 0.46, "\n".join(res[14:19]),verticalalignment ="center",horizontalalignment='left',size=5.5)
    
    fig2.axes[-3].text(0.6, 0.4, '''Solution provided by R-CET: Rapid Characterization of Earthquake and Tsunami''',horizontalalignment='center',  size=5.5,c="maroon")
    fig2.axes[-2].text(0.65, 0.4, '''A GNS Science Led Research Programme''',horizontalalignment='center',  size=5.5, c="maroon")
    
    for i in [0,1,2,3,-3,-2,-1]:
        fig2.axes[i].set_axis_off()

    for i,im,k,l,v,w in zip([4,5,6,7,8,9],[img6,img3,img4,img8,img9,img2],
                    [0,210,240,80,0,370],[0,255,215,50,50,225],
                    [0,110,100,50,0,0],[0,675,635,150,150,0]):
        #if i in [4,7,8]:continue
        img = asarray(Image.open(im))
        if i in [5,6]:
            a=.8
        else : 
            a=1
            if i == 8 : a = .9
            img = masked_where(img == [255, 255, 255,   0], img)
        fig2.axes[i].imshow(img[k:-1-l,v:-1-w],aspect=a)
        p.update(1)
        fig2.axes[i].set_axis_off()

    divider = make_axes_locatable(fig2.axes[3])
    ax2 = divider.append_axes("bottom", size="40%", pad=0.1, sharex=fig2.axes[3])

    for i,(l,c,a) in enumerate(zip(levels,level_c,level_a)):
        rect = mpatches.Rectangle((0,.7), 35-i*7, 1.1, linewidth=1,zorder=-10,
                                  edgecolor=c, facecolor=c,alpha=a)
        ax2.add_patch(rect)
    ax2.axvline(35-level,ymin=.7,ymax=1,c="k",lw=1)
    ax2.set_xlim(0,7*5)
    ax2.set_ylim(0,1)
    ax2.set_xticks(arange(0,35,7)[::-1]+3.5,labels =levels,size=4.5)
    ax2.xaxis.set_ticks_position('top')
    ax2.tick_params(axis='x', pad=-.1)
    ax2.set_yticks([])
    ax2.text(35/2, .6, f"Confidence level estimates : {levels[int(level//7)]}",
                      verticalalignment ="top", horizontalalignment='center',
                      size=5.5)
    
    for i in ["top","bottom","right","left"]:
        ax2.spines[i].set_visible(False)
        
    p.close()
    
    p.update(1)

    with PdfPages(f'{resultpath}/../../report_{resultpath.split('/')[-1]}_new.pdf') as pdf:
        pdf.savefig(fig2,
                     bbox_inches="tight",dpi=600,pad_inches=0.1)
        plt.close()

        # We can also set the file's metadata via the PdfPages object:
        d = pdf.infodict()
        d['Title'] = f'{resultpath.split("/")[1]}'
        d['Author'] = 'ESNZ 2026 - EnerRad'
        d['Keywords'] = 'Radiated seismi energy'
        d['CreationDate'] = datetime.datetime.today()