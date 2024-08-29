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


from numpy import array, all, any, mean, log10
from generate_event import moment

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
               'information':{}}

    if param.deviation_correction.type =="result":
        results.update({'corrected':{k:None for k in ['Ebb','Ehf','MeBB','MeHF','Tr',
                                                          'Tr_tacer','Tr_lines','theta',
                                                          'E_HF/Tr3']}})
        results.update({'corrected_smooth':{k:None for k in ['Ebb','Ehf','MeBB','MeHF','Tr',
                                                          'Tr_tacer','Tr_lines','theta',
                                                          'E_HF/Tr3']}})
    return results

def add_results(results, ev, **kwargs):
    M0 = moment(ev.moment,ev.exponent)
    for key in list(kwargs.keys()):
        Ebb, Ehf, tr_tacer, tr_lign = kwargs[key]

        Tr = mean([tr_tacer if tr_tacer else tr_lign,
                   tr_lign if tr_lign else tr_tacer])

        MeBB = Me(Ebb)
        MeHF = Me(Ehf/0.13)
        EhfTr = Ehf/Tr**3
        theta = log10(Ebb/M0)

        newdict = {'Ebb': Ebb, 'MeBB': MeBB, 'Ehf': Ehf, 'MeHF': MeHF,
                   'Tr': Tr,'theta': theta,
                   'Tr_tacer': tr_tacer if tr_tacer else -9999,
                    'Tr_lines': tr_lign if tr_lign else -9999,
                    'E_HF/Tr3': EhfTr}
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
        if "event" not in results["information"].keys():
            ev_info={"date":ev.date ,"mag":ev.mag , "magtype":"Mw",
                     "M0":moment(ev.moment,ev.exponent),
                     "lat":ev.lat, "lon":ev.lon, "depth":ev.depth,
                     "region":None,"subregion":None}
        else :
            ev_info = results['information']["event"]
            for (element,value) in ev.items():
                ev_info.update({element:value})
        results['information'].update({"event":ev_info})

    if "estimate" in kwargs.keys():
        rt, dist, (stbb,sthf), time = kwargs["estimate"]
        if "estimate" not in results["information"].keys():
            ener_info={"version":"v1.0.RT" ,"iter": rt if isinstance(rt, int) else None ,
                       "mindist": dist.min(),
                       "maxdist": dist.max(),
                       "nsta_bb":len(list(set([f"{tr.stats.network}.{tr.stats.station}.{tr.stats.location}" for tr in stbb]))),
                       "channel_list_bb":list(set([f"{tr.stats.channel}" for tr in stbb])),
                       "nsta_hf":len(list(set([f"{tr.stats.network}.{tr.stats.station}.{tr.stats.location}" for tr in sthf]))),
                       "channel_list_hf":list(set([f"{tr.stats.channel}" for tr in sthf])),
                       "duration":time,"ref_ratio":13}
        elif rt!=results['information']['estimate']['iter']:
            ener_info = results['information']["estimate"]
            ener_info.update({"iter": rt, "maxdist": dist.max(),
                       "nsta_bb":len(list(set([f"{tr.stats.network}.{tr.stats.station}.{tr.stats.location}" for tr in stbb]))),
                       "channel_list_bb":list(set([f"{tr.stats.channel}" for tr in stbb])),
                       "nsta_hf":len(list(set([f"{tr.stats.network}.{tr.stats.station}.{tr.stats.location}" for tr in sthf]))),
                       "channel_list_hf":list(set([f"{tr.stats.channel}" for tr in sthf]))})
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
                time_info.update({element: kwargs[element]})
        results['information'].update({"time": time_info})

def complete_from_template(results, path = None):
    if not path :
        file = sys.stdout
    else :
        file = open(f"{path}/results.txt","w")

    file.write(f"#======================================\n")
    file.write(f"#		         RESULTS\n")
    file.write(f"#======================================\n")
    file.write(f"\n")

    Ebb, MeBB, Ehf, MeHF, Tr, theta, Tr_tacer, Tr_lines, EhfTr3 =\
        list(results['raw'].values())
    file.write(f"#----------- Raw estimation -----------\n")
    file.write(f"Ebb = {Ebb:7.2e}J ; MeBB = {MeBB:3.1f}\n")
    file.write(f"Ehf = {Ehf:7.2e}J ; MeHF = {MeHF:3.1f}\n")
    file.write(f"\n")
    file.write(f"Tr = {Tr:3.0f}s; theta = {theta:3.1f}\n")
    file.write(f"(Tr_tacer = {Tr_tacer:3.0f}s)\n")
    file.write(f"(Tr_lines = {Tr_lines:3.0f}s)\n")
    file.write(f"\n")
    file.write(f"E_HF/Tr3 = {EhfTr3:7.2e}J\n")
    file.write(f"\n")

    if all(array(list(results['raw_smooth'].values()))):
        Ebb, MeBB, Ehf, MeHF, Tr, theta, Tr_tacer, Tr_lines, EhfTr3 =\
            list(results['raw_smooth'].values())
        file.write(f"#smoothed\n")
        file.write(f"Ebb = {Ebb:7.2e}J ; MeBB = {MeBB:3.1f}\n")
        file.write(f"Ehf = {Ehf:7.2e}J ; MeHF = {MeHF:3.1f}\n")
        file.write(f"\n")
        file.write(f"Tr = {Tr:3.0f}s; theta = {theta:3.1f}\n")
        file.write(f"(Tr_tacer = {Tr_tacer:3.0f}s)\n")
        file.write(f"(Tr_lines = {Tr_lines:3.0f}s)\n")
        file.write(f"\n")
        file.write(f"E_HF/Tr3 = {Ehf:7.2e}J\n")
        file.write(f"\n")

    if all(array(list(results['corrected'].values()))):
        Ebb, MeBB, Ehf, MeHF, Tr, theta, Tr_tacer, Tr_lines, EhfTr3 =\
            list(results['corrected'].values())
        file.write(f"#-------- Corrected estimation --------\n")
        file.write(f"Ebb = {Ebb:7.2e}J ; MeBB = {MeBB:3.1f}\n")
        file.write(f"Ehf = {Ehf:7.2e}J ; MeHF = {MeHF:3.1f}\n")
        file.write(f"\n")
        file.write(f"Tr = {Tr:3.0f}s; theta = {theta:3.1f}\n")
        file.write(f"(Tr_tacer = {Tr_tacer:3.0f}s)\n")
        file.write(f"(Tr_lines = {Tr_lines:3.0f}s)\n")
        file.write(f"\n")
        file.write(f"E_HF/Tr3 = {Ehf:7.2e}J\n")
        file.write(f"\n")

    if all(array(list(results['corrected_smooth'].values()))):
        Ebb, MeBB, Ehf, MeHF, Tr, theta, Tr_tacer, Tr_lines, EhfTr3 =\
            list(results['corrected_smooth'].values())
        file.write(f"#smoothed\n")
        file.write(f"Ebb = {Ebb:7.2e}J ; MeBB = {MeBB:3.1f}\n")
        file.write(f"Ehf = {Ehf:7.2e}J ; MeHF = {MeHF:3.1f}\n")
        file.write(f"\n")
        file.write(f"Tr = {Tr:3.0f}s; theta = {theta:3.1f}\n")
        file.write(f"(Tr_tacer = {Tr_tacer:3.0f}s)\n")
        file.write(f"(Tr_lines = {Tr_lines:3.0f}s)\n")
        file.write(f"\n")
        file.write(f"E_HF/Tr3 = {Ehf:7.2e}J\n")
        file.write(f"\n")

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
    file.write(f"# Region : {region}\n")
    file.write(f"# Subregion : {subregion}\n")
    file.write(f"\n")

    version, iter, mindist, maxdist, nsta_bb, channel_list_bb, nsta_hf, \
        channel_list_hf, duration, ref_ratio=\
        list(results['information']['estimate'].values())
    file.write(f"#------- Estimation information -------\n")
    file.write(f"# {version}\n")
    file.write(f"# iteration nÂ°{iter} : {mindist:2.0f}-{maxdist:2.0f}\n")
    file.write(f"# BroadBand : {nsta_bb:2.0f} stations [{channel_list_bb}]\n")
    file.write(f"# HighFrequencies : {nsta_hf:2.0f} stations [{channel_list_hf}]\n")
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
