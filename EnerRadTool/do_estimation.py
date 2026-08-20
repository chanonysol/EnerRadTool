  # -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 20:47:45 2024

@author: solen
"""

import os
import random
import colorama
import sys
import shutil

from numpy import array, vstack, argwhere, shape, nanmean, reshape,\
    sin, cos, pi, arcsin, linspace, arange, mean,where,delete,savetxt,\
        concatenate, log10, nanmedian, nansum
from obspy import UTCDateTime
from ener_functions import trace_ener, smooth_curve_mean, calc_ener_rt, calc_ener_S, weight_contrib
from results_writing import add_results, update_results, complete_from_template,\
    read_results,read_results_sta, write_station_log, write_raw_results_sta, add_results_S,\
    Me
from do_correction import do_corrections
from data_utils import modify_log, find_train, Fest, read_log, quality_level
from time_rupture_functions import joint_tr

from glob import glob
from time import time
from tqdm import tqdm
from multiprocessing import Pool
from colorama import Back, Fore, Style

def folder_name(param,rt,close=False):
    if not param.general.session_name :
        name = f'W{"p" if param.waves.P_teleseismic else ""}'\
            +f'{"r" if param.waves.P_regional else ""}'\
            +f'{"s" if param.waves.S_near else ""}'\
            +f'{"f" if param.waves.network_fixed else ""}_'\
            +f'D[{param.distance.min:.0f}.{param.distance.max:.0f}.{param.distance.totalstep:.0f}]_'\
            +f'T{param.time_max:.0f}'
        if close:
            name = f'W{"s" if param.distance.include_near else ""}_'\
                +f'D[{0.:.0f}.{2.:.0f}.{param.distance.totalstep:.0f}]_'\
                +f'T{param.time_max_s:.0f}'
    return f'{"RT" if isinstance(rt,int) else "SC"}_{name if not param.general.session_name else param.general.session_name}'

def do_estimations(ev, path, param, st_bb, st_hf, inv, results, stadist, rt=False):
    STARTTIME = time()
    #selecting the right results folder
    folder = folder_name(param,rt,close = True if all(stadist<2.) else False)
    resultpath = f"{path}/Results/{folder}"
    if isinstance(param.general.search_data,bool) :
        datapath,invpath = f"{path}/Data/Records",f"{path}/Data/Stations"
    if param.waves.network_fixed:
        datapath,invpath = f"{path}/Data/FixedRecords",f"{path}/Data/FixedStations"
    if param.distance.include_near and all(stadist<=2.):
        datapath,invpath = f"{path}/Data/CloseRecords",f"{path}/Data/CloseStations"
        

    sys.stdout.write(f"{Back.RED + Style.DIM + Fore.BLACK}"
                     +f"\n{'Energy estimation '.rjust(20):20s}"
                     +f"{Style.RESET_ALL}")
        
    if (" ".join(os.listdir(resultpath)).find(f"results_{folder[:2]}{rt}.txt")!=-1
        and not param.redoing_results.force_results):
        sys.stdout.write(f"{Fore.RED + Style.DIM}"
                          +f" > Already estimated - skipping to figures"
                          +f"{Style.RESET_ALL}\n")
        # results = read_results(f"{resultpath}/results_{folder[:2]}{rt}.txt")
        return resultpath, [inv, st_bb, st_hf]       
        
    else :
        if isinstance(rt,int) and not (param.distance.include_near and all(stadist<=2.)):
            ener_sta=None
            ener_sta_names = [],[]
            file_bb = f"{resultpath}/ener_all_bb.dat"
            file_hf = f"{resultpath}/ener_all_hf.dat"
            if rt!=0 or (rt>1 and all(stadist>=5.)): #load results from previous iterations
                ener_sta = (read_results_sta(file_bb)[1],
                            read_results_sta(file_hf)[1])
                ener_sta_names = (read_results_sta(file_bb)[2],
                                  read_results_sta(file_hf)[2])
                
                # dist_bb = []; dist_hf = []
                # for i in arange(0,rt,1):
                #     log = read_sta_log(resultpath.replace(f"iter{rt}",f"iter{i}"),hf=False)[1])
                #     dist_bb = concatenate(dist_bb,[sta_log[k]["dist"] for k in sta_log.keys()])
                #     log = read_sta_log(resultpath.replace(f"iter{rt}",f"iter{i}"),hf=True)[1])
                #     dist_hf = concatenate(dist_hf,[sta_log[k]["dist"] for k in sta_log.keys()])
            #estimates
            ener_bb_sta, raw_results, ener_hf_sta, smooth_results =\
                calc_ener_rt(st_bb,st_hf,inv,param, ev, resultpath,
                              ener_sta = ener_sta if ener_sta else None )
            #separate results
            (ener_bb,ener_hf,_,((tr_tacer,_),(tr_lign,_,_))) =  raw_results
            (ener_bb_s,ener_hf_s,_,((tr_tacer_s,_),(tr_lign_s,_,_))) = smooth_results
            # Rupture duration
            Tr = joint_tr(tr_tacer,tr_lign)
            Tr_s = joint_tr(tr_tacer_s,tr_lign_s)
            
                
            #correction of Fest based on Tr
            if ev.depth > 70. :
                t = arange(0,len(ener_bb),1)
                dt=abs(array([tr_tacer_s,tr_lign_s])-Tr_s).max()
                # train = find_train(ev,stadist,t) 
                train = concatenate(find_train(ev,stadist,[Tr],dt))
                if (train!="sP").any():
                    # Fest_new = Fest(vstack([stadist]*len(t)).T,
                    #                 phase_list = train)
                    # Fest_old = Fest(vstack([stadist]*len(t)).T)
                    Fest_new = Fest(stadist,phase_list = train)
                    Fest_old = Fest(stadist)
                    
                    # ener_bb_sta/=Fest_new/Fest_old
                    # ener_hf_sta/=Fest_new/Fest_old
                    ener_bb_sta/=array([Fest_new/Fest_old]*len(t)).T
                    ener_hf_sta/=array([Fest_new/Fest_old]*len(t)).T
                    
                    ener_bb = nanmean(ener_bb_sta,axis=0)
                    ener_hf = nanmean(ener_hf_sta,axis=0)
                    _, ener_bb_s = smooth_curve_mean(arange(0,len(ener_bb),1), ener_bb)
                    _ ,ener_hf_s = smooth_curve_mean(arange(0,len(ener_bb),1), ener_hf)
            
            # deviation
            from numpy import nan
            t_list = [tr_tacer, tr_lign]
            t_lists = [tr_tacer_s, tr_lign_s]
            
            dev = [abs(ener_bb[Tr] - ener_bb[t]) for t in t_list if t]
            dev= mean(dev/ener_bb[Tr]*100) #percent
            
            dev_s = [abs(ener_bb_s[Tr_s] - ener_bb_s[t]) for t in t_lists if t]
            dev_s= mean(dev_s/ener_bb_s[Tr_s]*100) #percent
            
            dev_hf = [abs(ener_hf[Tr] - ener_hf[t]) for t in t_list if t]
            dev_hf= mean(dev_hf/ener_hf[Tr]*100) #percent
            
            dev_hf_s = [abs(ener_hf_s[Tr_s] - ener_hf_s[t]) for t in t_lists if t]
            dev_hf_s= mean(dev_hf_s/ener_hf_s[Tr_s]*100) #percent
            # results in dict
            add_results(results, ev, raw = [ener_bb[Tr], ener_hf[Tr], dev, dev_hf,
                                            tr_tacer, tr_lign] )
            add_results(results, ev, raw_smooth = [ener_bb_s[Tr_s], ener_hf_s[Tr_s],
                                                   dev_s, dev_hf_s,
                                                   tr_tacer_s, tr_lign_s] )

            if param.general.saved_results in ['all','plot','detailed']:# save results
                file_sta = f"{resultpath}/sta_log.dat"
                st_bb_names,st_hf_names = write_station_log(file_sta,st_bb,st_hf,inv)
                
                modify_log(f"{resultpath}/",f"log_keptdata_{datapath.split('/')[-1]}_BB",st_bb_names,f"REMAIN ITER {rt}")
                modify_log(f"{resultpath}/",f"log_keptdata_{datapath.split('/')[-1]}_HF",st_hf_names,f"REMAIN ITER {rt}")
                                
                st_bb_names = list(ener_sta_names[0]) + st_bb_names
                st_hf_names = list(ener_sta_names[1]) + st_hf_names
                write_raw_results_sta(file_bb,ener_bb_sta[-len(st_bb):],ener_bb,ener_bb_s,st_bb_names[-len(st_bb):],rt=rt)
                write_raw_results_sta(file_hf,ener_hf_sta[-len(st_hf):],ener_hf,ener_hf_s,st_hf_names[-len(st_hf):],rt=rt)

                
            if param.deviation_correction.type == "result" :
            #`and not param.waves.network_fixed: #correct results
                sys.stdout.write(f"{Fore.RED + Style.DIM}"
                                  +f" > Correcting for deviation"
                                  +f"{Style.RESET_ALL}\n")
                st_bb_names = [tr.id for tr in st_bb]
                st_hf_names =  [tr.id for tr in st_hf]
                
                #previous correction incorporated
                #!!!TO DO
                if rt!=0 or (rt>1 and all(stadist>=5.)): 
                    stalog_bb = read_log(f"{resultpath}/",f"log_keptdata_{datapath.split('/')[-1]}_BB",
                                          return_mat=True)
                    keep_names_bb=stalog_bb[:,1][argwhere(array([any(i.find(f"REMAIN ITER {j}")!=-1 for j in range(rt)) 
                                                                  for i in stalog_bb[:,0]]))[:,0]]
                    stalog_hf = read_log(f"{resultpath}/",f"log_keptdata_{datapath.split('/')[-1]}_HF",
                                          return_mat=True)
                    keep_names_hf=stalog_hf[:,1][argwhere(array([any(i.find(f"REMAIN ITER {j}")!=-1 for j in range(rt)) 
                                                                  for i in stalog_hf[:,0]]))[:,0]]
                    
                    Lbb = [i for i,n in enumerate(ener_sta_names[0]) if n in keep_names_bb]
                    Lbb.sort()
                    Lhf = [i for i,n in enumerate(ener_sta_names[1]) if n in keep_names_hf]
                    Lhf.sort()
                    
                    ener_bb_sta = vstack((ener_bb_sta[:-len(st_bb)][Lbb,:],ener_bb_sta[-len(st_bb):]))
                    ener_hf_sta = vstack((ener_hf_sta[:-len(st_hf)][Lhf,:],ener_hf_sta[-len(st_hf):]))
                    
                st_bb,st_hf = do_corrections(ev, resultpath, datapath, param,
                                               ener_bb_sta, 
                                               ener_hf_sta,
                                               st_bb, st_hf,
                                               results, rt=rt)
                # st_bb_names = [n for n in st_bb_names if n not in [tr.id for tr in st_bb]]
                # st_hf_names = [n for n in st_hf_names if n not in [tr.id for tr in st_hf]]
                # modify_log(f"{resultpath}/",f"log_keptdata_{datapath.split('/')[-1]}_BB",st_bb_names,f"correct remov {rt}")
                # modify_log(f"{resultpath}/",f"log_keptdata_{datapath.split('/')[-1]}_HF",st_hf_names,f"correct remov {rt}")
                
            
            
        elif (param.distance.include_near and all(stadist<=2.)):
            
            ener_bb,ener_hf = calc_ener_S(st_bb,st_hf,inv,param,ev)
            
            vs = array([tr.stats.az for tr in st_bb])
            q = 14.6; w=weight_contrib(ev,vs)
            ener_bb = (1+1/q)*ener_bb*w/nansum(w)
            ener_hf = (1+1/q)*ener_hf*w/nansum(w)
            
            E = nansum(ener_bb)
            E_hf = nansum(ener_hf)
            add_results_S(results, ev, S_near = [E,E_hf] )
            
            sys.stdout.write(f"{Fore.RED + Style.DIM}"
                              +f" > Results"
                              +f"{Style.RESET_ALL}\n")
            
            # print(f"{E:.2e} N.m - HF {E_hf:.2e} N.m")
            # print(f"Me{Me(E):.2f} - MeHF {Me(E_hf):.2f}")
            
            # sys.stdout.write(f"{Fore.RED + Style.DIM}"
            #                   +f" > Correction"
            #                   +f"{Style.RESET_ALL}\n")
            
            dev = log10(ener_bb/nanmedian(ener_bb))
            dev_hf = log10(ener_hf/nanmedian(ener_hf))
            
            freq= st_bb[0].stats.freq
            mat = freq
            for i,tr in enumerate(st_bb):
                mat = vstack((mat,(1+1/q)*tr.stats.int*w[i]/nansum(w)))
            mat = vstack((mat,nansum(mat[1:,:],axis=0)))
            from numpy import savetxt
            savetxt(f"{resultpath}/ener_int.dat",mat,fmt="%.2e")
            
            with open(f"{resultpath}/station_ener.dat","w") as file:
                file.write(f"#   station     lon   lat   dist   az  site E\n")
                for tr in st_bb:
                    stainfo = inv.select(station=tr.stats.station)[0][0]
                    
                    dist = stainfo.extra.station_param.distance
                    az =  stainfo.extra.station_param.az
                    lon = stainfo.longitude
                    lat = stainfo.latitude
                    site = list(stainfo.extra.site_class.values())[0]
                    if not site:
                        site = "O"
                    file.write(f"{tr.id:16s} {lon:+6.2f} {lat:+6.2f} "+
                               f"{dist:5.2f} {az:3.0f} {site:1s} {tr.stats.ener:.2e}\n")
                
        
        update_results(results,certain = [rt,quality_level(resultpath,rt,ev,
                                                       correct=True, results = results)])
        
        sys.stdout.write(f"{Fore.RED + Style.DIM}"
                          +" > Saving results"
                          +f"{Style.RESET_ALL}\n")
        
        update_results(results, ener_time = time()-STARTTIME)
        times = array(list(results['information']['time'].values()))[:-1]
        update_results(results, tot_time = sum(where(times<0,0,times)))
        
        complete_from_template(results, path = resultpath,name=f"{folder[:2]}{rt}",
                               S = True if param.distance.include_near else False)
        
        return resultpath, [inv, st_bb, st_hf]
        