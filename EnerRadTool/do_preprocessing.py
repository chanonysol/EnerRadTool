# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 17:19:55 2024

@author: solen
"""
import os
import shutil
import sys
from glob import glob
from colorama import Fore, Style, Back
from numpy import arange

from data_preprocessing import preprocessing, read_inventory, read,\
    Inventory, Stream, request_val_p_train, clean_data
from data_utils import rem_line_log
from results_writing import update_results
from do_estimation import folder_name


def do_preprocessing(ev, path, param, current_dist, next_dist, results, rt=False, snrlim = 10):   
    if snrlim !=5. :
        sys.stdout.write(f"\n{Back.MAGENTA + Style.DIM + Fore.BLACK}"
                         +f"{'Preparation  '.rjust(20):20s}"
                         +f"{Style.RESET_ALL}")
        
    if isinstance(param.general.search_data,bool) :
        datapath,invpath = f"{path}/Data/Records",f"{path}/Data/Stations"
    else :
        datapath = param.general.search_data
        invpath = param.general.search_data+"stations"
    if param.waves.network_fixed:
        datapath,invpath = f"{path}/Data/FixedRecords",f"{path}/Data/FixedStations"
    if param.distance.include_near and current_dist==0.:
        datapath,invpath = f"{path}/Data/CloseRecords",f"{path}/Data/CloseStations"
    
    #No data kept
    if os.listdir(f"{datapath}/")==["run_time.txt"]:
        sys.stdout.write(f" >> {Back.MAGENTA + Style.DIM + Fore.BLACK}"
                         +f"{'No data to use - skip  '.rjust(20):20s}"
                         +f"{Style.RESET_ALL}")
        raise(FileNotFoundError("Not enough data to process"))
    
    
    folder = folder_name(param,rt,close = True if current_dist == 0 else False)
    resultpath = f"{path}/Results/{folder}"
    if glob(f"{datapath}/*d*.mseed") == [] and os.listdir(datapath)!=[]:
        clean_data(datapath,invpath,ev,param.general.keep_data)
    
    if snrlim !=5. :
        sys.stdout.write(f"{Fore.MAGENTA + Style.DIM}"
                          +f"> {folder}"
                          +f"{Style.RESET_ALL}")
    
    #result file and destination
    #check all is in result foder
        
    #Creating folder to avoid errors
    if ("Results" not in os.listdir(path)):
        os.mkdir(f"{path}/Results")
        
        
    if (os.path.exists(resultpath) and os.listdir(resultpath)!=[] 
        and param.redoing_results.force_results and (rt in [-1,0] or not rt)):
        # if rt == 0:
        #     # remove only results file
        # else :
        #     # remove lines in file
        shutil.rmtree(f"{resultpath}")
        sys.stdout.write(f"{Fore.MAGENTA + Style.DIM}"
                         +f"\n > Results folder has been reset"
                         +f"{Style.RESET_ALL}\n")
        
    if not os.path.exists(resultpath):
        os.mkdir(resultpath)
        sys.stdout.write(f"{Fore.MAGENTA + Style.DIM}"
                          +f"\n > Results folder has been created"
                          +f"{Style.RESET_ALL}") 
        
    if len(glob(f"{resultpath.replace(']','?').replace('[','?')}/log_keptdata_*"))==0:
        if not os.path.exists(f"{datapath}/../log_keptdata_{datapath.split('/')[-1]}.dat"):
            from data_utils import add_attribute_xml,write_log
            from obspy import read,read_inventory
            st = read(f"{datapath}/*d*.mseed")
            inv = read_inventory(f"{invpath}/*.xml")
            inv = add_attribute_xml(inv,None,st,
                                datapath,invpath,returninv=True)
            write_log(f"{datapath}/../",f"log_keptdata_{datapath.split('/')[-1]}", 
                      st, inv)
        shutil.copy(f"{datapath}/../log_keptdata_{datapath.split('/')[-1]}.dat",
                    f"{resultpath}/log_keptdata_{datapath.split('/')[-1]}_BB.dat")
        shutil.copy(f"{datapath}/../log_keptdata_{datapath.split('/')[-1]}.dat",
                    f"{resultpath}/log_keptdata_{datapath.split('/')[-1]}_HF.dat")
        if current_dist != 0. :
            rem_line_log(resultpath,f"log_keptdata_{datapath.split('/')[-1]}_BB",
                         arange(param.distance.min,param.distance.max,.1))
            rem_line_log(resultpath,f"log_keptdata_{datapath.split('/')[-1]}_HF",
                         arange(param.distance.min,param.distance.max,.1))
        
        
    sys.stdout.write(f"\n{Back.MAGENTA + Style.DIM + Fore.BLACK}"
                     +f"{'Preprocessing  '.rjust(20):20s}"
                     +f"{Style.RESET_ALL}")
    print(current_dist,next_dist)
    return preprocessing(datapath,invpath,resultpath,ev,param.time_max,param.time_max_s,
                  param.channel_code,current_dist,next_dist,
                  param.deviation_correction.type,snrlim)

    #real time and data doesn't exist
    if not isinstance(rt,int):
        sys.stdout.write(f"{Fore.CYAN}single event ")
        sys.stdout.write(f"{Back.RED + Fore.BLACK}>> IN DEVELOPMENT "
                         +f"{Style.RESET_ALL + Fore.RED} CODE NOT DONE YET"
                         f"{Style.RESET_ALL}\n")
        return
