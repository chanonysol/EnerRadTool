# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 13:27:41 2025

@author: solen
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 17:19:55 2024

@author: solen
"""
import os
import shutil
import sys
from glob import glob
from time import time
from colorama import Fore, Style, Back
from numpy import arange, array, loadtxt, argwhere, sum, shape, reshape, \
    unique, concatenate,nan

from data_utils import init_data_folder, UTCDateTime
from data_preprocessing import preprocessing, read_inventory, read,\
    Inventory, Stream, request_val_p_train, request_val_s_train, clean_data
from results_writing import update_results

def do_query(ev, path, param, current_dist, next_dist, results, rt=False):
    sys.stdout.write(f"\n{Back.YELLOW + Style.DIM + Fore.BLACK}"
                     +f"{'Data collection  '.rjust(20):20s}"
                     +f"{Style.RESET_ALL}")

    # Creating folder to avoid errors
    if ("Data" not in os.listdir(path)):
        init_data_folder(path)
        sys.stdout.write(f"{Fore.YELLOW + Style.DIM}"
                         +f"\n > Data folder has been created"
                         +f"{Style.RESET_ALL}\n")
    
    if isinstance(param.general.search_data,bool) :
        datapath,invpath = f"{path}/Data/Records",f"{path}/Data/Stations"
    if param.waves.network_fixed:
        datapath,invpath = f"{path}/Data/FixedRecords",f"{path}/Data/FixedStations"
    if param.distance.include_near and current_dist==0.:
        datapath,invpath = f"{path}/Data/CloseRecords",f"{path}/Data/CloseStations"
        
    #Reset folder because of parameters forcing query
    if (not os.path.exists(datapath) or 
        (os.path.exists(datapath)  
        and param.redoing_results.force_signal
            and (rt in [-1,0] or not rt))):
        if os.path.exists(datapath): shutil.rmtree(f"{datapath}")
        if os.path.exists(invpath): shutil.rmtree(f"{invpath}")
        os.mkdir(datapath);os.mkdir(invpath)
        for f in glob(f"{path}/Data/log_*{datapath.split('/')[-1]}.dat"):
            os.remove(f)
        sys.stdout.write(f"{Fore.YELLOW + Style.DIM}"
                          +f"\n > Data folder has been reset"
                          +f"{Style.RESET_ALL}\n")
    
    if len(glob(f"{datapath}/*[ENZ].mseed"))!=0:
        from data_preprocessing import clean_data
        STARTTIME = time()
        clean_data(datapath,invpath,ev,param.general.keep_data)
        if results : 
            update_results(results, pre_time = time()-STARTTIME)
            update_results(results, data_time = nan)
        
        mat = unique(array([i.split(".")[-2][1:] 
                            for i in glob(f"{datapath}/*d*.mseed")]).astype(float))
        
        times= array(list(results['information']['time'].values()))
        with open(f"{datapath}/run_time.txt","a") as file:
            file.write(f"{current_dist:2.0f} {next_dist:2.0f} "
                       +f"{times[0]:6.2f} {times[1]:6.2f}\n")
    
    alld = array([])
    if os.path.exists(f"{datapath}/run_time.txt"):
        lines = loadtxt(f"{datapath}/run_time.txt")
        if shape(lines)==(4,):
            lines = reshape(lines,(1,4))
        # else:
        #     lines=lines[argsort(lines[:,0])]
        ind = argwhere(lines[:,1]<=next_dist)
        alld = unique(lines[:,:2])
        
    alld = unique(concatenate(([int(p.split(".")[-2][1:]) 
                   for p in glob(f"{datapath}/*d*.mseed")],alld)))
    seld = array([i for i in alld if (i>=current_dist and i<=next_dist)])
    
    d=0
    
    if len(seld)==0 or (seld-current_dist).min()>7 or (next_dist-seld).min()>7:
        d = 1 
    
    if d==0: #check integrity data folder
        if len(glob(f"{invpath}/*.xml"))>len(glob(f"{datapath}/*.MSEED")):
            sys.stdout.write(f"{Fore.YELLOW + Style.DIM}"
                              +f" > Detected too many xml files - cleaning"
                              +f"{Style.RESET_ALL}")
            from data_preprocessing import clean_data, clean_close_data
            if param.waves.S_near and current_dist==0:
                clean_close_data(datapath,invpath,ev,param.general.keep_data)
            else:
                clean_data(datapath,invpath,ev,param.general.keep_data,ext = "*.MSEED")
        if (len(glob(f"{invpath}/*.xml"))<len(glob(f"{datapath}/*.MSEED"))
            or len(glob(f"{invpath}/*.xml"))!=len(glob(f"{invpath}/*.json"))):
                sys.stdout.write(f"{Fore.YELLOW + Style.DIM}"
                                  +f" > Deteted problem in inventory - restart"
                                  +f"{Style.RESET_ALL}")
                if os.path.exists(datapath): shutil.rmtree(f"{datapath}")
                if os.path.exists(invpath): shutil.rmtree(f"{invpath}")
                os.mkdir(datapath);os.mkdir(invpath)
                for f in glob(f"{path}/Data/log_{datapath.split('/')[-1]}.dat"):
                    os.remove(f)
                sys.stdout.write(f"{Fore.YELLOW + Style.DIM}"
                                  +f"\n > Data folder has been reset"
                                  +f"{Style.RESET_ALL}\n")
                d = 1
    
    if param.general.search_data==True and d!=0:
        STARTTIME = time()

        nbBH = len(glob(f"{datapath}/*BH[ENZ].mseed"))
        nbHH = len(glob(f"{datapath}/*HH[ENZ].mseed"))
        nbSH = len(glob(f"{datapath}/*SH[ENZ].mseed"))
        nsta = len(glob(f"{invpath}/*.xml"))

        times = request_val_p_train(['P','time'],ev.depth,array([5.,80.]),
                                    model_name="iasp91")[0]
        
        if len(seld)!=0:
            sys.stdout.write(f"{Fore.YELLOW} > Done from {seld.min()}° to {seld.max()}° ")
            if (seld-current_dist).min()>7:
                next_dist = seld.min()
            if (next_dist-seld).min()>7:
                current_dist = seld.max()
        
        mintime = request_val_p_train(['P','time'],ev.depth,array([next_dist]),
                                    model_name="iasp91")[0][0][0]
        
        #Waiting time if rt
        if rt == True :
            from data_query import waiting_P_smileys
            waiting_P_smileys(ev, mintime, next_dist)
        #Entering query
        sys.stdout.write(f"{Fore.YELLOW + Style.DIM}"
                          +f" > Starting FDSN query"
                          +f"{Style.RESET_ALL}")
        
        sys.stdout.write(f"{Fore.YELLOW} > Looking from {current_dist:2.0f}°"+ 
                         f" until {next_dist-1:2.0f}° distance\n{"":22s} ")
        
        if param.waves.S_near and current_dist==0:
            from data_query import regional_query
            from obspy.geodetics.base import degrees2kilometers,kilometer2degrees
            times = request_val_s_train(['S','time'],ev.depth,
                                        array([kilometer2degrees(200)]),
                                        model_name="iasp91")[0]
            regional_query(ev,path,0,times[0][-1] + 460,
                           [f"{code}N,{code}E" for code in param.channel_code])
        
        elif param.waves.network_fixed :
            from data_query import check4fixed_stations
            test = check4fixed_stations(path,current_dist,next_dist)
            if not test:
                from data_query import fixed_net_query
                fixed_net_query(ev,path,current_dist,next_dist-1,0,times[0][-1] + 460,
                               [f"{code}Z" for code in param.channel_code])
            else :
                sys.stdout.write(f"{Fore.RED + Style.BRIGHT}"
                                  +f"Copied from previous analysis"
                                  +f"{Style.RESET_ALL}\n{"":20s}")
                if results : update_results(results,data_time = time()-STARTTIME)
                if results : update_results(results,pre_time = 0)
                times= array(list(results['information']['time'].values()))
                with open(f"{datapath}/run_time.txt","a") as file:
                    file.write(f"{current_dist:2.0f} {next_dist:2.0f} "
                               +f"{times[0]:6.2f} {times[1]:6.2f}\n")
                return
        else :
            from data_query import selected_query
            selected_query(ev,path,current_dist,next_dist-1,0,times[0][-1] + 460,
                       [f"{code}Z" for code in param.channel_code] )        

        nbBH = len(glob(f"{datapath}/*BH[ENZ].mseed"))-nbBH
        nbHH = len(glob(f"{datapath}/*HH[ENZ].mseed"))-nbHH
        nbSH = len(glob(f"{datapath}/*SH[ENZ].mseed"))-nbSH
        nsta = len(glob(f"{invpath}/*.xml"))-nsta
        
        times=array([0,0]); times[0] = time()-STARTTIME
        
        if nsta!=0 and all(array([nbBH,nbHH,nbSH])==0):
            STARTTIME = time()
        else :   
            sys.stdout.write(f"{Fore.RED + Style.BRIGHT}>>> Result query :"
                             +f"{Style.RESET_ALL} ")
        sys.stdout.write(f"{Fore.RED + Style.BRIGHT}"
                          +f"Added {nsta} signals ("
                          +f"{nbBH} BHZ - {nbHH} HHZ - {nbSH} SHZ)"
                          +f"{Style.RESET_ALL}\n{"":20s}")
        if results : update_results(results,data_time = time()-STARTTIME)
        
        from data_preprocessing import clean_data, clean_close_data
        
        STARTTIME = time()
        if nsta!=0 and not all(array([nbBH,nbHH,nbSH])==0):
            sys.stdout.write(f"{Fore.YELLOW + Style.DIM}"
                              +f" > Formating data"
                              +f"{Style.RESET_ALL}\n")
            if param.waves.S_near and current_dist==0:
                clean_close_data(datapath,invpath,ev,param.general.keep_data)
            else:
                clean_data(datapath,invpath,ev,param.general.keep_data)
        if results : update_results(results, pre_time = time()-STARTTIME)
        times[1] = time()-STARTTIME
        #times= array(list(results['information']['time'].values()))
        with open(f"{datapath}/run_time.txt","a") as file:
            file.write(f"{current_dist:2.0f} {next_dist:2.0f} "
                       +f"{times[0]:6.2f} {times[1]:6.2f}\n")
        
    else :
        from numpy import argsort
        if os.path.exists(f"{datapath}/run_time.txt"):
            lines = loadtxt(f"{datapath}/run_time.txt")
            if shape(lines)==(4,):
                lines = reshape(lines,(1,4))
            lines = lines[argsort(lines[:,0])]
            if current_dist in lines[:,0] and next_dist in lines[:,1]:
                ind = argwhere(lines[:,1]<=next_dist)
                t = [sum(lines[ind,2]),sum(lines[ind,3])]
            else : 
                t = array([sum(lines[:,2]),sum(lines[:,3])])//int(param.distance.totalstep)    
            
            update_results(results, data_time = t[0])
            update_results(results, pre_time = t[1])
        else:
            update_results(results, data_time = nan)
            update_results(results, pre_time = nan)
            
        sys.stdout.write(f"{Fore.YELLOW + Style.DIM}"
                          +f" > Already collected - skipping to preprocessing"
                          +f"{Style.RESET_ALL}\n")
            
 
        
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
