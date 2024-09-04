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

from data_query import *
from data_preprocessing import preprocessing, read_inventory, read,\
    Inventory, Stream, request_val_p_train

def do_preprocessing(ev, path, param, current_dist, dist_list, rt=False):
    # getting parameters needed
    channels = param.channel_code
    time_max = param.time_max

    sys.stdout.write(f"\n{Back.YELLOW + Style.DIM + Fore.BLACK}"
                     +f"{'Data collection  '.rjust(20):20s}"
                     +f"{Style.RESET_ALL}")
    #Creating folder to avoid errors
    if ("Data" not in os.listdir(path)):
        os.mkdir(f"{path}/Data")
        sys.stdout.write(f"{Fore.YELLOW + Style.DIM}"
                         +f"\n > Data folder has been created"
                         +f"{Style.RESET_ALL}\n")

    #Reset folder because of parameters forcing query
    if ("Data" in os.listdir(path)) and param.redoing_results.force_signal:
        shutil.rmtree(f"{path}/Data")
        os.mkdir(f"{path}/Data")
        sys.stdout.write(f"{Fore.YELLOW + Style.DIM}"
                         +f"\n > Data folder has been reset"
                         +f"{Style.RESET_ALL}\n")

    if ("Data" in os.listdir(path)):
        #real time and data already exist
        if isinstance(rt,int) and len(glob(f"{path}/Data/iter{rt}/*.mseed"))!=0:
            #sys.stdout.write(f"{Fore.CYAN}------> real time processing\n")
            os.mkdir(f"{path}/Data/iter{rt}")
            os.mkdir(f"{path}/Data/iter{rt}/stations")
            sys.stdout.write(f"{Fore.YELLOW + Style.DIM}"
                              +f" > Already collected - skipping to preprocessing"
                              +f"{Style.RESET_ALL}\n")
            sys.stdout.write(f"\n{Back.MAGENTA + Style.DIM + Fore.BLACK}"
                             +f"{'Preprocessing  '.rjust(20):20s}"
                             +f"{Style.RESET_ALL}")
            return preprocessing(f"{path}/Data/iter{rt}",ev,channels,dist_list)

        #real time and data doesn't exist
        if not isinstance(rt,int):
            sys.stdout.write(f"{Fore.CYAN}single event ")
            sys.stdout.write(f"{Back.RED + Fore.BLACK}>> IN DEVELOPMENT "
                             +f"{Style.RESET_ALL + Fore.RED} CODE NOT DONE YET"
                             f"{Style.RESET_ALL}\n")
            return
        if isinstance(rt,int) and len(glob(f"{path}/Data/iter{rt}/*.mseed"))==0:
            if (f"iter{rt}" not in os.listdir(f"{path}/Data/")):
               os.mkdir(f"{path}/Data/iter{rt}")
               os.mkdir(f"{path}/Data/iter{rt}/stations")
            #sys.stdout.write(f"{Fore.CYAN}------> real time processing\n")
            #Entering query
            sys.stdout.write(f"{Fore.YELLOW + Style.DIM}"
                              +f" > Starting FDSN query"
                              +f"{Style.RESET_ALL}\n")
            real_time_query(ev,path,param.channel_code,current_dist,
                            dist_list,rt)
            sys.stdout.write(f"\n{Back.MAGENTA + Style.DIM + Fore.BLACK}"
                             +f"{'Preprocessing  '.rjust(20):20s}"
                             +f"{Style.RESET_ALL}")
            return preprocessing(f"{path}/Data/iter{rt}",ev,channels,dist_list)
        #not real time
            # Folder not empty and distance already done - skip
            # if len(glob(f"{path}/Data/*.mseed"))!=0:
            #     inv = read_inventory(f"{path}/Data/stations/*.xml")
            #     infodict=precalc(inv, ev)
            #     if max(infodict[0].round(0))>=current_dist:
                    # sys.stdout.write(f"{Fore.CYAN + Style.BRIGHT}"
                    #                   +f"Data already collected - skipping to processing"
                    #                   +f"{Style.RESET_ALL}\n")
            #         return