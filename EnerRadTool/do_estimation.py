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

from numpy import array, vstack, argwhere,\
    sin, cos, pi, arcsin, linspace, arange, mean
from obspy import UTCDateTime
from ener_functions import trace_ener, smooth_curve_mean, calc_ener_rt
from results_writing import add_results, update_results, complete_from_template

from glob import glob
from time import time
from tqdm import tqdm
from multiprocessing import Pool
from colorama import Back, Fore, Style


def do_estimations(ev, path, param, st_bb, st_hf, results, rt=False):
    devlim = param.deviation_correction.threshold
    correction = True if param.deviation_correction.type=="result" else False

    name = param.general.session_name
    day = UTCDateTime(time())
    STARTTIME = time()
    #selecting the right results folder
    if isinstance(rt,int):
        folder = f"RT{day.year}{day.month}{day.day}_{name}_iter{rt}"
    else :
        folder = f"RT{day.year}{day.month}{day.day}_{name}"
    resultpath = f"{path}/Results/{folder}"

    sys.stdout.write(f"\n{Back.RED + Style.DIM + Fore.BLACK}"
                     +f"{'Energy estimation '.rjust(20):20s}"
                     +f"{Style.RESET_ALL}")
    #Creating folder to avoid errors
    if ("Results" not in os.listdir(path)):
        os.mkdir(f"{path}/Results")
    if folder not in os.listdir(f"{path}/Results"):
        os.mkdir(resultpath)

        sys.stdout.write(f"{Fore.RED + Style.DIM}"
                         +f"\n > Results folder has been created"
                         +f"{Style.RESET_ALL}\n")

    #Reset folder because of parameters forcing estimation
    if os.listdir(resultpath)!=[] and param.redoing_results.force_results:
        shutil.rmtree(resultpath)
        os.mkdir(resultpath)
        sys.stdout.write(f"{Fore.RED + Style.DIM}"
                         +f"\n > Results folder has been reset"
                         +f"{Style.RESET_ALL}\n")

    if os.listdir(resultpath)!=[] :
        sys.stdout.write(f"{Fore.RED + Style.DIM}"
                          +f" > Already estimated - skipping to figures"
                          +f"{Style.RESET_ALL}\n")

    if os.listdir(resultpath)==[]:
        if isinstance(rt,int):
            ener_bb_sta, raw_results, ener_hf_sta, smooth_results =\
                calc_ener_rt(st_bb,st_hf,param)
            (ener_bb,ener_hf,_,((tr_tacer,_),(tr_lign,_,_))) =  raw_results
            (ener_bb_s,ener_hf_s,_,((tr_tacer_s,_),(tr_lign_s,_,_))) = smooth_results

            Tr = int(mean([tr_tacer if tr_tacer else tr_lign,
                       tr_lign if tr_lign else tr_tacer]))
            Tr_s = int(mean([tr_tacer_s if tr_tacer_s else tr_lign_s,
                       tr_lign_s if tr_lign_s else tr_tacer_s]))

            add_results(results, ev, raw = [ener_bb[Tr], ener_hf[Tr],
                                            tr_tacer, tr_lign] )
            add_results(results, ev, raw_smooth = [ener_bb[Tr_s], ener_hf[Tr_s],
                                                   tr_tacer_s, tr_lign_s] )
            update_results(results,ener_time=time()-STARTTIME)
            if correction:
                sys.stdout.write(f"{Fore.RED + Style.DIM}"
                                  +f" > Correcting for deviation"
                                  +f"{Style.RESET_ALL}\n")
                sys.stdout.write(f"{Back.RED + Fore.BLACK}>> IN DEVELOPMENT "
                                 +f"{Style.RESET_ALL + Fore.RED} CODE NOT DONE YET"
                                 f"{Style.RESET_ALL}\n")
                #correct(devlim)
        else:
            sys.stdout.write(f"{Back.RED + Fore.BLACK}>> IN DEVELOPMENT "
                             +f"{Style.RESET_ALL + Fore.RED} CODE NOT DONE YET"
                             f"{Style.RESET_ALL}\n")



        sys.stdout.write(f"{Fore.RED + Style.DIM}"
                          +f" > Saving results"
                          +f"{Style.RESET_ALL}")
        complete_from_template(results, path = resultpath)
