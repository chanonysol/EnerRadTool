# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 09:51:52 2024

@author: solen
"""
# =============================================================================
# LIBRARIES IMPORT
# =============================================================================
import argparse
import sys
import os
import subprocess
from numpy import arange
from colorama import Back, Fore, Style

# ---- add module to python path
CODE_PATH = os.path.dirname(os.path.realpath(__file__)).replace('\\', '/')
MODULE_PATH = f"{CODE_PATH}/EnerRadTool"
sys.path.insert(0, MODULE_PATH)
DIRLIST = [f"{MODULE_PATH}/{folder}"
           for folder in os.listdir(f"{MODULE_PATH}")
           if folder[:2] != "do"]
for DIR in DIRLIST:
    sys.path.insert(0, DIR)
# ---- Import module
from generate_param import Parameters
from generate_event import Event, moment
from do_preprocessing import do_preprocessing, Inventory, Stream, request_val_p_train
#from do_data import do_data_preprocessing
from do_estimation import do_estimations
from ener_functions import trace_ener, smooth_curve_mean
from results_writing import results_dictionnary_init, update_results,\
    complete_from_template
from time_rupture_functions import find_tr_tacer

from numpy import linspace, arange
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib import colormaps
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.ticker import MultipleLocator
from data_utils import verif_size

from numpy import vstack, mean, array
from multiprocessing import Pool
from tqdm import tqdm
from copy import deepcopy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help='path to folder of event')
    parser.add_argument("-rt", "--real_time", action="store_true",
                        help='apply the real time scheme')
    parser.add_argument("-n", "--new_param", action="store_true",
                        help='change parameter file')
    parser.add_argument("-sp", "--show_parameters", action="store_true",
                        help='write the parameters and set values for verification')

    sys.stdout.write(f"\n{Back.CYAN + Style.DIM + Fore.BLACK}"
                     +f"{'Initialization'.center(65):65s}"
                     +f"{Style.RESET_ALL}\n")
    args = parser.parse_args()
    args.path = f"{CODE_PATH}/{args.path}"

    parameters = Parameters(args.path, new=args.new_param, show = args.show_parameters)
    sys.stdout.write("-------> Parameters loaded\n")

    sys.stdout.write(f"{Back.YELLOW + Style.DIM + Fore.RED}>> TO DO "
                     +f"{Style.RESET_ALL + Fore.YELLOW} Improve Event Class"
                     f"{Style.RESET_ALL}\n")

    ev = Event(args.path)
    if args.show_parameters:
        ev.arglist()
    sys.stdout.write("-------> Event loaded\n")

    dist_list = arange(parameters.distance_max.max,parameters.distance_min,
                        -parameters.distance_max.step)[::-1]
    if not args.real_time :
        dist_list = dist_list[[0,-1]]
    dist_list[0]=parameters.distance_min

    general_st_bb = Stream()
    general_st_hf = Stream()

    parameters.time_max=request_val_p_train(["duration"],ev.depth,
                                             dist_list,
                                             model_name="iasp91")[0][0][1]
    results = results_dictionnary_init(parameters)
    update_results(results,ev=ev)

    for (iteration,current_dist) in enumerate(dist_list[:-1]):
        sys.stdout.write(f"\n{Back.CYAN + Style.DIM + Fore.BLACK}"
                          +f"{f'n°{iteration+1} iteration'.center(65):65s}"
                          +f"{Style.RESET_ALL}\n")

        new_st = do_preprocessing(ev, args.path, parameters, current_dist,
                                  dist_list, rt = iteration if args.real_time else args.real_time)

        general_st_bb += new_st[0]
        general_st_hf += new_st[1]

        iter_st_bb = general_st_bb.copy()
        iter_st_hf = general_st_hf.copy()

        iter_st_bb = verif_size(iter_st_bb, "BB")
        iter_st_hf = verif_size(iter_st_hf, "HF")

        iter_st_bb.sort(['distance'])
        iter_st_hf.sort(['distance'])

        #update_results(results,data_time=50, pre_time=37)

        update_results(results,estimate = [iteration, dist_list[:iteration+2],
                                           (iter_st_bb,iter_st_hf),
                                           parameters.time_max])

        do_estimations(ev, args.path, parameters, iter_st_bb, iter_st_hf,
                        results, rt=iteration)

        #complete_from_template(results)


