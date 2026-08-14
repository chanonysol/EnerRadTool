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
from do_query import do_query
from do_preprocessing import do_preprocessing, Inventory, Stream, request_val_p_train
from do_estimation import do_estimations, folder_name
from ener_functions import trace_ener, smooth_curve_mean, find_ener
from results_writing import results_dictionnary_init, update_results,\
    complete_from_template
from time_rupture_functions import find_tr_tacer
from do_plot import do_plots

from numpy import linspace, arange, concatenate, unique
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib import colormaps
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.ticker import MultipleLocator
from data_utils import verif_size, find_min_dist
from obspy import UTCDateTime

from numpy import vstack, mean, array, argwhere
from multiprocessing import Pool
from tqdm import tqdm
from copy import deepcopy
from time import time
from glob import glob
from traceback import format_exc

def correct_message(message):
    message = message.replace("\\","/")
    for DIR in DIRLIST:
        message = message.replace(DIR+"/","")
    message = message.replace(MODULE_PATH,"").replace(CODE_PATH,"")
    message = message.replace("c:/users/solen/documents/ener/project/","")
    return message+"\n"

def main(path,real_time=True, new_param=False, show_parameters=False, parameters = False,
         reg_lim = None):
    path = f"{CODE_PATH}/{path}"
    
    sys.stdout.write(f"\n{Back.CYAN + Style.DIM + Fore.BLACK}"
                     +f"{('Initialization'+' '*5+str(path.split('/')[-2]).replace('_',' ')).center(65)}"
                     +f"{Style.RESET_ALL}\n")
    
    if not parameters:
        parameters = Parameters(path, new=new_param, show = show_parameters)
        sys.stdout.write("-------> Parameters loaded\n")

    ev = Event(path)
    if reg_lim and ev.sreg!=reg_lim:
        sys.stdout.write(f"{Fore.CYAN + Style.DIM}"
                         +f"\n > Not targeted region. Moving on."
                         +f"{Style.RESET_ALL}\n")
        return
    if show_parameters:
        ev.arglist()
    sys.stdout.write("-------> Event loaded\n")

    if glob(f"{path}/*log*") == []:
        file = open(f"{path}/logfile.txt","x")
        file.close()
      
    if ((parameters.waves.S_near or parameters.distance.include_near) and
            (ev.reg == "SW.Pacific" or not ev.sreg)) :
        sys.stdout.write(f"{Fore.CYAN + Style.DIM}"
                         +f"\n > Region out of scope for Swaves. Moving on."
                         +f"{Style.RESET_ALL}\n")
        return
      
    #adapt dist to depth
    parameters.distance.min = find_min_dist(ev,parameters)    
        
    dist_list = linspace(parameters.distance.min,parameters.distance.max,
                         int(parameters.distance.totalstep+1),endpoint=True).round(0)
    
    if not real_time :
        dist_list = dist_list[[0,-1]]
    dist_list[0]=parameters.distance.min
    
    # if parameters.distance.include_near and 0. not in dist_list:
    #     dist_list = concatenate(([0.],dist_list))
    if dist_list[-1]<= parameters.distance.min:
        dist_list=dist_list[:-1]

    general_st_bb = Stream()
    general_st_hf = Stream()
    general_inv = Inventory()

    if not parameters.time_max:
        parameters.time_max=request_val_p_train(["duration"],ev.depth,
                                                 array([25.]),
                                                 model_name="iasp91")[0][0][1]
    if not parameters.time_max_s:
        parameters.time_max_s=60
    results = results_dictionnary_init(parameters)
    update_results(results,ev=ev)

    for (iteration,(current_dist,next_dist)) in enumerate(
            zip(concatenate(([0] if parameters.distance.include_near else [],dist_list[:-1])),
                concatenate(([2] if parameters.distance.include_near else [],dist_list[1:])))):
        if parameters.distance.include_near : iteration-=1
        sys.stdout.write(f"\n{Back.CYAN + Style.DIM + Fore.BLACK}"
                          +f"{f'n°{iteration} iteration'.center(65):65s}"
                          +f"{Style.RESET_ALL}\n")
        # if iteration==0:
        #     results = results_dictionnary_init(parameters)
        #     update_results(results,ev=ev)
        try :
        
            do_query(ev, path, parameters, current_dist, next_dist,
                     results, rt=iteration)
        except Exception as error :
            with open(f"{path}/Data/0.ERROR.log","a") as file:
                file.write(f"#{str(UTCDateTime().now()):=^60}\n ")
                file.write("Error occured in the data collection phase -")
                file.write(correct_message(format_exc()))
                # exc_type, exc_obj, exc_tb = sys.exc_info()
                # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                # file.write(f"{fname} : line {exc_tb.tb_lineno} \n")
                # file.write(f"{exc_type} : {error}")
                file.write("\n >>> moving on to another iteration\n")
                return format_exc()
                continue
        else : 
            if os.path.exists(f"{path}/Data/0.ERROR.log"):
                os.remove(f"{path}/Data/0.ERROR.log")
        try :
            STARTTIME = time()
            st,st_hf,inv = do_preprocessing(ev, path, parameters, 
                                            current_dist,
                                            dist_list[iteration+1]-1 if current_dist!=0 else 2., 
                                            results,rt = iteration 
                                            if real_time else real_time)
            update_results(results, 
                            pre_time = (results['information']['time']['pre_time'] 
                                        + time()- STARTTIME ))
            general_st_bb += st.copy()
            general_st_hf += st_hf.copy()
            general_inv += inv.copy()
        except FileNotFoundError :
            return
        except Exception as error:
            with open(f"{path}/Results/{folder_name(parameters,iteration)}/0.ERROR.log","a") as file:
                file.write(f"#{str(UTCDateTime().now()):=^60}\n ")
                file.write("Error occured in the preprocessing phase -")
                file.write(correct_message(format_exc()))
               
                # exc_type, exc_obj, exc_tb = sys.exc_info()
                # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                # file.write(f"{fname} : line {exc_tb.tb_lineno} \n")
                # file.write(f"{exc_type} : {error}")
                file.write("\n >>> moving on to another iteration\n")
                continue
            
        update_results(results,estimate = [iteration, 
                                           dist_list[:iteration+2] if iteration!=-1 else array([0,2]),
                                           unique([sta.extra.station_param.az 
                                                   for net in general_inv for sta in net]),
                                           unique([sta.extra.station_param.distance
                                                   for net in general_inv for sta in net]),
                                            dist_list,
                                            (general_st_bb,general_st_hf),
                                            parameters.time_max if iteration!=-1 else parameters.time_max_s])
        stadist = array([inv.select(station=tr.stats.station)[0][0].extra.station_param.distance
                              for tr in st ])
        try :
            if iteration == -1 or (iteration==0 and not parameters.distance.include_near):
                resultpath, data = do_estimations(ev, path, parameters, 
                                                  st,st_hf,inv, results, 
                                                  stadist, rt=iteration)
            else :
                resultpath = None
                resultpath, [inv, st_bb, st_hf] = do_estimations(ev, path, parameters, 
                                                  st,st_hf,inv,
                                                  results, stadist, rt=iteration)
                data[0] = general_inv.copy();data[1]+=st_bb;data[2]+=st_hf
        except Exception as error:
            with open(f"{path}/Results/{folder_name(parameters,iteration)}/0.ERROR.log","a") as file:
                file.write(f"#{str(UTCDateTime().now()):=^60}\n ")
                file.write("Error occured in energy estimation phase : ")
                file.write(correct_message(format_exc()))
                
                # exc_type, exc_obj, exc_tb = sys.exc_info()
                # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                # file.write(f"{fname} : line {exc_tb.tb_lineno} \n")
                # file.write(f"{exc_type} : {error}")
                file.write("\n >>> moving on to another iteration\n")
        else : 
            if os.path.exists(f"{path}/Results/{folder_name(parameters,iteration)}/0.ERROR.log"):
                os.remove(f"{path}/Results/{folder_name(parameters,iteration)}/0.ERROR.log")
        finally : continue
    
    if parameters.plot.level:
        try :
            do_plots(ev,parameters,resultpath,data)
        except :
            sys.stdout.write(f"\n{Back.GREEN + Style.DIM + Fore.BLACK}"
                              +"Cannot do figures - verify data and results"
                              +f"{Style.RESET_ALL}")
            with open(f"{path}/Results/{folder_name(parameters,iteration)}/0.ERROR.log","a") as file:
                file.write(f"#{str(UTCDateTime().now()):=^60}\n ")
                file.write("Cannot do figures - verify data and results : ")
                file.write(correct_message(format_exc()))
                
                # exc_type, exc_obj, exc_tb = sys.exc_info()
                # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                # file.write(f"{fname} : line {exc_tb.tb_lineno} \n")
                # file.write(f"{exc_type} : {error}")
                file.write("\n >>> finishing estimation\n")
            
            sys.stdout.write("\n >>> finishing estimation\n")
        else : 
            if os.path.exists(f"{path}/Results/{folder_name(parameters,iteration)}/0.ERROR.log"):
                os.remove(f"{path}/Results/{folder_name(parameters,iteration)}/0.ERROR.log")
            
    
    if (not os.path.exists(f"{path}/Data/0.ERROR.log")
        and not os.path.exists(f"{path}/Results/{folder_name(parameters,iteration)}/0.ERROR.log")):
    # if not os.path.exists(f"{path}/Results/{folder_name(parameters,iteration)}/0.ERROR.log"):
        sys.stdout.write(f"\n{Back.GREEN + Style.BRIGHT + Fore.BLACK}"
                          +f"{f'FINISHED SUCCESSFULLY'.center(65):65s}"
                          +f"{Style.RESET_ALL}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("path", help='path to folder of event')
    parser.add_argument("-rt", "--real_time", action="store_true",
                        help='apply the real time scheme')
    parser.add_argument("-n", "--new_param", action="store_true",
                        help='change parameter file')
    parser.add_argument("-sp", "--show_parameters", action="store_true",
                        help='write the parameters and set values for verification')
    
    args = parser.parse_args()
    # # "CATData/CATgcmtNZ/"
    # parameters = Parameters(args.path, new=args.new_param, show = args.show_parameters)
    # for evpath in [p.split("\\")[-2] for p in concatenate([glob(f"{args.path}/*M*{m:.1f}*/") 
    #                 for m in arange(5.,8.2,.1).round(1)])
    #                   if p.find("_")!=-1][::-1]:
    # # for evpath in [p.split("\\")[-2] for p in concatenate([glob(f"{args.path}/{m:.0f}*/") 
    # #                 for m in arange(2024,2026,1.).round(0)])
    # #                   if p.find("_")!=-1]:   
    #     if len(glob(f'{args.path}{evpath.replace("\\","/")}/Data/CloseRecords/*.MSEED'.replace(CODE_PATH,"")))==0:
    #         continue
    #     parameters = Parameters(args.path, new=args.new_param, show = args.show_parameters)
    #     main(f'{args.path}{evpath.replace("\\","/")}/'.replace(CODE_PATH,""),
    #           real_time=args.real_time,
    #           new_param=args.new_param,show_parameters=args.show_parameters,
    #           parameters=parameters)
    
    main(args.path,real_time=args.real_time,
          new_param=args.new_param,show_parameters=args.show_parameters)
  