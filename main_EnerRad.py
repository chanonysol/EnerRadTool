# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 09:51:52 2024

@author: solen
"""
__version__ = "v2.2.2"
# =============================================================================
# LIBRARIES IMPORT
# =============================================================================
import argparse
import sys
import os
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
from results_writing import results_dictionnary_init, update_results, AttribDict, write_rt_cat
from do_plot import do_plots

from numpy import linspace, concatenate, loadtxt, dtype, nan
from data_utils import find_min_dist, find_az_dist_gap
from obspy import UTCDateTime
from copy import deepcopy

from numpy import array, argwhere
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

def main(path,stages = 0, real_time=True, new_param=False, show_parameters=False, 
         parameters = False, reg_lim = None):
    path = f"{CODE_PATH}/{path}"
    
    sys.stdout.write(f"\n{Back.CYAN + Style.DIM + Fore.BLACK}"
                     +f"{('Initialisation'+' '*5+str(path.split('/')[-2]).replace('_',' ')).center(65)}"
                     +f"{Style.RESET_ALL}\n")
    
    if not parameters:
        parameters = Parameters(path, new=new_param, show = show_parameters)
        skip_s_figures = False #added to manage swaves try that failed #TODO replace with distance functions
        sys.stdout.write("-------> Parameters loaded\n")

    ev = Event(path)
    if reg_lim and ev.sreg!=reg_lim:
        sys.stdout.write(f"{Fore.CYAN + Style.DIM}"
                         +"\n > Not targeted region. Moving on."
                         +f"{Style.RESET_ALL}\n")
        return
    if show_parameters:
        ev.arglist()
    sys.stdout.write("-------> Event loaded\n")

    if glob(f"{path}/*log*") == []:
        file = open(f"{path}/logfile.txt","x")
        file.close()
      
    if ((parameters.distance.include_near) and
            (ev.reg == "SW.Pacific" or not ev.sreg)) :
        sys.stdout.write(f"{Fore.CYAN + Style.DIM}"
                         +"\n > Region out of scope for Swaves. Moving on."
                         +f"{Style.RESET_ALL}\n")
        return
      
    #adapt dist to depth
    parameters.distance.min = find_min_dist(ev,parameters)    
        
    if parameters.waves.P_teleseismic or parameters.waves.P_regional:
        dist_list = linspace(parameters.distance.min,parameters.distance.max,
                             int(parameters.distance.totalstep+1),endpoint=True).round(0)
        
        if not real_time :
            dist_list = dist_list[[0,-1]]
        dist_list[0]=parameters.distance.min
        
        if dist_list[-1]<= parameters.distance.min:
            dist_list=dist_list[:-1]
    else:
        dist_list = array([])

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
    
    if stages == 42 : return ev, parameters, results, path, dist_list, None
    
    for (iteration,(current_dist,next_dist)) in enumerate(
            zip(concatenate(([0] if parameters.distance.include_near else [],dist_list[:-1])),
                concatenate(([2] if parameters.distance.include_near else [],dist_list[1:])))):
        if parameters.distance.include_near : iteration-=1
        if stages !=6:
            sys.stdout.write(f"\n{Back.CYAN + Style.DIM + Fore.BLACK}"
                              +f"{f'n°{iteration} iteration'.center(65):65s}"
                              +f"{Style.RESET_ALL}\n")
        if stages in [0,1,3,5]:
            try :
                do_query(ev, path, parameters, current_dist, next_dist,
                         results, rt=iteration)
            except Exception:
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
        if stages in [0,2,3,4,5,7]:
            try :
                STARTTIME = time()
                st,st_hf,inv = do_preprocessing(ev, path, parameters, 
                                                current_dist,
                                                next_dist, 
                                                results,rt = iteration 
                                                if real_time else real_time,
                                                snrlim = parameters.deviation_correction.snr_max)
                
                update_results(results, 
                                pre_time = (#results['information']['time']['pre_time'] if 'time' in list(results["information"].keys()) else 0 +
                                            time() - STARTTIME ))
                general_st_bb += st.copy()
                general_st_hf += st_hf.copy()
                general_inv += inv.copy()
                
                update_results(results,estimate = [iteration, 
                                                   dist_list[:iteration+2] if iteration!=-1 else array([0,2]),
                                                   find_az_dist_gap(None,iteration,correct=False,inv=general_inv),
                                                    (general_st_bb,general_st_hf),
                                                    parameters.time_max if iteration!=-1 else parameters.time_max_s])
            except FileNotFoundError :
                if iteration==-1: 
                    # parameters.distance.include_near = False
                    skip_s_figures = True
                    continue
                else : return
            except Exception:
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
        if stages in [0,4,5,7]:
            stadist = array([inv.select(station=tr.stats.station)[0][0].extra.station_param.distance
                                  for tr in st ])
            try :
                if iteration == -1 or (iteration==0 and 
                                       (not parameters.distance.include_near 
                                        or skip_s_figures)):
                    resultpath, data = do_estimations(ev, path, parameters, 
                                                      st,st_hf,inv, results, 
                                                      stadist, rt=iteration)
                else :
                    resultpath = None
                    resultpath, [inv, st_bb, st_hf] = do_estimations(ev, path, parameters, 
                                                      st,st_hf,inv,
                                                      results, stadist, rt=iteration)
                    data[0] = general_inv.copy();data[1]+=st_bb;data[2]+=st_hf
            except Exception:
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
    
    if stages in [2,3]:
        data = [general_inv, general_st_bb, general_st_hf]
        return ev, parameters, results, path, dist_list, data
    
    if stages in [0,4,5,7]:
        write_rt_cat(ev,results)
    
    if parameters.plot.level and stages in [0,6,7]:
        if skip_s_figures:
            parameters.distance.include_near = False
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
    return None,None,None,path,None,None

# import tarfile
# import os.path

# def make_tarfile(output_filename, source_dir):
#     with tarfile.open(output_filename, "w:gz") as tar:
#         tar.add(source_dir, arcname=os.path.basename(source_dir))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("path", help='''path to folder of event or catalog; 
                        [if rt] Folder contains the cmt file and parameter file of the source;
                        [if cat] Folder contains parameters file and one separate folder per source with the corresponding cmt file
                        ''')
    parser.add_argument("-rt", "--real_time", action="store_true",
                        help='apply the real time code or one source')
    parser.add_argument("-n", "--new_param", action="store_true",
                        help='change parameter file')
    parser.add_argument("-sp", "--show_param", action="store_true",
                        help='print the code and source parameters for verification')
    parser.add_argument("-cat", "--catalogue",action="store_true",
                        help='apply calculation for multiple sources in folder')
       
    parser.add_argument("-stage", choices=[1,2,3,4,5,
                                           #6,
                                           7,42], type=int,default = 0, 
                        help="separate stages [Default : 0 = do all]"
                        +"\n; 1 = Only do data collections"
                        +"\n; 2 = Only do data processing"
                        +"\n; 3 = Do both data collection and processing"
                        +"\n; 4 = Do both data processing and calculation"
                        +"\n; 5 = Do all but figures"
                        #+"\n; 6 = Only do plots"
                        +"\n; 7 = Do calculation and plots."
                        +"\n (If necessarry previous steps have not been performed, the path and missing stages will be stored in an error file)"
                        +"\n; 42 = Load event and parameters and return them for manual testing"
                        )
    
    args = parser.parse_args()
    
    if args.catalogue:
        PARAMETERS = Parameters(args.path, new=args.new_param, show = args.show_parameters)
        for evpath in [p.split("\\")[-2] for p in concatenate([glob(f"{args.path}/*/")])
                          if p.find("_")!=-1][::-1]:
            #parameters = Parameters(args.path, new=args.new_param, show = args.show_parameters)
            main(f'{args.path}{evpath.replace("\\","/")}/'.replace(CODE_PATH,""),
                 stages = args.stage, real_time=args.real_time,
                  new_param=args.new_param,show_parameters=args.show_parameters,
                  parameters=deepcopy(PARAMETERS))
    
    if not args.catalogue and args.real_time:
        ev,parameters,results,path,dist_list,data = main(args.path,stages = args.stage,real_time=args.real_time,
              new_param=args.new_param,show_parameters=args.show_param)
        if data:
            inv,st,st_hf = data
     