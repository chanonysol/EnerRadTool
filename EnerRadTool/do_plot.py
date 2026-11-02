# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 15:21:43 2024

@author: solen
"""
import os
import sys

FILE_PATH = os.path.dirname(os.path.realpath(__file__)).replace('\\', '/')

DIR=f"{FILE_PATH}/../2.data_functions"
sys.path.insert(0, DIR)

from results_plots import plot_growth, result_in_cat, plot_theta_depth_apparent, plot_s_int, plot_s_map, plot_correct_staflux
from data_plots import plot_preprocessed_data_az, plot_preprocessed_data_dist, plot_preprocessed_spectrum
from results_writing import make_report
from other_plots import certain_report
from do_estimation import folder_name
from colorama import Back, Fore, Style
from numpy import array, argwhere
   
def do_plots(ev,param,resultpath,data):
    inv, st_bb, st_hf = data
    correct = (param.deviation_correction.threshold
                if param.deviation_correction.type == "result" 
                else False)
    if param.distance.include_near :
        stadist = array([inv.select(station=tr.stats.station)[0][0].extra.station_param.distance
                              for tr in st_bb])
        ind = argwhere(stadist<5.)[:,0]
        st_bb_close,st_hf_close = st_bb[:ind[-1]+1],st_hf[:ind[-1]+1]
        st_bb,st_hf = st_bb[ind[-1]+1:],st_hf[ind[-1]+1:]
        close_folder = folder_name(param,-1,close = True)
        close_folder = resultpath.replace(resultpath[-resultpath[::-1].find("/"):],close_folder)
    sys.stdout.write(f"\n{Back.GREEN + Style.DIM + Fore.BLACK}"
                     +f"{'Plotting  '.rjust(30):30s}"
                     +f"{Style.RESET_ALL}")
       
    if param.distance.include_near :
        # S waves station integrals
        plot_s_int(close_folder,ev,
                   savefile= True if (param.plot.level in ["report","all"] 
                                  or param.plot.details.ener_growth) else False, 
                   ext = param.general.do_plot, 
                   redo=param.redoing_results.force_plot)
    
    if param.waves.P_teleseismic :
        # Comparison to catalogue
        result_in_cat(resultpath,ev, correct = correct, 
                      savefile= True if (param.plot.level in ["minimum","report","all"] 
                                     or param.plot.details.cat_region_results) else False, 
                      ext = param.general.do_plot, 
                      redo=param.redoing_results.force_plot)
        
        # Apparent stress
        plot_theta_depth_apparent(resultpath,ev,correct = correct,
                                savefile= True if (param.plot.level in ["minimum","report","all"] 
                                               or param.plot.details.apparent_stress) else False, 
                                ext = param.general.do_plot, 
                                redo=param.redoing_results.force_plot)
        # Signals
        plot_preprocessed_data_dist(resultpath, st_bb, st_hf, inv,
            correct = (param.deviation_correction.type == "result"),
            savefile= True if (param.plot.level in ["minimum","report","all"] 
                               or param.plot.details.signal_dist) else False,
            ext = param.general.do_plot)
        plot_preprocessed_data_az(resultpath, st_bb, st_hf, inv,
            correct = (param.deviation_correction.type == "result"),
            savefile= True if (param.plot.level in ["minimum","report","all"]
                           or param.plot.details.signal_az) else False,
            ext = param.general.do_plot)
        
        # plot_preprocessed_spectrum(resultpath, st_bb, st_hf, inv,
        #     correct = (param.deviation_correction.type == "result"),
        #     savefile= True if (param.plot.level in ["report","all"]
        #                     or param.plot.details.spectrum) else False,
        #     ext = param.general.do_plot)
    
        for i in range(0 if param.plot.step=="all" else int(param.distance.totalstep)-1,
                       int(param.distance.totalstep)):
            #E growth 
            plot_growth(resultpath, rt = i, 
                        savefile= True if (param.plot.level in ["minimum","report","all"] 
                                       or param.plot.details.ener_growth) else False, 
                        ext = param.general.do_plot, 
                        redo=param.redoing_results.force_plot)
            # Corrected
            plot_growth(resultpath, rt = i, correct = True,
                        savefile= True if (param.plot.level in ["minimum","report","all"] 
                                       or param.plot.details.ener_growth_corrected) else False, 
                        ext = param.general.do_plot, 
                        redo=param.redoing_results.force_plot)
            # Energy fluw at station + correction
            plot_correct_staflux(resultpath, correct = correct, rt=i, 
                                 savefile= True if (param.plot.level in ["all"] 
                                                or param.plot.details.sta_ener_flux_deviation) else False, 
                                 ext = param.general.do_plot, 
                                 redo=param.redoing_results.force_plot)
            # Certainty evaluation
            certain_report(resultpath,ev, correct = correct, rt=i,
                           savefile= True if (param.plot.level in ["minimum","report","all"] 
                                          or param.plot.details.confidence_radar) else False, 
                           ext = param.general.do_plot, 
                           redo=param.redoing_results.force_plot)
        # Certainty evaluation evolution accross iterations
        certain_report(resultpath,ev, correct = correct, rt=i,
                       savefile= True if (param.plot.level in ["minimum","report","all"] 
                                      or param.plot.details.confidence_radar) else False, 
                       ext = param.general.do_plot, multi=True,
                       redo=param.redoing_results.force_plot)
    
    if (param.plot.level in ["report","all"] or param.plot.details.map_sta_dev) :
        from map_plots import plot_stations_contrib_map
        if param.waves.P_teleseismic:
            plot_stations_contrib_map(resultpath, ev, inv, st_bb, st_hf, correct = correct,
                    savefile= True if (param.plot.level in ["report","all"] 
                                   or param.plot.details.cat_region_results) else False,
                    ext = param.general.do_plot,
                    redo=param.redoing_results.force_plot)
        if param.distance.include_near :
            plot_s_map(close_folder,ev,
                       savefile= True if (param.plot.level in ["minimum","report","all"] 
                                      or param.plot.details.cat_region_results) else False, 
                       ext = param.general.do_plot, 
                       redo=param.redoing_results.force_plot)
    
    if param.general.do_report and param.waves.P_teleseismic:
        sys.stdout.write(f"\n{Back.GREEN + Style.DIM + Fore.BLACK}"
                         +f"{'Compiling a report  '.rjust(30):30s}"
                         +f"{Style.RESET_ALL}")
        make_report(resultpath,correct=correct)
        