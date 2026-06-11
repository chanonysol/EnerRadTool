# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:50:28 2024

@author: solen
"""
import os
import random
import sys
import logging

from colorama import Fore, Back, Style
from time import time, sleep
from obspy.clients.fdsn.mass_downloader import CircularDomain, \
    Restrictions, MassDownloader
from obspy import UTCDateTime
from data_utils import request_val_p_train
from glob import glob
from numpy import argwhere, array
from shutil import copyfile
import os
#parameters to select channels see
#https://docs.obspy.org/packages/autogen/obspy.clients.fdsn.mass_downloader.restrictions.Restrictions.html#obspy.clients.fdsn.mass_downloader.restrictions.Restrictions

# from obspy.clients.fdsn import Client
# client = Client("GEONET")
# inv = client.get_stations(level="response",channel="?HN,?HE",
#                           latitude = -43, longitude = 172,
#                           maxradius = 5, startafter=UTCDateTime("2000-01-01"),
#                           endafter=UTCDateTime.now())


if "MDL" not in globals():
    logger = logging.getLogger("obspy.clients.fdsn.mass_downloader")
    logger.setLevel(logging.WARNING)
    #https://docs.python.org/3/library/logging.html#logging-levels
    # For details on logging levels
    # NOTSET / DEBUG / INFO / WARNING / ERROR / CRITICAL
    
    # No specified providers will result in all known ones being queried.
    # "GEONET"
    MDL = MassDownloader(providers = ["IRIS", 
                                      "AUSPASS", "GEONET","NCEDC", 
                                      #"IPGP", 
                                      "http://service-nrt.geonet.org.nz",
                                      #"RESIF" 
                                      ])
    
    while logger.hasHandlers():
        logger.handlers[0].close
        logger.removeHandler(logger.handlers[0])


def get_mseed_storage(network, station, location, channel, starttime, endtime):
    # Returning True means that neither the data nor the StationXML file
    # will be downloaded.
    if f"{network}.{station}.{location}.{channel}.mseed" in os.listdir(globals()['ROOT']):
        return True
    # If a string is returned the file will be saved in that location.
    return os.path.join(globals()['ROOT'],f"{network}.{station}.{location}.{channel}.mseed")
# mseed_storage = get_mseed_storage

def get_stationxml_storage(network, station, channels, startime,endtime):
    available_channels = []
    missing_channels = []
    for location, channel in channels:
        if f"{network}.{station}.{location}.{channel}.xml" in os.listdir(globals()['ROOT'].replace("Records","Stations")):
            available_channels.append((location, channel))
        else:
            missing_channels.append((location, channel))
    filename = os.path.join(globals()['ROOT'].replace("Records","Stations"), f"{network}.{station}.{location}.{channel}.xml")
    return {
        "available_channels": available_channels,
        "missing_channels": missing_channels,
        "filename": filename}

def check4fixed_stations(path,min_dist,max_dist):
    from  numpy import dtype, loadtxt, concatenate
    file = 'EnerRadTool/2.data_functions/sta_cor_perma0.5.dat'
    mat = loadtxt(file,usecols = 0,comments="#",dtype=str)
    
    L = concatenate([glob(f"{path}Data/Records/{n}*") for n in mat])
    dists = array([int(p[-8:-6]) for p in L])
    ind = argwhere((dists>=min_dist) & (dists<max_dist))[:,0]
    L = L[ind]
    if L.size==0: return False
    else :
        Lsta = concatenate([glob(f"{path}Data/Stations/{n}*") for n in [p.split("\\")[-1][:-10] for p in L]])
        for p in L:
            copyfile(p,p.replace("Records","FixedRecords"))
        for p in Lsta:
            copyfile(p,p.replace("Stations","FixedStations"))
        return True
    

def fixed_net_query(ev,path,minrad,maxrad,startime,endtime,channels):
    from  numpy import dtype, loadtxt, unique
    file = 'EnerRadTool/2.data_functions/sta_cor_perma0.5.dat'
    mat = loadtxt(file,usecols = 0,comments="#",dtype=str)
    
    netlist,stalist = array([n.split(".")[:2] for n in mat]).T
    netlist = unique(netlist)
    stalist = unique(stalist)
    
    # if len(glob(f"{path}/Data/Records/*d*.MSEED"))!=0:
    #     for p in glob(f"{path}/Data/Records/{netlist}.{stalist}*d*.MSEED"):
    #         copyfile(p,p.replace("Records","FixedRecords"))
    #     for p in glob(f"{path}/Data/Stations/{netlist}.{stalist}*"):
    #         copyfile(p,p.replace("Stations","FixedStations"))
            
    #     sys.stdout.write(f"{Fore.YELLOW + Style.DIM}"
    #                       +f" > Query done before - copying files   "
    #                       +f"{Style.RESET_ALL}")
    #     return
    
    origin_time = ev.date

    domain = CircularDomain(latitude=ev.lat, longitude=ev.lon,
                            minradius=minrad, maxradius=maxrad)

    globals()['ROOT'] = f"{path}Data/FixedRecords/"
    mseed_storage=get_mseed_storage
    stationxml_storage=get_stationxml_storage

    STARTTIME = time()
    for channel in channels :
        restrictions = Restrictions(
            starttime= origin_time + startime,
            endtime= origin_time + endtime,
            channel = channel,
            network = ','.join(netlist),
            station = ','.join(stalist),
            location = "*",
            reject_channels_with_gaps=True,
            minimum_length=0.95,
            minimum_interstation_distance_in_m=100.0,
            sanitize=True)

        MDL.download(domain, restrictions, mseed_storage=mseed_storage,
                      stationxml_storage = stationxml_storage,
                      print_report=True)
    sys.stdout.write(f"      ¤ fixed stations done "
                     +f"{Style.RESET_ALL}"
                     +f" (lasted {time()-STARTTIME: 3.0f}s) \n")


def selected_query(ev,path,minrad,maxrad,startime,endtime,channels):
    origin_time = ev.date

    domain = CircularDomain(latitude=ev.lat, longitude=ev.lon,
                            minradius=minrad, maxradius=maxrad)
    
    globals()['ROOT'] = f"{path}Data/Records/"
    mseed_storage=get_mseed_storage
    stationxml_storage=get_stationxml_storage

    STARTTIME = time()
    for channel in channels :
        restrictions = Restrictions(
            starttime= origin_time + startime,
            endtime= origin_time + endtime,
            channel = channel,location="*",
            exclude_networks=["1?","2?","3?","4?","5?","6?","7?","8?","9?",
                              "X?","Y?","Z?"],
            reject_channels_with_gaps=True,
            minimum_length=0.95,
            minimum_interstation_distance_in_m=100.0,
            sanitize=True)

        MDL.download(domain, restrictions, mseed_storage=mseed_storage,
                      stationxml_storage = stationxml_storage,
                      print_report=True)
        sys.stdout.write(f"      ¤ other {channel} channels done "
                         +f"{Style.RESET_ALL}"
                         +f" (lasted {time()-STARTTIME: 3.0f}s) \n")

def regional_query(ev,path,startime,endtime,channels):
    from obspy.geodetics.base import degrees2kilometers,kilometer2degrees
    origin_time = ev.date

    domain = CircularDomain(latitude=ev.lat, longitude=ev.lon,minradius=0,
                            maxradius=kilometer2degrees(200))
    
    globals()['ROOT'] = f"{path}Data/CloseRecords/"
    # if not os.path.exists(globals()['ROOT']):
    #     os.mkdir(globals()['ROOT'])
        #os.mkdir(globals()['ROOT'].replace("Records","Stations"))
    mseed_storage=get_mseed_storage
    stationxml_storage=get_stationxml_storage

    STARTTIME = time()
    for channel in channels :
        restrictions = Restrictions(
            starttime= origin_time + startime - 60,
            endtime= origin_time + endtime,
            channel = channel, location="*",
            exclude_networks=["1?","2?","3?","4?","5?","6?","7?","8?","9?",
                              "X?","Y?","Z?"],
            exclude_stations=["DSV"],
            reject_channels_with_gaps=True,
            minimum_length=0.95,
            minimum_interstation_distance_in_m=100.0,
            sanitize=True)

        MDL.download(domain, restrictions, mseed_storage=mseed_storage,
                      stationxml_storage = stationxml_storage,
                      print_report=True)
        sys.stdout.write(f"      ¤ close {channel} channels done "
                         +f"{Style.RESET_ALL}"
                         +f" (lasted {time()-STARTTIME: 3.0f}s) \n")

def waiting_P_smileys(ev, limit_time, dist):
    smiley_characters = ["๑(◕‿◕)๑","°˖✧◝(⁰▿⁰)◜✧˖°","ʅ（´◔౪◔）ʃ",
                         "ヽ(⌐■_■)ノ♪♬","┬┴┬┴┤ʕ•ᴥ├┬┴┬┴","U・ﻌ・U",
                         "(⊙_☉)","(~‾⌣‾)~","(◕▿◕✿)"]
    #
    first_enter = 0
    lign_count = 0
    c_time =int(limit_time - (UTCDateTime.now() - ev.date))
    while c_time>0:
        c_time =int(limit_time - (UTCDateTime.now() - ev.date))
        if first_enter == 0: #
            sys.stdout.write(f"{Fore.YELLOW + Style.DIM}"
                              +f" > Waiting for first P-waves to reach {dist}° :\n"
                              +f"{Style.RESET_ALL}")
            first_enter = 1
        if c_time > 60 : #
            sys.stdout.write(f"{Fore.RED + Style.BRIGHT}"
                            +f" {c_time//60: 1.0f}m {c_time%60:02.0f}s"
                            +f"{Style.RESET_ALL}")

        #
        smileys = "".join([" -- "+smiley_characters[random.randint(0, 8)]+" --" for i in range(3)])
        c_time =int(limit_time - (UTCDateTime(time())- ev.date))
        for i in range(min([c_time,len(smileys)])) :
            sleep(1)
            sys.stdout.write(f"{Fore.YELLOW + Style.DIM}"
                              +smileys[i%len(smileys)]
                              +f"{Style.RESET_ALL}")
        lign_count+=1
        if lign_count == 1:
            sys.stdout.write("\n")
            lign_count=0
        c_time =int(limit_time - (UTCDateTime.now() - ev.date))

# def real_time_query(ev,path,tmax,channels,current_dist,distlist):
#     start =  time()

#     nbBH = len(glob(f"{path}/Data/All/*BHZ.mseed"))
#     nbHH = len(glob(f"{path}/Data/All/*HHZ.mseed"))
#     nbSH = len(glob(f"{path}/Data/All/*SHZ.mseed"))
#     nsta = len(glob(f"{path}/Data/All/stations/*.xml"))

#     times,[_,max_duration] = request_val_p_train(['P','time','duration'],
#                                                  ev.depth,distlist,
#                                                  model_name="iasp91")[0]

#     if max_duration<tmax : max_duration = tmax
#     nbiter = argwhere(array(distlist) == current_dist)[0,0]
#     next_time, next_dist = times[nbiter+1],distlist[nbiter+1]
#     waiting_P_smileys(ev,next_time,next_dist)

#     #STARTIME = time()
#     # TODO Make a calc time file
#     sys.stdout.write(f"{Fore.YELLOW} > Looking from {current_dist:2.0f}° until {next_dist:2.0f}° distance ")

#     selected_query(ev,path,current_dist+0.1,next_dist,0,
#                        times[-1] + max_duration,channels)

#     nbBH = len(glob(f"{path}/Data/All/*BHZ.mseed"))-nbBH
#     nbHH = len(glob(f"{path}/Data/All/*HHZ.mseed"))-nbHH
#     nbSH = len(glob(f"{path}/Data/All/*SHZ.mseed"))-nbSH
#     nsta = len(glob(f"{path}/Data/All/stations/*.xml"))-nsta

#     sys.stdout.write(f"{Back.CYAN + Fore.BLACK}>>> Result query "
#                      +f"{Style.RESET_ALL} ")
#     sys.stdout.write(f"{Fore.CYAN + Style.BRIGHT}"
#                       +f"Added {nsta} signals ("
#                       +f"{nbBH} BHZ - {nbHH} HHZ - {nbSH} SHZ)"
#                       +f"{Style.RESET_ALL}\n")

