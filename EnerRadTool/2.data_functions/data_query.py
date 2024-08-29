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

#parameters to select channels see
#https://docs.obspy.org/packages/autogen/obspy.clients.fdsn.mass_downloader.restrictions.Restrictions.html#obspy.clients.fdsn.mass_downloader.restrictions.Restrictions

if "MDL" not in globals():
    # No specified providers will result in all known ones being queried.
    # "GEONET"
    MDL = MassDownloader(providers = ["NCEDC",
                                      "RESIF","TEXNET", "IRIS"])

    logger = logging.getLogger("obspy.clients.fdsn.mass_downloader")
    #https://docs.python.org/3/library/logging.html#logging-levels
    # For details on logging levels
    # NOTSET / DEBUG / INFO / WARNING / ERROR / CRITICAL
    logger.setLevel(logging.WARNING)


def get_mseed_storage(network, station, location, channel, starttime, endtime):
    # Returning True means that neither the data nor the StationXML file
    # will be downloaded.
    if f"{network}.{station}.{location}.{channel}.mseed" in os.listdir(ROOT):
        return True
    # If a string is returned the file will be saved in that location.
    return os.path.join(ROOT, f"{network}.{station}.{location}.{channel}.mseed")
# mseed_storage = get_mseed_storage

def get_stationxml_storage(network, station, channels, starttime, endtime):
    available_channels = []
    missing_channels = []
    for location, channel in channels:
        if f"{network}.{station}.{location}.{channel}.xml" in os.listdir(ROOT+"/stations"):
            available_channels.append((location, channel))
        else:
            missing_channels.append((location, channel))
    filename = os.path.join(ROOT, f"stations/{network}.{station}.{location}.{channel}.xml")
    return {
        "available_channels": available_channels,
        "missing_channels": missing_channels,
        "filename": filename}

def general_query(ev,path,startime,endtime):
    origin_time = ev.date
    domain = CircularDomain(latitude=ev.lat, longitude=ev.lon,
                            minradius=25, maxradius=80)

    restrictions = Restrictions(
        starttime=origin_time + startime - 2*60,
        endtime=origin_time + endtime,
        reject_channels_with_gaps=True, minimum_interstation_distance_in_m=30,
        minimum_length=0.95, channel_priorities=["BHZ", "SHZ"],
        location_priorities=["", "00", "10"], sanitize=True)

    globals()['ROOT'] = f"{path}Data/"
    mseed_storage=get_mseed_storage
    stationxml_storage=get_stationxml_storage

    MDL.download(domain, restrictions, mseed_storage=mseed_storage,
                  stationxml_storage=stationxml_storage,print_report=True)
    #sys.stdout.write(f"{Fore.RED}{logger.log}{Style.RESET_ALL}")
    sys.stdout.write(f"{Fore.BLUE + Style.BRIGHT}"
                     +"      ¤ BHZ channels done "
                     +f"{Style.RESET_ALL}\n")

    restrictions = Restrictions(
        starttime=origin_time + startime - 2*60,
        endtime=origin_time + startime,
        reject_channels_with_gaps=True, minimum_interstation_distance_in_m=30,
        minimum_length=0.95, channel_priorities=["HHZ"],
        location_priorities=["", "00", "10"], sanitize=True)

    MDL.download(domain, restrictions, mseed_storage=mseed_storage,
                  stationxml_storage=stationxml_storage,print_report=True)
    sys.stdout.write(f"{Fore.BLUE + Style.BRIGHT}"
                     +"      ¤ HHZ channels done "
                     +f"{Style.RESET_ALL}\n")


def selected_query(ev,path,minrad,maxrad,startime,endtime,channels,nbiter):
    origin_time = ev.date

    domain = CircularDomain(latitude=ev.lat, longitude=ev.lon,
                            minradius=minrad, maxradius=maxrad)
    globals()['ROOT'] = f"{path}Data/iter{nbiter}/"
    mseed_storage=get_mseed_storage
    stationxml_storage=get_stationxml_storage

    if "BH" in channels:
        STARTIME = time()
        restrictions = Restrictions(
            starttime=origin_time + startime - 60,
            endtime=origin_time + endtime + 60,
            reject_channels_with_gaps=True, minimum_interstation_distance_in_m=1000,
            minimum_length=0.95, channel_priorities=["BHZ"],
            location_priorities=["", "00", "10"], sanitize=True)


        MDL.download(domain, restrictions, mseed_storage=mseed_storage,
                      stationxml_storage=stationxml_storage,print_report=True)
        sys.stdout.write(f"      ¤ SHZ channels done "
                         +f"{Style.RESET_ALL}"
                         +f" (lasted {time()-STARTTIME: 3.0f}s) \n")

    if "HH" in channels:
        STARTIME = time()
        restrictions = Restrictions(
            starttime=origin_time + startime - 60,
            endtime=origin_time + endtime + 60,
            reject_channels_with_gaps=True, minimum_interstation_distance_in_m=1000,
            minimum_length=0.95, channel_priorities=["HHZ"],
            location_priorities=["", "00", "10"], sanitize=True)

        MDL.download(domain, restrictions, mseed_storage=mseed_storage,
                      stationxml_storage=stationxml_storage,print_report=True)
        sys.stdout.write(f"      ¤ SHZ channels done "
                         +f"{Style.RESET_ALL}"
                         +f" (lasted {time()-STARTTIME: 3.0f}s) \n")

    if "SH" in channels:
        STARTIME = time()
        restrictions = Restrictions(
            starttime=origin_time + startime - 60,
            endtime=origin_time + endtime + 60,
            reject_channels_with_gaps=True, minimum_interstation_distance_in_m=1000,
            minimum_length=0.95, channel_priorities=["SHZ"],
            location_priorities=["", "00", "10"], sanitize=True)

        MDL.download(domain, restrictions, mseed_storage=mseed_storage,
                      stationxml_storage=stationxml_storage,print_report=True)
        sys.stdout.write(f"      ¤ SHZ channels done "
                         +f"{Style.RESET_ALL}"
                         +f" (lasted {time()-STARTTIME: 3.0f}s) \n")


def waiting_P_smileys(ev, limit_time, dist):
    smiley_characters = ["๑(◕‿◕)๑","°˖✧◝(⁰▿⁰)◜✧˖°","ʅ（´◔౪◔）ʃ",
                         "ヽ(⌐■_■)ノ♪♬","┬┴┬┴┤ʕ•ᴥ├┬┴┬┴","U・ﻌ・U",
                         "(⊙_☉)","(~‾⌣‾)~","(◕▿◕✿)"]
    #
    first_enter = 0
    lign_count = 0
    c_time =int(limit_time - (UTCDateTime(time())- ev.date))
    while c_time>0:
        c_time =int(limit_time - (UTCDateTime(time())- ev.date))
        if first_enter == 0: #
            sys.stdout.write(f"waiting for the first P-waves to reach '{dist}'° distance :\n")
            first_enter = 1
        if c_time > 60 : #
            sys.stdout.write(f" {c_time//60:1.0f}m {c_time%60:02.0f}s")
        elif c_time >0 :
            sys.stdout.write(f" {c_time:02.0f}s")
        #
        smileys = "".join([" -- "+smiley_characters[random.randint(0, 8)]+" --" for i in range(3)])
        c_time =int(limit_time - (UTCDateTime(time())- ev.date))
        for i in range(min([c_time,len(smileys)])) :
            sleep(1)
            sys.stdout.write(smileys[i%len(smileys)])
        lign_count+=1
        if lign_count == 1:
            sys.stdout.write("\n")
            lign_count=0
        c_time =int(limit_time - (UTCDateTime(time())- ev.date))

def real_time_query(ev,path,channels,current_dist,distlist,nbiter):
    start =  time()

    nbBH = len(glob(f"{path}/Data/iter{nbiter}/*BHZ.mseed"))
    nbHH = len(glob(f"{path}/Data/iter{nbiter}/*HHZ.mseed"))
    nbSH = len(glob(f"{path}/Data/iter{nbiter}/*SHZ.mseed"))
    nsta = len(glob(f"{path}/Data/iter{nbiter}/stations/*.xml"))

    times,[_,max_duration] = request_val_p_train(['P','time','duration'],
                                                 ev.depth,distlist,
                                                 model_name="iasp91")[0]


    #nbiter = argwhere(array(distlist) == current_dist)[0,0]
    next_time, next_dist = times[nbiter+1],distlist[nbiter+1]
    waiting_P_smileys(ev,next_time,next_dist)

    #STARTIME = time()
    # TODO Make a calc time file
    sys.stdout.write(f"{Fore.YELLOW} > Looking until {next_dist:2.0f}° distance ")
    selected_query(ev,path,current_dist,next_dist,times[0],
                   times[-1] + max_duration,channels,nbiter)

    nbBH = len(glob(f"{path}/Data/iter{nbiter}/*BHZ.mseed")) - nbBH
    nbHH = len(glob(f"{path}/Data/iter{nbiter}/*HHZ.mseed")) - nbHH
    nbSH = len(glob(f"{path}/Data/iter{nbiter}/*SHZ.mseed")) - nbSH
    nsta = len(glob(f"{path}/Data/iter{nbiter}/stations/*.xml")) - nsta

    sys.stdout.write(f"{Back.CYAN + Fore.BLACK}>>> Result query "
                     +f"{Style.RESET_ALL} ")
    sys.stdout.write(f"{Fore.CYAN + Style.BRIGHT}"
                      +f"Added {nsta} stations ("
                      +f"{nbBH} BHZ - {nbHH} HHZ - {nbSH} SHZ)"
                      +f"{Style.RESET_ALL}\n")