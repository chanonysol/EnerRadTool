# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:50:28 2024

@author: solen

Description :
    Look for events on IRIS, GCMT etc within set parameters
"""
## ALLELHUIA
from obspy import UTCDateTime, read, read_inventory, read_events
from obspy.clients.fdsn import Client
from obspy.core.event import Catalog
import urllib.request 
from urllib.error import HTTPError
from numpy import array, argwhere, vstack, nan
import sys
from tqdm import tqdm
from colorama import Fore, Style

# providers = ["AusPass","BGR","BGS","CATAC","EarthScope","EMSC","ETH","GEOFON","GeoNet",
#              "ICGC","IESDMC","IGN","INGV","IPGP","ISC","KAGSR","KIT","KOERI","LMU","NCEDC","NIEP","NOA",
#              "NRCAN","ODC","RASPISHAKE","RESIF","RSBR","SCEDC","UIB-NORSAR","USGS","USP"]
routor = ["ISC","IRIS","USGS"]
#client = Client('IRIS')

def find_catalog_ISC(startdate,enddate,minmag,maxmag,region = [],
                     same_author=False,pref_solution="GCMT"):
    client = Client('ISC')
    sys.stdout.write(f"{Style.BRIGHT}{Fore.GREEN}Client ISC loaded.{'':10s}")
    sys.stdout.write(f"Retrieving catalog.{'':10s}") 
    
    cat = Catalog()
    
    sys.stdout.write(f"\nMag by mag: Mw{'':2s}") 
    mag = maxmag
    while mag>minmag:
        sys.stdout.write(f"{mag:3.1f} -")
        try:
            if len(region)==4: 
                cat2 = client.get_events(starttime=UTCDateTime(startdate), 
                              endtime=UTCDateTime(enddate),
                              maxdepth=70.,
                              minlatitude=region[1], maxlatitude=region[3],
                              minlongitude=region[0],maxlongitude=region[2],
                              minmagnitude = min(minmag,mag-0.5),maxmagnitude = mag,magnitudetype="mw",
                              includeallorigins=True)
            else :
                cat2 = client.get_events(starttime=UTCDateTime(startdate), 
                              endtime=UTCDateTime(enddate),
                              maxdepth=70.,
                              minmagnitude = min(minmag,mag-0.5),maxmagnitude = mag,magnitudetype="mw",
                              includeallorigins=True)
        except: continue
        else:
            cat+=cat2
        finally:mag-=.5
        
    sys.stdout.write(f"{Style.BRIGHT}{Fore.GREEN}Done.{Style.RESET_ALL}\n")
    if same_author : new_cat = Catalog()
    for (i,evi) in tqdm(enumerate(cat),total=len(cat),desc="Formating cat".center(24)):
        try:
            if len(evi.focal_mechanisms)!=1:
                pref_meca_id = str(evi.preferred_focal_mechanism_id).split("/")[-1]
                if pref_meca_id == None : continue
                f_id = argwhere(array([str(f.resource_id).split(";")[-1] 
                                       for f in evi.focal_mechanisms])==pref_meca_id)[0,0]
                evi.preferred_focal_mechanism_id = evi.focal_mechanisms[f_id].resource_id
                aut = array([f.creation_info.author 
                             if f.creation_info 
                             else "" for f in evi.focal_mechanisms])
                if (evi.preferred_focal_mechanism().creation_info.author!=pref_solution
                    and any(aut==pref_solution)) :
                    f_id = argwhere(aut==pref_solution)[-1,0]
                    evi.preferred_focal_mechanism_id = evi.focal_mechanisms[f_id].resource_id
                evi.focal_mechanisms=[evi.preferred_focal_mechanism()]
            else :
                evi.preferred_focal_mechanism_id = evi.focal_mechanisms[0].resource_id
                
            if (same_author and
                    evi.preferred_focal_mechanism().creation_info.author!=pref_solution):
                continue
            
            if (len(evi.focal_mechanisms)==1
                and len(evi.origins)==1
                and len(evi.magnitudes)==1) : continue
            
            if len(evi.origins)!=1: 
                f = evi.preferred_focal_mechanism()
                vals = [k for k,v in zip(list(f.keys()),list(f.values())) if v]
                if 'triggering_origin_id' in vals:
                    evi.preferred_origin_id = f.triggering_origin_id
                elif 'moment_tensor' in vals :
                    evi.preferred_origin_id = f.moment_tensor.derived_origin_id  
                evi.origins=[evi.preferred_origin()]
            
            if len(evi.magnitudes)!=1:
                if (evi.preferred_magnitude().creation_info.author!=pref_solution
                    or evi.preferred_magnitude().origin_id!=evi.preferred_origin_id):
                        mid = argwhere(array([str(m.origin_id).split('=')[-1] 
                                              for m in evi.magnitudes])
                                       ==str(evi.preferred_origin_id).split('=')[-1])[0,0]
                        evi.preferred_magnitude_id == evi.magnitudes[mid].resource_id
                evi.magnitudes=[evi.preferred_magnitude()]
            if same_author : new_cat.append(evi)
        except : continue
    if same_author : return new_cat
    else : return cat

cat = Catalog()
# for reg in [[160,-56,167,-46],[167,-46,178,-37],
#             [178,-37,-174,-24],[162,-23,175,-10],
#             [175,-24,-171,-12]]:
cat += find_catalog_ISC(UTCDateTime(2000,1,1),UTCDateTime().now(),7,10,
                       region = [160,-50,-172,-10],same_author=True,pref_solution="GCMT")

import pygmt
grid = pygmt.datasets.load_earth_relief(resolution="10m", region="g")*1e-3

fig = pygmt.Figure()
# projection azimuthale equidistant lon0/lat0/distance max/resolution
fig.basemap(region="g", projection="E%f/%f/30/20c" %
            (174, -30), frame=["af10", f"WSne"])
# PLOT
fig.grdimage(grid=grid, cmap="elevgray.cpt", verbose="q")  # plot elevation
fig.coast(shorelines="0.15p,black", land=None,
          resolution='h')  # coastlines
# Contour distance tout les 20 degres
# fig.contour(pen="1p", x=xv, y=yv, z=d, levels=20, annotation=20)
fig.plot(data=array([[160,-50,-172,-10]]), style='r+s', pen="1p,yellow")
#fig.plot(data=array([[160,-10,-170,-24]]), style='r+s', pen="1p,yellow")

elon,elat = array([(ev.preferred_origin().longitude, ev.preferred_origin().latitude) for ev in cat]).T
# for ev in cat :
#     elon,elat = ev.preferred_origin().longitude, ev.preferred_origin().latitude
fig.plot(x=elon, y=elat, style="c.2c", fill="red4", pen="white")

fig.savefig(f"C:/Users/solen/Documents/Ener/project/invtest/cat_ex_map.png",
                    transparent=True, crop=True, anti_alias=True, dpi = 300)
fig.show(verbose="q")

def repair_focal_meca(cat):
    for evi in cat:
        pref_meca_id = str(evi.preferred_focal_mechanism_id).split("/")[-1]
        if pref_meca_id == None : continue
        f_id = argwhere(array([str(f.resource_id).split(";")[-1] 
                               for f in evi.focal_mechanisms])==pref_meca_id)[0,0]
        evi.preferred_focal_mechanism_id = evi.focal_mechanisms[f_id].resource_id
        aut = array([f.creation_info.author 
                     if f.creation_info 
                     else "" for f in evi.focal_mechanisms])
    return cat

# from numpy import log10,floor

# def extract(evi):
#     foc = evi.preferred_focal_mechanism()
#     M0 = foc.moment_tensor.scalar_moment
#     M = array(list(foc.moment_tensor.tensor.values())[::2])
#     # expon = max(floor(log10(abs(M))).astype(int))
#     # M /= 10.**expon
#     # M.round(3)
#     ori = evi.preferred_origin()
#     lon = ori.lon
#     lat = ori.lat
#     depth = ori.depth*1e-3
#     date = ori.time
#     magi = evi.preferred_magnitude()
#     return
import os
autrecat = Catalog(); path = "CATData/CATgcmtNZ"
for evi in tqdm(cat2):
    pref_meca_id = "fmid"+str(evi.preferred_focal_mechanism_id).split("fmid")[-1]
    if pref_meca_id == None : continue
    try :
        f_id = argwhere(array([str(f.resource_id).split(";")[-1] 
                               for f in evi.focal_mechanisms])==pref_meca_id)[0,0]
        evi.preferred_focal_mechanism_id = evi.focal_mechanisms[f_id].resource_id
    except : 
        autrecat.append(evi)
        continue
    else:
        try:
            aut = evi.preferred_focal_mechanism().creation_info.author
            if aut!="GCMT":
                autrecat.append(evi)
                continue
            ind = argwhere(array([mag.creation_info.author for mag in evi.magnitudes])==aut)
            if ind.size==0:
                mag = evi.preferred_magnitude()
            else :
                mag = evi.magnitudes[ind[0,0]]
            ind = argwhere(array([ori.creation_info.author for ori in evi.origins])==aut)
            if ind.size==0:
                ori = evi.preferred_origin()
            else :
                ori = evi.origins[ind[0,0]]
            d = ori.time
            name = evi.event_descriptions[0].text.replace(" ","")
            fold = os.path.join(path,
                                  "_".join([f"{d.year:4d}.{d.month:02d}.{d.day:02d}.{d.hour:02d}.{d.minute:02d}",
                                  f"{mag.magnitude_type}{mag.mag:3.1f}",
                                  f"{name}"]))
            os.mkdir(fold)
            evi.write(f"{fold}/{d.year:4d}{d.month:02d}{d.day:02d}{d.hour:02d}{d.minute:02d}.cmt",
                      format  = "CMTSOLUTION")
        except:
            autrecat.append(evi)
            continue

# path="CATdata/fixedstatestevents"
# import os
# from tqdm import tqdm
# for ev in tqdm(cat):
#     name = "".join(ev.event_descriptions[0].text.split(" "))
#     date = str(ev.preferred_origin().time).replace("-","").replace("T","").replace(":","")[:-8]
#     mag = f"{ev.preferred_magnitude().magnitude_type}{ev.preferred_magnitude().mag:.2f}"
#     folder = "_".join([date,mag,name])
#     os.mkdir(f"{path}/{folder}")
#     ev.write(f"{path}/{folder}/{date}.cmt",format="CMTSOLUTION")

