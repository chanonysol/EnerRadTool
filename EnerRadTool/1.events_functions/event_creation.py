# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:50:28 2024

@author: solen

Description :
    Make a folder for the event, with a cmt file, data folder
    and automatic personalized parameter file
"""
import json
with open("exresp.json","r") as file : test = json.load(file) #response RCET

from obspy import UTCDateTime, read, read_inventory, read_events
with "C:/Users/solen/Downloads/quakeml.xml" as file : ev = read_events(file)[0] #xml event

with open("template.cmt","r") as file : temp=file.read() #cmt template

try : ori = ev.preferred_origin()
except : ori = ev.origins[0]
finally :
    lon,lat,date,dep = ori.longitude, ori.latitude, ori.time, ori.depth
    temp = temp.replace("@lat",f"{lat:+9.4}"); temp =temp.replace("@lon",f"{lon:+9.4}")
    temp = temp.replace("@dep",f"{dep*1e-3:5.2f}")
    temp = temp.replace("@ye",f"{date.year:4.0f}")
    temp = temp.replace("@mo",f"{date.month:2.0f}"); temp =temp.replace("@da",f"{date.day:2.0f}")
    temp = temp.replace("@ho",f"{date.hour:2.0f}"); temp =temp.replace("@mi",f"{date.minute:2.0f}")
    temp = temp.replace("@se",f"{date.second}.{date.microsecond*1e-4:.0f}")
    idev = str(date).replace("-","").replace(":","").replace("T","")[:-10]
    temp = temp.replace("@ref",f"{idev:12s}")
    temp = temp.replace("@region_name",f"{None}")
    
from numpy import array
try : foc = ev.preferred_focal_mechanism()
except : foc = ev.focal_mechanisms[0]
finally :
    mom = array(list(foc.moment_tensor.tensor.values())[::2])*1e7
    for st,val in zip(["@mrr","@mtt","@mpp","@mrt","@mrp","@mtp"],mom):
        temp = temp.replace(st,f"{val:+.6e}")
        
try : mag = ev.preferred_magnitude()
except : mag = ev.magnitudes[0]
finally :
    temp = temp.replace("@mb",f"{mag.mag:3.1f}")
        
with open("C:/Users/solen/Documents/Ener/project/test.cmt","w") as file :
    file.write(temp)