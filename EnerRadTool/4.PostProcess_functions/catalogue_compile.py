# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 10:20:25 2026

@author: solen

Description :
    Compile catalogue results in table files
"""
# =============================================================================
# LIBRARIES IMPORT
# =============================================================================
import os
import sys

FILE_PATH = os.path.dirname(os.path.realpath(__file__)).replace('\\', '/')

DIR=f"{FILE_PATH}/../1.events_functions"
sys.path.insert(0, DIR)
DIR=f"{FILE_PATH}/../2.data_functions"
sys.path.insert(0, DIR)

from results_writing import read_results, read_results_sta, read_sta_log, AttribDict
from generate_event import Event, moment
from numpy import nan, isnan, nanmean, nanmin, nanmax, nanmedian, dtype, loadtxt,\
 format_float_scientific, concatenate, arange, array, vstack, floor, log10
from glob import glob
from tqdm import tqdm

def write_stats(catpath):
    with open(f"{catpath}/all_stats.dat","w") as file:
        file.write(f"#{'DateTime':19s} {'lat':7s} {'lon':7s} "
                       +f"{'region':12s} {'zone':12s} "
                        +f"{'moment':8s} {'mw':4s} "
                        +f"{'data':8s} {'proc':8s} {'calc':8s} "
                        +f"{'total':8s} \n" )
        for evpath in tqdm([p.split("\\")[-2] for p in concatenate([glob(f"{catpath}/*M*{m:.1f}*/") 
                        for m in arange(5.,8.2,.1).round(1)])
                          if p.find("_")!=-1][::-1]):
            try :
                ev = read_results(glob(f"{catpath}/{evpath}/Results/*Wprf*/results*")[-1]).ev
                times = read_results(glob(f"{catpath}/{evpath}/Results/*Wprf*/results*")[-1]).time
            except:
                ev = Event(f"{catpath}/{evpath}/")
                file.write(f"{str(ev.date)[:-8]:19s} {ev.lat:+7.2f} {ev.lon:+7.2f} "
                               +f"{ev.reg:12s} {ev.sreg:12s} "
                                +f"{ev.m0:8.2e} {ev.mag:4.2f} "
                                +" ".join(array([nan]*4).astype(str))+"\n" )
                continue
            else : 
                file.write(f"{str(ev.date)[:-8]:19s} {ev.lat:+7.2f} {ev.lon:+7.2f} "
                               +f"{ev.reg:12s} {ev.sreg:12s} "
                                +f"{ev.M0:8.2e} {ev.mw:4.2f} "
                                +" ".join(array(list(times.values())).astype(str))+"\n" )

def write_results(catpath):
    with open(f"{catpath}/all_results_events.dat","w") as file:
        file.write(f"#{'DateTime':19s} {'lat':7s} {'lon':7s} "
                       +f"{'region':12s} {'zone':12s} "
                        +f"{'moment':8s} {'mw':4s} "
                        +f"{'Ebb':8s} {'dbb':3s} {'MeBB':3s} "
                        +f"{'Ehf':8s} {'dhf':3s} {'Mehf':3s} "
                        +f"{'Tr':3s} {'lines':3s} {'tacer':3s} "
                        +f"{'theta':4s}\n" )
        for evpath in tqdm([p.split("\\")[-2] for p in concatenate([glob(f"{catpath}/*M*{m:.1f}*/") 
                        for m in arange(5.,8.2,.1).round(1)])
                          if p.find("_")!=-1][::-1]):
            ev = Event(f"{catpath}/{evpath}/")
            try :
                res = read_results(glob(f"{catpath}/{evpath}/Results/*Wprf*/results*")[-1]).corr
            except :
                file.write(f"{str(ev.date)[:-8]:19s} {ev.lat:+7.2f} {ev.lon:+7.2f} "
                               +f"{ev.reg:12s} {ev.sreg:12s} "
                                +f"{moment(ev.moment,ev.exponent):8.2e} {ev.mag:4.2f} "
                                +f"{nan:8.2e} {nan:3.2f} {nan:3.2f} "
                                +f"{nan:8.2e} {nan:3.2f} {nan:3.2f} "
                                +f"{nan: 3.0f} {nan: 3.0f} {nan: 3.0f} "
                                +f"{nan:+4.1f}\n" )
       
            else : 
                file.write(f"{str(ev.date)[:-8]:19s} {ev.lat:+7.2f} {ev.lon:+7.2f} "
                               +f"{ev.reg:12s} {ev.sreg:12s} "
                                +f"{moment(ev.moment,ev.exponent):8.2e} {ev.mag:4.2f} "
                                +f"{res.Ebb:8.2e} {res.dEbb:3.2f} {res.MeBB:3.2f} "
                                +f"{res.Ehf:8.2e} {res.dEhf:3.2f} {res.MeHF:3.2f} "
                                +f"{res.Tr: 3.0f} {res.Tr_lines: 3.0f} {res.Tr_tacer: 3.0f} "
                                +f"{res.theta:+4.1f}\n" )

def write_sta_contrib(catpath):
    with open(f"{catpath}/results_contribs.dat","w") as file:
        file.write(f"#{'DateTime':19s} {'lat':7s} {'lon':7s} "
                       +f"{'region':12s} {'zone':12s} "
                        +f"{'moment':8s} {'mw':4s} "
                        +f"{'Ebb':8s} {'Ehf':8s} {'station':16s} "
                        +f"{'az':5s} {'dist':5s} {'stalon':8s} {'stalat':4s} "
                        +f"{'ebb':8s} {'ehf':8s} "
                        +f"{'rbb':4s} {'rhf':4s} {'rHF':4s} {'rBB':4s} {'Ehf/ebb':4s}\n" )
        # for evpath in tqdm([p.split("\\")[-2] for p in concatenate([glob(f"{catpath}/*M*{m:.1f}*/") 
        #                 for m in arange(5.,8.2,.1).round(1)])
        #                   if p.find("_")!=-1][::-1]):
        for evpath in tqdm([p.split("\\")[-2] for p in concatenate([glob(f"D:/save/CATgcmtNZ/*M*{m:.1f}*/") 
                        for m in arange(5.,8.2,.1).round(1)])
                          if p.find("_")!=-1][::-1]):
            try :
                ev = read_results(glob(f"{catpath}/{evpath}/Results/*Wprf*/results*")[-1]).ev
            except:
                ev = Event(f"{catpath}/{evpath}/")
                file.write(f"{str(ev.date)[:-8]:19s} {ev.lat:+7.2f} {ev.lon:+7.2f} "
                               +f"{ev.reg:12s} {ev.sreg:12s} "
                                +f"{moment(ev.moment,ev.exponent):8.2e} {ev.mag:4.2f} "
                                +f"{nan:8.2e} {nan:8.2e} {'X':16s} "
                                +" ".join(array([nan]*4).astype(str))
                                +f" {nan: 8.2e} {nan: 8.2e} "
                                +" ".join(array([nan]*5).astype(str))+"\n" )
                continue
            try :
                res = read_results(glob(f"{catpath}/{evpath}/Results/*Wprf*/results*")[-1]).corr 
                t,ebb,n = read_results_sta(glob(f"{catpath}/{evpath}/Results/*Wprf*/ener_all_bb*")[-1])
                _,ehf,_ = read_results_sta(glob(f"{catpath}/{evpath}/Results/*Wprf*/ener_all_hf*")[-1])
                ehf=ehf[:,int(res.Tr)];ebb=ebb[:,int(res.Tr)]
                names,stalog=read_sta_log(glob(f"{catpath}/{evpath}/Results/*Wprf*/sta_log.dat")[-1][:-12])
                m=vstack([concatenate(([n],list(i.values())[1:],
                                       [eibb,eihf,eibb/res.Ebb,eihf/res.Ehf,
                                        eihf/eibb,eihf/res.Ebb,res.Ehf/eibb])) 
                          for i,n,eibb,eihf in zip(stalog.values(),names,ebb,ehf)])
            except :
                file.write(f"{str(ev.date)[:-8]:19s} {ev.lat:+7.2f} {ev.lon:+7.2f} "
                               +f"{ev.reg:12s} {ev.sreg:12s} "
                                +f"{ev.M0:8.2e} {ev.mw:4.2f} "
                                +f"{nan:8.2e} {nan:8.2e} {'X':16s} "
                                +" ".join(array([nan]*4).astype(str))
                                +f" {nan: 8.2e} {nan: 8.2e} "
                                +" ".join(array([nan]*5).astype(str))+"\n" )
       
            else : 
                for i in m:
                    file.write(f"{str(ev.date)[:-8]:19s} {ev.lat:+7.2f} {ev.lon:+7.2f} "
                                   +f"{ev.reg:12s} {ev.sreg:12s} "
                                    +f"{ev.M0:8.2e} {ev.mw:4.2f} "
                                    +f"{res.Ebb:8.2e} {res.Ehf:8.2e} {i[0]:16s} "
                                    +" ".join(i[1:5].astype(float).round(2).astype(str))
                                    +f" {float(i[5]): 8.2e} {float(i[6]): 8.2e} "
                                    +" ".join(i[-5:].astype(float).round(2).astype(str))+"\n" )

def writeXYresults(catpath):
    dty = dtype([("X","f8"),("Y","f8"),("mag","f8"),("val","f8"),("ID","U25"),("R","U25")])
   
    with open(f"{catpath}/all_XY_events.dat","w") as file:
        file.write(f"#X_Kaverina Y_Kaverina Magnitude_Mw Val ID rupture_type\n" )
        for evpath in tqdm([p.split("\\")[-2] for p in concatenate([glob(f"{catpath}/*M*{m:.1f}*/") 
                            for m in arange(5.,8.2,.1).round(1)])
                              if p.find("_")!=-1][::-1]):
            #ev = Event(f"{catpath}/{evpath}/")
            try :
               res = loadtxt(glob(f"{catpath}/{evpath}/meca.dat")[-1],dtype = dty,
                             skiprows=1)
            except :
                file.write(f"{res['X']:+9.6f} {res['Y']:+9.6f} "
                               +f"{res['mag']:3.1f} {nan:4.0f} "
                                +f"{res['ID']:10s} {res['R']:4s}\n")
       
            else : 
                if len(res["ID"].item())!=6:
                    a = res["ID"].item()
                    res["ID"] = a[:a.find("+")-1].replace(".","")+"E"+a[a.find("+")+1:]
                file.write(f"{res['X']:+9.6f} {res['Y']:+9.6f} "
                               +f"{res['mag']:3.1f} {9999:4.0f} "
                                +f"{res['ID']:10s} {res['R']:4s}\n")
    # mat["val"] = reshape(array([9999]*len(mat["val"])),(len(mat["val"]),1))
    # savetxt("all_XY_events.dat",mat,fmt = "%+9.6f %+9.6f %3.1f %4.0f %10s %4s")

def writePSmecaresults(catpath):
    dty = dtype([("lon","f8"),("lat","f8"),("depth","f8"),
                 ("mrr","f8"),("mtt","f8"),("mpp","f8"),
                 ("mrt","f8"),("mrp","f8"),("mtp","f8"),
                 ("exp","f8"),
                 ("X","U25"),("Y","U25"),("ID","U25")])
    with open("all_psmeca_events.dat","w") as file:
        for evpath in tqdm([p.split("\\")[-2] for p in concatenate([glob(f"{catpath}/*M*{m:.1f}*/") 
                            for m in arange(5.,8.2,.1).round(1)])
                              if p.find("_")!=-1][::-1]):
            try :
                res = loadtxt(glob(f"{catpath}/{evpath}/temp_psmeca.dat")[-1],dtype = dty)
                
            except :
                ev = Event(f"{catpath}/{evpath}/")
                file.write(f"{ev.lon:+6.1f} {ev.lat:+6.1f} {ev.depth:4.0f} "
                            +" ".join(ev.moment.round(3).astype(str))
                            +f"{ev.moment:2.0f} "
                            +f"{'X':1s} {'Y':1s} {ev.id:10s}\n")
               
       
            else : 
                if res["exp"]==7.:
                    M = loadtxt(glob(f"{catpath}/{evpath}/temp_psmeca.dat")[-1],usecols=arange(3,9,1))
                    expon = max(floor(log10(abs(M))).astype(int))
                    M /= 10.**expon
                    M.round(3)
                    # if expon >21 : expon-=7
                    file.write(f"{res['lon']:+6.1f} {res['lat']:+6.1f} {res['depth']:4.0f} "
                                +f"{M[0]:+6.4f} {M[1]:+6.4f} {M[2]:+6.4f} "
                                +f"{M[3]:+6.4f} {M[4]:+6.4f} {M[5]:+6.4f} "
                                +f"{expon:2.0f} "
                                +f"{res['X']:1s} {res['Y']:1s} {res['ID']:10s}\n")
                    continue
                    
                file.write(f"{res['lon']:+6.1f} {res['lat']:+6.1f} {res['depth']:4.0f} "
                            +f"{res['mrr']:+6.4f} {res['mtt']:+6.4f} {res['mpp']:+6.4f} "
                            +f"{res['mrt']:+6.4f} {res['mrp']:+6.4f} {res['mtp']:+6.4f} "
                            +f"{res['exp']:2.0f} "
                            +f"{res['X']:1s} {res['Y']:1s} {res['ID']:10s}\n")
                
    # mat["val"] = reshape(array([9999]*len(mat["val"])),(len(mat["val"]),1))
    # savetxt("all_XY_events.dat",mat,fmt = "%+9.6f %+9.6f %3.1f %4.0f %10s %4s")

def write_swaves_files(catpath):
    with open(f"{catpath}/all_swaves_events.dat","w") as file:
        file.write(f"#{'DateTime':19s} {'lat':7s} {'lon':7s} {'depth':5s} "
                       +f"{'region':12s} {'zone':12s} "
                        +f"{'moment':8s} {'mw':4s} "
                        +f"{'Ebb':8s} {'dbb':3s} {'MeBB':3s} "
                        +f"{'Ehf':8s} {'dhf':3s} {'Mehf':3s} "
                        +f"{'Tr':3s} {'theta':4s} "
                        +f"{'Es':8s} {'MeS':3s} {'theta_s':4s}\n" )
        for evpath in tqdm([p.split("\\")[-2] for p in concatenate([glob(f"{catpath}/*M*{m:.1f}*/") 
                            for m in arange(5.,8.2,.1).round(1)])
                              if p.find("_")!=-1][::-1]):
            if len(glob(f'{catpath}/{evpath.replace("\\","/")}/Data/CloseRecords/*.MSEED'.replace(CODE_PATH,"")))==0:
                continue
            ev = Event(f"{catpath}/{evpath}/")
            try :
                res = read_results(glob(f"{catpath}/{evpath}/Results/*Wp*s*/results*")[-1]).corr
            except :
                res = AttribDict({"Ebb":nan,"dEbb":nan,"MeBB":nan,
                                  "Ehf":nan,"dEhf":nan,"MeHF":nan,
                                  "Tr":nan,"theta":nan})  
            try:
                res_s = read_results(glob(f"{catpath}/{evpath}/Results/*Ws_Qtest*/results*")[-1]).raw_s
            except :
                res_s = AttribDict({"Ebb":nan,"MeBB":nan,"theta":nan})  
            file.write(f"{str(ev.date)[:-8]:19s} {ev.lat:+7.2f} {ev.lon:+7.2f} {ev.depth:5.2f} "
                           +f"{ev.reg:12s} {ev.sreg:12s} "
                            +f"{moment(ev.moment,ev.exponent):8.2e} {ev.mag:4.2f} "
                            +f"{res.Ebb:8.2e} {res.dEbb:4.1f} {res.MeBB:3.2f} "
                            +f"{res.Ehf:8.2e} {res.dEhf:4.1f} {res.MeHF:3.2f} "
                            +f"{res.Tr: 3.0f} {res.theta:+4.1f} "
                            +f"{res_s.Ebb:8.2e} {res_s.MeBB:3.2f} {res_s.theta:+4.1f}\n" )

def write_swaves_sta(catpath):
    from numpy import shape
    with open(f"{catpath}/all_swaves_stations.dat","w") as file_w:
        file_w.write(f"#{'DateTime':19s} {'lat':7s} {'lon':7s} {'depth':5s} "
                       +f"{'region':12s} {'zone':12s} "
                        +f"{'moment':8s} {'mw':4s} "
                        +f"{'Es':8s} {'MeS':3s} {'theta_s':4s} " 
                        +f"{'station':16s}{'sta_lon':8s} {'sta_lat':8s} {'dist':4s} " 
                        +f"{'az':5s} {'site':1s} {'E_sta':8s}\n" )
        for evpath in tqdm([p.split("\\")[-2] for p in concatenate([glob(f"{catpath}/*M*{m:.1f}*/") 
                            for m in arange(5.,8.2,.1).round(1)])
                              if p.find("_")!=-1][::-1]):
            if len(glob(f'{catpath}/{evpath.replace("\\","/")}/Data/CloseRecords/*.MSEED'.replace(CODE_PATH,"")))==0:
                continue
            ev = Event(f"{catpath}/{evpath}/") 
            try:
                res_s = read_results(glob(f"{catpath}/{evpath}/Results/*Ws_Qtest*/results*")[-1]).raw_s
                file = glob(f"{catpath}/{evpath}/Results/*Ws_Qtest*/station_ener.dat")[-1]
                datatype=open(file,"r").readlines()[0][1:].split()
                vals = ["f8"]*len(datatype)
                vals[0] = "U25";vals[5] = "U25"
                dty = dtype([(d,v) for d,v in zip (datatype, vals)])
                mat = loadtxt(file,comments="#",dtype=dty)
            except :
                res_s = AttribDict({"Ebb":nan,"MeBB":nan,"theta":nan})  
                sta,lo,la,dist,az,site,E = "None",nan,nan,nan,nan,"X",nan
                file_w.write(f"{str(ev.date)[:-8]:19s} {ev.lat:+7.2f} {ev.lon:+7.2f} {ev.depth:5.2f} "
                               +f"{ev.reg:12s} {ev.sreg:12s} "
                                +f"{moment(ev.moment,ev.exponent):8.2e} {ev.mag:4.2f} "
                                +f"{res_s.Ebb:8.2e} {res_s.MeBB:3.2f} {res_s.theta:+4.1f} "
                                +f"{sta:16s} {lo:+7.2f} {la:+7.2f} {dist:4.2f} "
                                +f"{az:5.1f} {site:1s} {E:+7.2f}\n")
            if shape(mat)==():
                sta,lo,la,dist,az,site,E = mat.item()
                file_w.write(f"{str(ev.date)[:-8]:19s} {ev.lat:+7.2f} {ev.lon:+7.2f} {ev.depth:5.2f} "
                               +f"{ev.reg:12s} {ev.sreg:12s} "
                                +f"{moment(ev.moment,ev.exponent):8.2e} {ev.mag:4.2f} "
                                +f"{res_s.Ebb:8.2e} {res_s.MeBB:3.2f} {res_s.theta:+4.1f} "
                                +f"{sta:16s} {lo:+7.2f} {la:+7.2f} {dist:4.2f} "
                                +f"{az:5.1f} {site:1s} {E:+7.2f}\n")
            else :
                for sta,lo,la,dist,az,site,E in mat:
                    file_w.write(f"{str(ev.date)[:-8]:19s} {ev.lat:+7.2f} {ev.lon:+7.2f} {ev.depth:5.2f} "
                                   +f"{ev.reg:12s} {ev.sreg:12s} "
                                    +f"{moment(ev.moment,ev.exponent):8.2e} {ev.mag:4.2f} "
                                    +f"{res_s.Ebb:8.2e} {res_s.MeBB:3.2f} {res_s.theta:+4.1f} "
                                    +f"{sta:16s} {lo:+7.2f} {la:+7.2f} {dist:4.2f} "
                                    +f"{az:5.1f} {site:1s} {E:+7.2f}\n")

            