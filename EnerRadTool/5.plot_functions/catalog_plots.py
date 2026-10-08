# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:50:28 2024

@author: solen

Description :
    Treats catalogue results to compile figures and analysis
"""
import os
import sys

FILE_PATH = os.path.dirname(os.path.realpath(__file__)).replace('\\', '/')

DIR=f"{FILE_PATH}/../1.events_functions"
sys.path.insert(0, DIR)
DIR=f"{FILE_PATH}/../2.data_functions"
sys.path.insert(0, DIR)

from data_utils import verif_size, find_min_dist, name_reg_event

from numpy import loadtxt, unique, argsort,pi, dtype, nanmean, argwhere,\
    delete, nanmin, nanmax, nanstd, where, cos, sin, log, sqrt,log10,\
        isnan, arange


def mw(m):
    return (2/3*log10(m)-6.07).round(1)
def me(m):
    return (2/3*log10(m)-2.9).round(1)

def load_cat(catpath):
    file = f'../../{catpath}/all_results_events.dat'
    datatype=open(file,"r").readlines()[0][1:].split()
    vals = ["f8"]*len(datatype)
    # vals[0] = "U25"; vals[11]="U25"
    vals[0] = "U25"; vals[3]="U25";  vals[4]="U25"
    dty = dtype([(d,v) for d,v in zip (datatype, vals)])
    mat = loadtxt(file,comments="#",dtype=dty)

    #---contribs
    file = f'../../{catpath}/all_results_contribs.dat'
    datatype=open(file,"r").readlines()[0][1:].split()
    vals = ["f8"]*len(datatype)
    # vals[0] = "U25"; vals[11]="U25"
    vals[0] = "U25"; vals[3]="U25";  vals[4]="U25"; vals[9]="U25"
    dty = dtype([(d,v) for d,v in zip (datatype, vals)])
    mat_contribs = loadtxt(file,comments="#",dtype=dty)


    #s results
    file = f'../../{catpath}/all_swaves_eventsv4.dat'
    datatype=open(file,"r").readlines()[0][1:].split()
    vals = ["f8"]*len(datatype)
    # vals[0] = "U25"; vals[11]="U25"
    vals[0] = "U25"; vals[4]="U25";  vals[5]="U25"
    dty = dtype([(d,v) for d,v in zip (datatype, vals)])
    mat_s = loadtxt(file,comments="#",dtype=dty)

    for i in argwhere(mat_s["moment"]>1e22)[:,0]:
        # print(mat[i])
        mat_s[i]["mw"]=round(2/3*log10(mat_s[i]["moment"])-10.7,1)
        mat_s[i]["moment"]*=1e-7
        mat_s[i]["theta"]=log10(mat_s[i]["Ebb"]/mat_s[i]["moment"])
        mat_s[i]["theta_s"]=log10(mat_s[i]["Es"]/mat_s[i]["moment"])

    file = f'../../{catpath}/all_swaves_events.dat'
    datatype=open(file,"r").readlines()[0][1:].split()
    vals = ["f8"]*len(datatype)
    # vals[0] = "U25"; vals[11]="U25"
    vals[0] = "U25"; vals[4]="U25";  vals[5]="U25"
    dty = dtype([(d,v) for d,v in zip (datatype, vals)])
    mat_s3 = loadtxt(file,comments="#",dtype=dty)

    for i in argwhere(mat_s3["moment"]>1e22)[:,0]:
        # print(mat[i])
        mat_s3[i]["mw"]=round(2/3*log10(mat_s3[i]["moment"])-10.7,1)
        mat_s3[i]["moment"]*=1e-7
        mat_s3[i]["theta"]=log10(mat_s3[i]["Ebb"]/mat_s3[i]["moment"])
        mat_s3[i]["theta_s"]=log10(mat_s3[i]["Es"]/mat_s3[i]["moment"])


    #s waves stations    
    file = f'../../{catpath}/all_swaves_stationsv4.dat'
    datatype=open(file,"r").readlines()[0][1:].split()
    vals = ["f8"]*len(datatype)
    # vals[0] = "U25"; vals[11]="U25"
    vals[0] = "U25"; vals[4]="U25";  vals[5]="U25";  vals[11]="U25";  vals[16]="U25"
    dty = dtype([(d,v) for d,v in zip (datatype, vals)])
    mat_s_sta = loadtxt(file,comments="#",dtype=dty)

    for i in argwhere(mat_s_sta["moment"]>1e22)[:,0]:
        # print(mat[i])
        mat_s_sta[i]["mw"]=round(2/3*log10(mat_s_sta[i]["moment"])-10.7,1)
        mat_s_sta[i]["moment"]*=1e-7
        mat_s_sta[i]["theta_s"]=log10(mat_s_sta[i]["Es"]/mat_s_sta[i]["moment"])

    #psmeca
    dty = dtype([("lon","f8"),("lat","f8"),("depth","f8"),
                  ("mrr","f8"),("mtt","f8"),("mpp","f8"),
                  ("mrt","f8"),("mrp","f8"),("mtp","f8"),
                  ("exp","f8"),
                  ("X","U25"),("Y","U25"),("ID","U25")])
    matps = loadtxt("all_psmeca_events.dat",dtype=dty)

    #matxy ternary
    dty =  dtype([("X","f8"),("Y","f8"),("mag","f8"),("val","f8"),("ID","U25"),("R","U25")])
    matxy = loadtxt("all_XY_events.dat",dtype=dty)
    matxy["ID"]=matps["ID"]


    for i in argwhere(mat["moment"]>1e22)[:,0]:
        # print(mat[i])
        mat[i]["mw"]=round(2/3*log10(mat[i]["moment"])-10.7,1)
        mat[i]["moment"]*=1e-7
        mat[i]["theta"]=log10(mat[i]["Ebb"]/mat[i]["moment"])
    for i in argwhere(mat["region"]=="SW.Pacific")[:,0]:
        mat[i]["zone"] = name_reg_event(mat[i]["lat"],mat[i]["lon"],new_zone=True)[1]

    mat["mw"]=mw(mat["moment"])
    matxy["mag"]=mat["mw"]
    return

def select_in_cat(catpath, mag_range = None, date_range = None, s_wave = False):
    #results                 
    

    nb_event = len(mat["Ebb"])
    no_ener = len(argwhere(isnan(mat["Ebb"])))
    print(f"Total & {nb_event:3.0f} & {100-no_ener/nb_event*100:4.1f}\% \\\\")
    print("\hline")
    for m in arange(4.9,8.2,.1).round(1):
        ind = argwhere(mw(mat["moment"])==m)[:,0]
        ind2 = argwhere(isnan(mat["Ebb"][ind]))[:,0]

        print(f"Mw{m:.1f} & {len(ind):3.0f} & {100-len(ind2)/len(ind)*100:4.1f}\% \\\\")

    ind = argwhere(mat["theta"]>=0)
    mat,matps,matxy = [delete(m,ind) for m in [mat,matps,matxy]]
    ind2=argwhere(isnan(mat["Ebb"]))
    mat,matps,matxy = [delete(m,ind2) for m in [mat,matps,matxy]]
    print(f"Remaining : {len(mat):3.0f}")
    tr_teo = ((1.2*1e-8*(mat["moment"]*1e7)**(1/3))+5.8)*2
    tr_teo2 = ((1.2*1e-8*(mat["moment"]*1e7)**(1/3))+2.8)*2
    ratio = 100*mat["Ehf"]/mat["Ebb"]
    inderr = argwhere((((ratio.round(0)>90)|(ratio.round(0)<5))
                      &((mat["tacer"]>tr_teo2)|(mat["tacer"]<-tr_teo2)))
                      |(mat["tacer"]>3*tr_teo+2)
                      |((ratio.round(0)>99)|(ratio.round(0)<1))
                      )[:,0]


    return