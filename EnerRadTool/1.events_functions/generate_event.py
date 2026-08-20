# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 13:55:59 2024

@author: solen
"""
import os
import shutil
import inspect
import subprocess

from numpy import array, floor, abs, log10, concatenate, nan, ceil, argwhere
from numpy.linalg import norm
from copy import deepcopy
from glob import glob

from obspy.core.utcdatetime import UTCDateTime
from obspy.imaging.beachball import MomentTensor, mt2plane, aux_plane, mt2axes,\
    PrincipalAxis, NodalPlane

from colorama import init as colorama_init
from colorama import Fore, Back, Style

from data_utils import name_reg_event

MON = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
       'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

EVENT_PATH = os.path.dirname(os.path.realpath(__file__)).replace('\\', '/')

MECA_DIC =     {'N':' Normal',
    'N-SS':'Normal - Strike-slip',
    'SS-N':'Strike-slip - Normal',
    'SS':'Strike-slip',
    'SS-R':'Strike-slip - Reverse',
    'R-SS':'Reverse - Strike-slip',
    'R':' Reverse'}

def moment(M,exponent = None):
    M = deepcopy(M)
    if exponent :
        M*=float(f"1e{exponent}")
    Mrr,Mtt,Mpp,Mrt,Mrp,Mtp = M
    M = array([[Mrr,Mrt,Mrp],[Mrt,Mtt,Mtp],[Mrp,Mtp,Mpp]])
    return norm(M)*2**-0.5

def mw(M):
    if not M.any!= 0.:
        return 0.
    M0 = moment(M)
    return round(2/3*log10(M0)-10.73,1)

def write_psmeca_ind(path,ev):
    if not f"{path}temp_psmeca.dat" in [p.replace('\\','/') for p in glob(f"{path}/*.dat")]:
        with open(f"{path}temp_psmeca.dat","w") as file:
            file.write(f"{ev.lon} {ev.lat} {ev.depth} "
                       +f"{' '.join(ev.moment.astype(str))} "
                       +f"{ev.exponent+7} X Y {ev.id}")

def write_psmeca_cat(path,ev,filename):
    if not f"{path}/cat_psmeca.dat" in [p.replace('\\','/') for p in glob(f"{path}/*.dat")]:
        mod = "w"
    else :
        mod ="a"
    with open(f"{path}/psmeca_{filename}.dat",mod) as file:
        file.write(f"{ev.lon} {ev.lat} {ev.depth} "
                   +f"{' '.join(ev.moment.astype(str))} "
                   +f"{ev.exponent+7} X Y {ev.id}\n")

class Event:
    def __init__(self, path):
        ev_file, extension = self.find_file(path)
        if extension == "cmt" :
            with open(f"{path}/{ev_file}",'r') as file:
                lines = file.readlines()
            self.read_cmt(lines)
            self.mecanism = self.mechanism_type(path)

    def find_file(self, path):
        extension_list = ['cmt']
        file_list = [file for file in os.listdir(path)]
        ind = argwhere(array([file.split(".")[-1] in extension_list for file in file_list]))[0,0]
        file = file_list[ind]; ext = file.split(".")[-1]
        return file, ext

    def read_cmt(self, lines):
        info = lines[0].split()[1:]
        date = concatenate(((array(info[0:6]).astype(
            float)//1).astype(int), array([int(info[5][3:])*10000])))
        if date[5] == 60:
            date[5] = 0
            date[4] += 1
        date = UTCDateTime(date[0], date[1], date[2],
                           date[3], date[4], date[5], date[6])
        [_, _, _, mb, ms] = array(info[6:11]).astype(float)
        [dt, d2, evla, evlo, evdp] = [
            float(i.strip().split(":")[-1]) for i in lines[2:7]]
        evid = lines[1].split(':')[-1].strip()
        evname = ''
        for i in info[11:]:
            evname += ' '+i
        M = {}
        for i in lines[7:]:
            n, v = i.strip().split(":")
            M[n] = float(v)
        M = [M[i] for i in M.keys()]
        M = array(M)
        mag = mw(M)
        if mag == 0.:
            expon = 0
        else:
            expon = max(floor(log10(abs(M))).astype(int))
            M /= 10.**expon
            M.round(3)
            if expon >21 : expon-=7
        reg,sreg = name_reg_event(evla,evlo,new_zone=True)
        self.lat = evla
        self.lon = evlo
        self.depth = evdp
        self.reg = reg
        self.sreg = sreg
        self.mag = mag
        self.moment = M
        # self.moment = dict(mrr=M[0], mtt=M[1], mff=M[2],
        #                    mrt=M[3], mrf=M[4], mtf=M[5])
        self.exponent = expon
        self.m0 = moment(self.moment,exponent = self.exponent)
        self.name = evname
        self.id = evid
        self.date = date
        self.time_shift = dt
        self.trhalf = d2

        self.plan1, self.plan2 = self.get_planes(M,expon)

        self.vector_t, self.vector_n, self.vector_p = self.get_vector(M,expon)

    def get_planes(self,M,exp):
        [mrr, mtt, mpp, mrt, mpt, mtp]= M
        sec_plan = mt2plane(MomentTensor(mrr, mtt, mpp, mrt, mpt, mtp, exp))
        fst_plan = aux_plane(sec_plan.strike, sec_plan.dip, sec_plan.rake)
        fst_plan = NodalPlane(strike = fst_plan[0] , dip = fst_plan[1], rake = fst_plan[2])
        return fst_plan,sec_plan

    def get_vector(self,M,exp):
        [mrr, mtt, mpp, mrt, mpt, mtp]= M
        t, n, p = mt2axes(MomentTensor(mrr, mtt, mpp, mrt, mpt, mtp, exp))
        return t,n,p

    def mechanism_type(self,path):
        write_psmeca_ind(path, self)
        # write_psmeca_cat(path, self)
        #https://github.com/Jose-Alvarez/FMC
        cmd = f"python {EVENT_PATH}/FMC.py {path}temp_psmeca.dat -o K ".split()
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, universal_newlines=True)
        with open(f"{path}meca.dat", "w") as file:
            file.write(proc.stdout)
        return proc.stdout.strip().split()[-1]
    
    def levels_conf(self):
        MOM = moment(self.moment,self.exponent)
        
        Tr_mom = array([(1.2*1e-8*(MOM*1e7)**(1/3)+5.8)*2,(1.2*1e-8*(MOM*1e7)**(1/3)+2.8)*2,(1.2*1e-8*(MOM*1e7)**(1/3))*2])
        Tr_mom = concatenate(([Tr_mom[0]*3,Tr_mom[0]*2],Tr_mom))
        Tr_mom = ((Tr_mom/Tr_mom[2])/3).round(2)
        
        LEVEL_DIC={ "GREAT":{"val":[0. ,  0/75, 0.  , 0.  , 0.  , 0.         ,   0/360],
                            "color":"aquamarine","alpha":0.3},
                    "GOOD" :{"val":[0.3, 10/75, 0.07, 0.05, 0.05, Tr_mom[-1] ,  45/360],
                            "color":"yellowgreen","alpha":0.3},
                    "OK"   :{"val":[0.5, 25/75, 0.14, 0.1 , 0.1 , Tr_mom[-2] , 105/360],
                            "color":"khaki","alpha":0.4},
                    "LIMIT":{"val":[0.7, 50/75, 0.42, 0.35, 0.35, Tr_mom[-3] , 270/360],
                            "color":"orange","alpha":0.3},
                    "BAD"  :{"val":[0.8 , 60/75, 0.71, 0.6 , 0.6 , Tr_mom[-4] , 315/360],
                            "color":"salmon","alpha":0.25}
                    }
        return MOM,LEVEL_DIC
    
    def arglist(self, col = Style.DIM + Fore.RED,space=0):
        """
        Print all the parameters and their value

        """
        args = [i for i in inspect.getmembers(self) if not i[0].startswith('_')
                and not inspect.ismethod(i[1])]
        if space==0: print(Style.DIM + Back.MAGENTA + "SOURCE PARAMETERS")
        for arg in args :
            if arg[0] in ["plan1","plan2","vector_n","vector_p","vector_t"]:
                val = " - ".join([f"{i[0]} : {i[1]:+.2f}"
                       for i in inspect.getmembers(arg[1]) 
                       if not i[0].startswith('_')
                       and not inspect.ismethod(i[1])])
                print(f"{' '*space}" + col + f"{arg[0]}" + Style.RESET_ALL
                      + f" : {val}")
            else:
                print(f"{' '*space}" + col + f"{arg[0]}" + Style.RESET_ALL
                      + f" : {arg[1]}")

