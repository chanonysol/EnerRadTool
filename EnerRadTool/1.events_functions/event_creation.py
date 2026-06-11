# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:50:28 2024

@author: solen

Description :
    Make a folder for the event, with a cmt file, data folder
    and automatic personalized parameter file
"""
import json
with open("exresp.json","r") as file : test = json.load(file)



def moment_r(M,exponent = None):
    M = deepcopy(M)
    if exponent :
        M*=float(f"1e{exponent}")
    Mrr,Mtt,Mpp,Mrt,Mrp,Mtp = M
    M = array([[Mrr,Mrt,Mrp],[Mrt,Mtt,Mtp],[Mrp,Mtp,Mpp]])
    return norm(M)*2**-0.5

def mw_r(mag):
    M0 = 10**(3/2*(mag+10.7))
    exp = round(log10(10**(3/2*(8.1+10.7))),0)
    M /= 10.**exp
    exp -=7
    return

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
self.lat = evla
self.lon = evlo
self.depth = evdp
self.mag = test["mag"]
self.moment = M
# self.moment = dict(mrr=M[0], mtt=M[1], mff=M[2],
#                    mrt=M[3], mrf=M[4], mtf=M[5])
self.exponent = expon
self.name = None
self.id = test["evid"]
self.date = UTCDateTime(test['origin_time'])
self.time_shift = None
self.trhalf = None

self.plan1, self.plan2 = self.get_planes(M,expon)

self.vector_t, self.vector_n, self.vector_p = self.get_vector(M,expon)