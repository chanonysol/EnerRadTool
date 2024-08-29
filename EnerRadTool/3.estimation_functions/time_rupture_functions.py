# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:50:28 2024

@author: solen

Description :
    Determine the rupture duuration based on various methods
"""
from numpy import argwhere, mean, array, arange, log10, median
from numpy import ones, vstack
from numpy.linalg import lstsq
from colorama import Back, Fore, Style

def reg_lin(x, y, xtot):
    m, c = lstsq(vstack([x, ones(len(x))]).T, y, rcond=None)[0]
    return m*xtot+c


def find_same_point(x, A, B):
    mat = abs(A-B)
    mat /= mean(mat)
    i = argwhere(mat == min(mat))[0, 0]
    return i

def find_tr_ligns(t, eghf):
    sl_eghf = array([eghf[i+1]-eghf[i]
                    for i in arange(-1, len(eghf)-1, 1)[::-1]])[::-1]
    sl_eghf[0] = 0
    sloperange=((sl_eghf/mean(sl_eghf))).round(0)

    indexmax=argwhere((sloperange>=max((sloperange/1.6)[::3])))[:,0]
    [i1,i2]=indexmax[0],indexmax[-1]

    indexmin=argwhere(sloperange[i2:-10]==sloperange[-1])[:,0]+i2
    i3=indexmin[argwhere(array([indexmin[i]==indexmin[i-1]+1 for i in range(len(indexmin))])==False)[::5,0][-1]]

    ts,te=t[i1:i2+1],t[i3:]
    eghf_s,eghf_e=eghf[i1:i2+1],eghf[i3:]
    start=reg_lin(ts,eghf_s,t)
    end=reg_lin(te,eghf_e,t)

    Tr=t[find_same_point(t,start,end)]
    return Tr, start, end

def find_tr_tacer(t, eghf):
    sl_eghf = array([eghf[i+1]-eghf[i]
                    for i in arange(-1, len(eghf)-1, 1)[::-1]])[::-1]
    sl_eghf[0] = 0
    sloperange = array(
        [sum(sl_eghf[:j])/j for j in arange(1, len(eghf)+1, 1).astype(int)])
    exp = median(list(set((log10(sloperange[2:])-0.5).astype(int))))
    # sloperange=sloperange.round(-int(exp))
    Tr = argwhere(sloperange.round(-int(exp)) ==
                  max(sloperange.round(-int(exp))))[-1, 0]
    return Tr, sloperange


def find_freq_flumax(fre, flux):
    slope = array([flux[i+1]-flux[i]
                   for i in arange(-1, len(flux)-1, 1)[::-1]])[::-1]
    slope[0] = 0
    return fre[argwhere(slope == max(slope))[0, 0]]

def find_tr(method,t,eghf):
    # sys.stdout.write(f"{Fore.RED + Style.DIM}"
    #                   +f" > Rupture time estimation"
    #                   +f"{Style.RESET_ALL}\n")


    tr_tacer, tacer = find_tr_tacer(t, eghf)
    tr_ligns, fst_lign, sec_lign = find_tr_ligns(t, eghf)

    if method=="tacer":
        return (tr_tacer, tacer), (None,None,None)
    if method=="trend":
        return (None,None), (tr_ligns, fst_lign, sec_lign)
    if method=="both":
        return (tr_tacer, tacer), (tr_ligns, fst_lign, sec_lign)
