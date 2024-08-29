# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:50:28 2024

@author: solen

Description :
    Based on a list of events, create all needed events to constitute a
    catalog
"""

# colormap angular distance
cmap_dist = ListedColormap(colormaps["tab20c_r"](linspace(0., 1., 80)))
col_dist = cmap_dist.colors

fig,[ax,ax2] = plt.subplots(2,1,figsize = (17,10))
for tr in general_st_bb:
    t=linspace(-60, parameters.time_max, tr.stats.npts, endpoint=True)
    ax.plot(t, tr.data * 1e3, color = col_dist[int(tr.stats.distance)],lw = 0.5)

for tr in  general_st_hf:
    t=linspace(-60, parameters.time_max, tr.stats.npts, endpoint=True)
    ax2.plot(t, tr.data * 1e3, color = col_dist[int(tr.stats.distance)],lw = 0.5)

for axi in [ax,ax2]:
    axi.grid()
    axi.set_xlim(-60,parameters.time_max)
fig.colorbar(cm.ScalarMappable(norm=Normalize(0,80), cmap=cmap_dist),
              ax=[ax,ax2],ticks=arange(0,81,8))
fig.show()



with Pool(14) as p:
    argums = [(tr,parameters.time_max) for tr in general_st_bb]
    enbb = vstack([result for result in tqdm(p.imap(trace_ener,argums),total=len(argums))])
    argums = [(tr,parameters.time_max) for tr in general_st_hf]
    enhf = vstack([result for result in tqdm(p.imap(trace_ener,argums),total=len(argums))])
    #vstack()

e_bb = mean(enbb,axis=0)
e_hf = mean(enhf,axis=0)
t = arange(-30,parameters.time_max)

e_bb -= (e_bb[30]-e_bb[0])/30
e_hf -= (e_hf[30]-e_hf[0])/30

t = arange(0,parameters.time_max)
e_bb,e_hf=e_bb[30:],e_hf[30:]

ts,ens = smooth_curve_mean(deepcopy(t), e_bb)
tshf,enshf = smooth_curve_mean(deepcopy(t), e_hf)

fig,[ax,ax2] = plt.subplots(2,1,figsize = (8,10))
ax1 = ax.twinx()
ax3 = ax2.twinx()
# for e in enbb:
#     ax.plot(t,e,color="silver")
# for e in enhf:
#     ax2.plot(t,e,color="silver")
for i in range(len(enbb)):
    ti,eni = smooth_curve_mean(arange(-30,parameters.time_max), enbb[i,:])
    ax1.plot(ti,eni/max(eni),color='silver',alpha = 0.1)
ax.plot(t,e_bb,":", color='purple',zorder = 10)
ax.plot(ts,ens, color="darkmagenta")
test=[]
test2=[]
for i in range(len(enhf)):
    ti,eni = smooth_curve_mean(arange(-30,parameters.time_max), enhf[i,:])
    ti,eni = arange(-30,parameters.time_max), enhf[i,:]
    ax3.plot(ti,eni/max(eni),color='silver',alpha = 0.1)
    tr = find_tr_tacer(ti, eni)[0]
    test.append(tr)
    test2.append(eni[tr])
ax2.plot(t,e_hf,":r")
ax2.plot(tshf,enshf, color="r")
test = array(test)
Tr = find_tr_tacer(tshf,enshf)[0]
for axi in [ax,ax2]:
    axi.grid()
    axi.set_xlim(0,parameters.time_max)
    for ti in test:
        axi.axvline(ti-30,c='yellow')
    axi.axvline(mean(test)-30,c='k')
    axi.axvline(Tr,c='k')