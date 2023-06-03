#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 17:11:56 2023

@author: as822
"""

# =============================================================================
# import packages
# =============================================================================
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pingouin as pg
import sys
# also have openpyxl

# =============================================================================
# Load the data
# =============================================================================
# raw data
df = pd.read_excel('FLNSUS2022Combo_Filtered(04-Combo).xlsx');



# get just the combo data
df_combo=df.loc[df['POST:Present?'],:];

# =============================================================================
# Figure 1, make a map
# =============================================================================
fig, ax=plt.subplots(figsize=(16,6),ncols=1,nrows=1,)

m = Basemap(projection='robin',lon_0=0,resolution='c')
m.drawcoastlines()
m.fillcontinents(color=[0.8,0.8,0.8],lake_color='aqua')
m.drawmapboundary(fill_color='aqua')

x_,y_=m(df.loc[:,'PRE:ID_Long'],df.loc[:,'PRE:ID_Lat']);
ax.plot(x_,y_,'ok',linewidth=2,label='pre-survey');

x_,y_=m(df_combo.loc[:,'POST:ID_Long'],df_combo.loc[:,'POST:ID_Lat']);
ax.plot(x_,y_,'xr',linewidth=2,label='post-survey',markersize=10);

ax.legend()
fig.savefig('Fig_map_v1.png',dpi=600)

# =============================================================================
# Figure 2, look at the pre/post
# =============================================================================
pre_delim="PRE:Select your level of agreement for the following statements - "
post_delim="POST:Select your level of agreement for the following statements - "

col_pre_NSU=[col for col in df if col.startswith(pre_delim)]
col_post_NSU=[col for col in df if col.startswith(post_delim)]
# Q_name=[split(col)[1] for col in col_post_NSU]
Q_name=[i.split(' - ', 1)[1] for i in col_post_NSU]

df_pre_NSU=df_combo[col_pre_NSU].replace({'Strongly agree': 5, 
                                          'Somewhat agree': 4,
                                          'Neither agree nor disagree': 3,
                                          'Somewhat disagree': 2,
                                          'Strongly disagree': 1,})
df_post_NSU=df_combo[col_post_NSU].replace({'Strongly agree': 5, 
                                          'Somewhat agree': 4,
                                          'Neither agree nor disagree': 3,
                                          'Somewhat disagree': 2,
                                          'Strongly disagree': 1,})

# df_del=df_post_NSU-df_pre_NSU.values

# df_pre_NSU.columns=Q_name;
# df_post_NSU.columns=Q_name;
# df_pre_NSU_m=df_pre_NSU.melt()
# df_pre_NSU_m['time']='pre';
# df_post_NSU_m=df_post_NSU.melt()
# df_post_NSU_m['time']='post';

# df_all=pd.concat([df_pre_NSU_m,df_post_NSU_m])

# sns.boxplot(data=df_all, x="value", y="variable", hue="time");

fig, ax=plt.subplots(figsize=(12,5),ncols=1,nrows=1,);

bonf=1;
# sys.exit()
for idx,col in enumerate(col_pre_NSU):
    stats=pg.wilcoxon(df_pre_NSU.iloc[:,idx],
                df_post_NSU.iloc[:,idx], 
                alternative='two-sided')
    
    #stats['p-val'][0]    
    ax.plot(np.mean(df_pre_NSU.iloc[:,idx]),idx,'xk');
    ax.plot([np.mean(df_pre_NSU.iloc[:,idx]),
                     np.mean(df_post_NSU.iloc[:,idx])],[idx,idx],'-',color='k');
    
    if stats['p-val'][0]<0.001/bonf:
        pcolor='red'
    elif stats['p-val'][0]<0.01/bonf:
        pcolor='orange'
    elif stats['p-val'][0]<0.05/bonf:
        pcolor='green'
    else:
        pcolor='grey'
    
    ax.plot(np.mean(df_post_NSU.iloc[:,idx]),idx,'o',color=pcolor);
    ax.text(5.1,idx,"{0:.3f}".format(stats['p-val'][0]),
            verticalalignment='center',color=pcolor)

ax.set_yticks(np.arange(0,len(Q_name)));
ax.set_yticklabels(Q_name);
ax.set_xticks(np.arange(1,6));
ax.set_xticklabels(['Strongly\ndisagree','Somewhat\ndisagree',
                    'Neither agree\nnor disagree','Somewhat\nagree',
                    'Strongly\nagree'])    
ax.grid(axis = 'x',linewidth=0.5)
ax.grid(axis = 'y',linewidth=0.5)        

plt.tight_layout()
fig.savefig('Fig_Wilcoxon_v1.png',dpi=600)
 
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)

    # fig, ax=plt.subplots(figsize=(4,8),ncols=1,nrows=1,);
    
    # ax.plot([df_pre_NSU.iloc[:,idx].values,
             # df_post_NSU.iloc[:,idx].values],'-k')

# fig, ax=plt.subplots(figsize=(8,4),ncols=1,nrows=1,)
