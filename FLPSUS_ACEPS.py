#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 17:11:56 2023

@author: as822
    - conda activate flxsus

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
import sys, os
import nltk
import tqdm
import geopandas
from shapely.geometry import Point
import flxsus_module as flxmod
# also have openpyxl
# sys.exit()
# =============================================================================
# Set init parameters and organize graphics
# =============================================================================
savefig=True
deprecate = False

plt.rcParams['font.size'] = '12'
plt.rcParams['font.family'] = 'serif'

# colorblind tool
# https://davidmathlogic.com/colorblind/#%23D81B60-%231E88E5-%23FFC107-%23004D40
palette_wong = ["#000000",
                "#E69F00",
                "#56B4E9",
                "#009E73",
                "#0072B2",
                "#D55E00",
                "#CC79A7"];

# =============================================================================
# Load the data
# =============================================================================
homedir=os.getcwd()
datadir='/Users/as822/Downloads/FLPSUS_ACEPS/';

os.chdir(datadir)

pre23=pd.read_excel("FLPSUS2023Pre.xlsx")
post23=pd.read_excel("FLPSUS2023Post.xlsx")

dflist=[pre23,post23,];
dfname=['presurvey 2023',
        'postsurvey 2023',];

os.chdir(homedir)
# sys.exit()
# =============================================================================
# map figure with 2021 v.s. 2022
# =============================================================================
fig,ax=flxmod.map2yrs_1panel(df1 = pre23,
                  df2 = post23, 
                  label1 = 'FLPSUS 2023 pre', 
                  label2 = 'FLPSUS 2023 post', 
                  figsize = (14,7),
                  markersize = 60,
                  marker1='o',
                  marker2='x',
                  alpha1 = 0.8,
                  alpha2 = 0.5,
                  linewidth1=1.5,
                  linewidth2=1.5,
                  color1=palette_wong[6],
                  color2=palette_wong[4],
                  facecolor1='none',
                  facecolor2=palette_wong[4],)
# sys.exit()
# fig,ax=flxmod.map2yrs_panels(df1 = pre21,
#                   df2 = pre22, 
#                   label1 = 'FLNSUS 2021', 
#                   label2 = 'FLNSUS 2022', 
#                   figsize = (14,7),
#                   markersize = 60,
#                   marker1='o',
#                   marker2='x',
#                   alpha1 = 0.8,
#                   alpha2 = 0.5,
#                   linewidth1=1.5,
#                   linewidth2=1.5,
#                   color1=palette_wong[6],
#                   color2=palette_wong[4],
#                   facecolor1='none',
#                   facecolor2=palette_wong[4],)

os.chdir(datadir)
if savefig: fig.savefig('Figures/F1_map_FLPSUS2023.jpeg',dpi=600);
os.chdir(homedir)

# =============================================================================
# Pre-Post 2023
# =============================================================================
col_perceptions=["Select your level of agreement for the following statements - I can become a plastic surgeon",
"Select your level of agreement for the following statements - I have the ability to shadow plastic surgery procedures",
"Select your level of agreement for the following statements - I am familiar with the career pathway to become a plastic surgeon",
"Select your level of agreement for the following statements - I have the institutional support and resources to become a plastic surgeon",
"Select your level of agreement for the following statements - I am connected to mentors that can help me become a plastic surgeon",
"Select your level of agreement for the following statements - I know the day-to-day responsibilities of a plastic surgeon",
"Select your level of agreement for the following statements - I can list at least three subspecialties of plastic surgery",
"Select your level of agreement for the following statements - Plastic Surgery is a good field for minorities and women",
"Select your level of agreement for the following statements - I have seen or met a plastic surgeon before",
"Select your level of agreement for the following statements - I have seen or met a Woman plastic surgeon",
"Select your level of agreement for the following statements - I have seen or met a Black plastic surgeon",
"Select your level of agreement for the following statements - I have seen or met a Latinx plastic surgeon",
"Select your level of agreement for the following statements - Plastic Surgeons are intimidating",
"Select your level of agreement for the following statements - Plastic Surgeons have a good work-life balance",
"Select your level of agreement for the following statements - Plastic Surgeons have reasonable work hours",
"Select your level of agreement for the following statements - Plastic Surgeons improve their patients' quality of life",
"Select your level of agreement for the following statements - Plastic Surgeons will always have secure jobs",
"Select your level of agreement for the following statements - Plastic Surgeons are financially stable",]


col_names=[i.split(' - ', 1)[1] for i in col_perceptions]

df_pre=pre23;
df_post=post23;

uid_pre=set(df_pre["Unique ID"]);
uid_post=set(df_post["Unique ID"]);

uid_all=list(uid_pre.intersection(uid_post))
uid_all.sort()

df_pre_uid=df_pre.loc[df_pre['Unique ID'].isin(uid_all),['Unique ID']+col_perceptions];
df_pre_uid=df_pre_uid.set_index(df_pre_uid['Unique ID']).sort_index();
df_post_uid=df_post.loc[df_post['Unique ID'].isin(uid_all),['Unique ID']+col_perceptions];
df_post_uid=df_post_uid.set_index(df_post_uid['Unique ID']).sort_index();

df_pre_uid=df_pre_uid.replace({'Strongly agree': 5, 
                               'Somewhat agree': 4,
                               'Neither agree nor disagree': 3,
                               'Somewhat disagree': 2,
                               'Strongly disagree': 1,})

df_post_uid=df_post_uid.replace({'Strongly agree': 5, 
                               'Somewhat agree': 4,
                               'Neither agree nor disagree': 3,
                               'Somewhat disagree': 2,
                               'Strongly disagree': 1,})

pre, post=df_pre_uid.align(df_post_uid,join="outer",axis=None)

fig, ax=plt.subplots(figsize=(8,5),ncols=1,nrows=1,);
bonf=1;

# ax.set_yticks(np.arange(0,len(col_names)));

for idx,col in enumerate(col_perceptions):
    stats=pg.wilcoxon(pre.loc[:,col],
                      post.loc[:,col], 
                      alternative='two-sided')

    # print(col_names[idx]);
    # print('    p-val = ',stats['p-val'].values[0])
   
    ax.plot(np.mean(pre.loc[:,col]),idx,'xk');
    ax.plot([np.mean(pre.loc[:,col]),
                     np.mean(post.loc[:,col])],[idx,idx],'-',color='k');
    
    if stats['p-val'][0]<0.001/bonf:
        pcolor='red'
    elif stats['p-val'][0]<0.01/bonf:
        pcolor='#ff781f'
    elif stats['p-val'][0]<0.05/bonf:
        pcolor='green'
    else:
        pcolor='grey'
    
    ax.plot(np.mean(post.loc[:,col]),idx,'o',color=pcolor);
    
    
    ax.text(5.1,idx,"{0:.3f}".format(stats['p-val'][0]),
            verticalalignment='center',color=pcolor)
    
    if stats['p-val'][0]<0.05:
        ax.text(0.9,idx,col_names[idx],
                verticalalignment='center',horizontalalignment='right',color='black',fontweight='bold');
    else:
        ax.text(0.9,idx,col_names[idx],
                verticalalignment='center',horizontalalignment='right',color='black');


    # future work to do posthoc test in significant ones
    # if stats['p-val'][0]<0.05:
    #     tempdf=pd.concat([pre.loc[:,col],post.loc[:,col],])
    #     pg.pairwise_tests(dv='Scores', within='Time', subject='Subject',
    #                    data=df, parametric=False).round(3)


ax.set_yticks(np.arange(0,len(col_names)),labels=[''] * len(col_names));
# ax.set_yticklabels(col_names,fontweight='bold');
ax.set_xticks(np.arange(1,6));
ax.set_xticklabels(['Strongly\ndisagree','Somewhat\ndisagree',
                    'Neither agree\nnor disagree','Somewhat\nagree',
                    'Strongly\nagree'])    
ax.grid(axis = 'x',linewidth=0.5)
ax.grid(axis = 'y',linewidth=0.5)        

# ax.set_yticklabels

# now set specific ylabels as bold based on change from 2021
# labs=ax.get_yticklabels(0);
# import matplotlib.text as txt
# labs[15]=txt.Text(0, 15, 'Neurosurgeons have reasonable work hours',fontweight='bold')
# labs[14]=txt.Text(0, 14, 'Neurosurgeons have a good work-life balance',fontweight='bold')

ax.set_title('FLPSUS 2023 Pre/Post Data')


# plt.tight_layout()
# 
# sys.exit()

os.chdir(datadir)
if savefig: fig.savefig('Figures/Fig_Wilcoxon_2023_bolded.jpeg',dpi=600,bbox_inches='tight');
os.chdir(homedir)

# =============================================================================
# temp
# =============================================================================

col_temp = ["Select your level of agreement for the following statements - I have seen or met a plastic surgeon before",
"Select your level of agreement for the following statements - I have seen or met a Woman plastic surgeon",
"Select your level of agreement for the following statements - I have seen or met a Black plastic surgeon",
"Select your level of agreement for the following statements - I have seen or met a Latinx plastic surgeon",]

col_names=[i.split(' - ', 1)[1] for i in col_temp]

df_heatmap_pre = pd.DataFrame(data=None,index=col_names,columns=['Strongly disagree',
                                                         'Somewhat disagree',
                                                         'Neither agree nor disagree',
                                                         'Somewhat agree',
                                                         'Strongly agree'])

for r,col in enumerate(col_temp):
    for s, rate in enumerate(df_heatmap_pre.columns):
        df_heatmap_pre.loc[col_names[r],rate]=float((pre[col]==s+1).sum());
        
df_heatmap_pre=df_heatmap_pre/len(pre)*100

df_heatmap_post = pd.DataFrame(data=None,index=col_names,columns=['Strongly disagree',
                                                         'Somewhat disagree',
                                                         'Neither agree nor disagree',
                                                         'Somewhat agree',
                                                         'Strongly agree'])

for r,col in enumerate(col_temp):
    for s, rate in enumerate(df_heatmap_post.columns):
        df_heatmap_post.loc[col_names[r],rate]=float((post[col]==s+1).sum());
        
df_heatmap_post=df_heatmap_post/len(post)*100        


fig,ax=plt.subplots(nrows=2,ncols=1,sharex=True,)


s=sns.heatmap(df_heatmap_pre.astype('float64'),annot=True,fmt = '.0f',ax=ax[0],vmin=0,vmax=100)
for t in s.texts: t.set_text(t.get_text() + " %")


s=sns.heatmap(df_heatmap_post.astype('float64'),annot=True,fmt = '.0f',ax=ax[1],vmin=0,vmax=100)
for t in s.texts: t.set_text(t.get_text() + " %")
