#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 17:11:56 2023

@author: as822
    - conda activate flxsus
    - compare experiences between the three specialties - Antoinette Charles
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
# import flxsus_module as flxmod
from scipy.stats import chisquare
# also have openpyxl
# sys.exit()
# =============================================================================
# Set init parameters and organize graphics
# =============================================================================
savefig=True
deprecate = False

plt.rcParams['font.size'] = '12'
plt.rcParams['font.family'] = 'sans-serif'

# colorblind tool
# https://davidmathlogic.com/colorblind/#%23D81B60-%231E88E5-%23FFC107-%23004D40
palette_flxsus = ["#59388bff", # NSU
                  "#800020ff", # Plastics
                  "#2d6d77ff",# Ortho
                  ]

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
datadir='/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/Database';

os.chdir(datadir)

pre23=pd.read_excel("DATA_FLXSUS_pre_2023.xlsx")
post23_FLOSUS=pd.read_excel("DATA_FLOSUS_post_2023.xlsx")
post23_FLPSUS=pd.read_excel("DATA_FLPSUS_post_2023.xlsx")
post23_FLNSUS=pd.read_excel("DATA_FLNSUS_post_2023.xlsx")
idfile=pd.read_excel("IDFile.xlsx")

os.chdir(homedir)


# =============================================================================
# Make Wrap Text
# =============================================================================
import textwrap
def wrap_labels_y(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_yticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                      break_long_words=break_long_words))
    ax.set_yticklabels(labels, rotation=0)
    
def wrap_labels_x(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                      break_long_words=break_long_words))
    ax.set_xticklabels(labels, rotation=0)

# =============================================================================
# Plot different 2023 specialty data on the map (postsurvey)
# =============================================================================

# load in the world map
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))    

# init the figure    
fig, axs=plt.subplots(figsize=(10,5),ncols=1,nrows=1,)

set_excel=['DATA_FLNSUS_post_2023.xlsx',
           'DATA_FLOSUS_post_2023.xlsx',
           'DATA_FLPSUS_post_2023.xlsx']
# set_df=[post23_FLNSUS,post23_FLOSUS,post23_FLPSUS]
set_names=['Neurosurgery','Orthopaedic Surgery','Plastic Surgery']


UID_post=list()


ax=axs
# create a world plot on that axis
world.plot(ax=ax,color='#CCCCCC',zorder=-1000)

# plot boundaries between countries
world.boundary.plot(color=[0.5,0.5,0.5],linewidth=0.5,ax=ax,zorder=-900)
# set figure metadata

markerset=[4,5,6]

for i, excel_name in enumerate(set_excel):
    ax=axs
    

    df1=idfile.loc[idfile[excel_name]==True,['Unique ID','Latitude','Longitude']]

    UID_post.append(df1['Unique ID'])

    ax.scatter(df1.loc[:,'Longitude'],
                df1.loc[:,'Latitude'],
                s=20,
                marker=markerset[i],
                facecolors=palette_flxsus[i],
                edgecolors=palette_flxsus[i],
                linewidths=0,
                alpha=0.7,
                label=set_names[i],)
    # sys.exit()
    ax.set_axis_off()
    # ax.set_title(set_names[i])
ax.legend(loc=8,ncol=3)
# plt.suptitle('Post-Survey Map',fontsize=20)
plt.tight_layout()

os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
# if savefig: fig.savefig('AMEC_FLXSUS_virtual/post_map_2023_oneplot.jpeg',dpi=300);
os.chdir(homedir)



# =============================================================================
# Plot different 2023 specialty data on the map (postsurvey)
# =============================================================================

# load in the world map
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))    

# init the figure    
fig, axs=plt.subplots(figsize=(10,10),ncols=1,nrows=3,)

set_excel=['DATA_FLNSUS_post_2023.xlsx',
           'DATA_FLOSUS_post_2023.xlsx',
           'DATA_FLPSUS_post_2023.xlsx']
# set_df=[post23_FLNSUS,post23_FLOSUS,post23_FLPSUS]
set_names=['Neurosurgery','Orthopaedic Surgery','Plastic Surgery']

UID_post=list()

for i, excel_name in enumerate(set_excel):
    ax=axs[i]
    
    # create a world plot on that axis
    world.plot(ax=ax,color='#CCCCCC',)
    
    # plot boundaries between countries
    world.boundary.plot(color=[0.5,0.5,0.5],linewidth=0.5,ax=ax,)
    # set figure metadata

    df1=idfile.loc[idfile[excel_name]==True,['Unique ID','Latitude','Longitude']]

    UID_post.append(df1['Unique ID'])

    ax.scatter(df1.loc[:,'Longitude'],
                df1.loc[:,'Latitude'],
                s=10,
                marker='o',
                facecolors=palette_flxsus[i],
                edgecolors=palette_flxsus[i],
                linewidths=1,
                alpha=0.8,
                label=set_names[i])

    ax.set_axis_off()
    ax.set_title(set_names[i])
plt.suptitle('Post-Survey Map',fontsize=20)
plt.tight_layout()

os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
# if savefig: fig.savefig('AMEC_FLXSUS_virtual/post_map_2023_newcolor.jpeg',dpi=300);
os.chdir(homedir)


# =============================================================================
# Pre/Post Perceptions
# =============================================================================
# pre23=pd.read_excel("DATA_FLXSUS_pre_2023.xlsx")
# post23_FLOSUS=pd.read_excel("DATA_FLOSUS_post_2023.xlsx")
# post23_FLPSUS=pd.read_excel("DATA_FLPSUS_post_2023.xlsx")
# post23_FLNSUS=pd.read_excel("DATA_FLNSUS_post_2023.xlsx")

col_general=['Select your level of agreement for the following statements - I will get into medical school',
             'Select your level of agreement for the following statements - I will become a doctor',
             'Select your level of agreement for the following statements - I have the ability to shadow physicians',
             'Select your level of agreement for the following statements - I am familiar with the career pathway to become a doctor',
             'Select your level of agreement for the following statements - I have the institutional support and resources to become a doctor',
             'Select your level of agreement for the following statements - I am connected to mentors that can help me become a doctor',
             'Select your level of agreement for the following statements - Medicine is a good field for minorities and women',
             'Select your level of agreement for the following statements - I have seen or met a Woman doctor',
             'Select your level of agreement for the following statements - I have seen or met a Black doctor',
             'Select your level of agreement for the following statements - I have seen or met a Latinx doctor',
             'Select your level of agreement for the following statements - Doctors have a good work-life balance',
             'Select your level of agreement for the following statements - Doctors have reasonable work hours',
             "Select your level of agreement for the following statements - Doctors improve their patients' quality of life",
             'Select your level of agreement for the following statements - Doctors will always have secure jobs',
             'Select your level of agreement for the following statements - Doctors are financially stable',]

col_neuro=['Select your level of agreement for the following statements - I can become a neurosurgeon',
             'Select your level of agreement for the following statements - I have the ability to shadow neurosurgical procedures',
             'Select your level of agreement for the following statements - I am familiar with the career pathway to become a neurosurgeon',
             'Select your level of agreement for the following statements - I have the institutional support and resources to become a neurosurgeon',
             'Select your level of agreement for the following statements - I am connected to mentors that can help me become a neurosurgeon',
             'Select your level of agreement for the following statements - I know the day-to-day responsibilities of a neurosurgeon',
             'Select your level of agreement for the following statements - I can list at least three subspecialties of neurosurgery',
             'Select your level of agreement for the following statements - Neurosurgery is a good field for minorities and women',
             'Select your level of agreement for the following statements - I have seen or met a neurosurgeon before',
             'Select your level of agreement for the following statements - I have seen or met a Woman neurosurgeon',
             'Select your level of agreement for the following statements - I have seen or met a Black neurosurgeon',
             'Select your level of agreement for the following statements - I have seen or met a Latinx neurosurgeon',
             'Select your level of agreement for the following statements - Neurosurgeons are intimidating',
             'Select your level of agreement for the following statements - Neurosurgeons have a good work-life balance',
             'Select your level of agreement for the following statements - Neurosurgeons have reasonable work hours',
             "Select your level of agreement for the following statements - Neurosurgeons improve their patients' quality of life",
             'Select your level of agreement for the following statements - Neurosurgeons will always have secure jobs',
             'Select your level of agreement for the following statements - Neurosurgeons are financially stable']

col_ortho=['Select your level of agreement for the following statements - I can become an orthopaedic surgeon',
 'Select your level of agreement for the following statements - I have the ability to shadow orthopaedic surgery procedures',
 'Select your level of agreement for the following statements - I am familiar with the career pathway to become an orthopaedic surgeon',
 'Select your level of agreement for the following statements - I have the institutional support and resources to become an orthopaedic surgeon',
 'Select your level of agreement for the following statements - I am connected to mentors that can help me become an orthopaedic surgeon',
 'Select your level of agreement for the following statements - I know the day-to-day responsibilities of an orthopaedic surgeon',
 'Select your level of agreement for the following statements - I can list at least three subspecialties of orthopaedic surgery',
 'Select your level of agreement for the following statements - Orthopaedic Surgery is a good field for minorities and women',
 'Select your level of agreement for the following statements - I have seen or met an orthopaedic surgeon before',
 'Select your level of agreement for the following statements - I have seen or met a Woman orthopaedic surgeon',
 'Select your level of agreement for the following statements - I have seen or met a Black orthopaedic surgeon',
 'Select your level of agreement for the following statements - I have seen or met a Latinx orthopaedic surgeon',
 'Select your level of agreement for the following statements - Orthopaedic Surgeons are intimidating',
 'Select your level of agreement for the following statements - Orthopaedic Surgeons have a good work-life balance',
 'Select your level of agreement for the following statements - Orthopaedic Surgeons have reasonable work hours',
 "Select your level of agreement for the following statements - Orthopaedic Surgeons improve their patients' quality of life",
 'Select your level of agreement for the following statements - Orthopaedic Surgeons will always have secure jobs',
 'Select your level of agreement for the following statements - Orthopaedic Surgeons are financially stable']

col_plastics=['Select your level of agreement for the following statements - I can become a plastic surgeon',
'Select your level of agreement for the following statements - I have the ability to shadow plastic surgery procedures',
'Select your level of agreement for the following statements - I am familiar with the career pathway to become a plastic surgeon',
'Select your level of agreement for the following statements - I have the institutional support and resources to become a plastic surgeon',
'Select your level of agreement for the following statements - I am connected to mentors that can help me become a plastic surgeon',
'Select your level of agreement for the following statements - I know the day-to-day responsibilities of a plastic surgeon',
'Select your level of agreement for the following statements - I can list at least three subspecialties of plastic surgery',
'Select your level of agreement for the following statements - Plastic Surgery is a good field for minorities and women',
'Select your level of agreement for the following statements - I have seen or met a plastic surgeon before',
'Select your level of agreement for the following statements - I have seen or met a Woman plastic surgeon',
'Select your level of agreement for the following statements - I have seen or met a Black plastic surgeon',
'Select your level of agreement for the following statements - I have seen or met a Latinx plastic surgeon',
'Select your level of agreement for the following statements - Plastic Surgeons are intimidating',
'Select your level of agreement for the following statements - Plastic Surgeons have a good work-life balance',
'Select your level of agreement for the following statements - Plastic Surgeons have reasonable work hours',
"Select your level of agreement for the following statements - Plastic Surgeons improve their patients' quality of life",
'Select your level of agreement for the following statements - Plastic Surgeons will always have secure jobs',
'Select your level of agreement for the following statements - Plastic Surgeons are financially stable']

col_setid=['abilities',
           'abilities',
           'knowledge',
           'support',
           'support',
           'knowledge',
           'knowledge',
           'diversity',
           'na',
           'diversity',
           'diversity',
           'diversity',
           'na',
           'field',
           'field',
           'field',
           'field',
           'field',];

subscore_names=['abilities', 'diversity', 'field', 'knowledge', 'support'];

# col_perceptions=[col for col in post23_FLNSUS if col.startswith('Select your level of agreement for the following statements - ')]


def prepost_plot(df_pre, df_post, cols):
    col_names=[i.split(' - ', 1)[1] for i in cols]
    uid_pre=set(df_pre['Unique ID']);
    uid_post=set(df_post['Unique ID']);

    uid_all=list(uid_pre.intersection(uid_post))
    uid_all.sort()

    df_pre_uid=df_pre.loc[df_pre['Unique ID'].isin(uid_all),['Unique ID']+cols];
    df_pre_uid=df_pre_uid.set_index(df_pre_uid['Unique ID']).sort_index();
    df_post_uid=df_post.loc[df_post['Unique ID'].isin(uid_all),['Unique ID']+cols];
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

    fig, ax=plt.subplots(figsize=(4,5),ncols=1,nrows=1,);
    bonf=1;

    # ax.set_yticks(np.arange(0,len(col_names)));

    for idx,col in enumerate(cols):
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

        ax.text(0.9,idx,col_names[idx],
                    verticalalignment='center',horizontalalignment='right',color='black');


    ax.set_yticks(np.arange(0,len(col_names)),labels=[''] * len(col_names));
    ax.set_xticks(np.arange(1,6));
    ax.set_xticklabels(['Strongly\ndisagree','Somewhat\ndisagree',
                        'Neither agree\nnor disagree','Somewhat\nagree',
                        'Strongly\nagree'])    
    ax.grid(axis = 'x',linewidth=0.5)
    ax.grid(axis = 'y',linewidth=0.5)        
    
    return fig, ax

def prepost_data(df_pre, df_post, cols):
    col_names=[i.split(' - ', 1)[1] for i in cols]
    uid_pre=set(df_pre['Unique ID']);
    uid_post=set(df_post['Unique ID']);

    uid_all=list(uid_pre.intersection(uid_post))
    uid_all.sort()

    df_pre_uid=df_pre.loc[df_pre['Unique ID'].isin(uid_all),['Unique ID']+cols];
    df_pre_uid=df_pre_uid.set_index(df_pre_uid['Unique ID']).sort_index();
    df_post_uid=df_post.loc[df_post['Unique ID'].isin(uid_all),['Unique ID']+cols];
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
    
    
    
    return col_names, pre, post

### do stuff to make just subsets
col_names, per_pre_neuro, per_post_neuro= prepost_data(pre23, post23_FLNSUS, col_neuro)
col_names, per_pre_ortho, per_post_ortho= prepost_data(pre23, post23_FLOSUS, col_ortho)
col_names, per_pre_plastics, per_post_plastics= prepost_data(pre23, post23_FLPSUS, col_plastics)

colsets=[col_neuro,col_ortho,col_plastics];
postsets=[post23_FLNSUS,post23_FLOSUS,post23_FLPSUS];
set_names=['Neurosurgery','Orthopaedic Surgery','Plastic Surgery'];

### way to do it, combining them all 

fig,ax = plt.subplots(nrows=3,ncols=5,sharey=True,sharex=True,figsize=(5,10))

for i,name in enumerate(set_names):
    
    col_names, pre, post= prepost_data(pre23, postsets[i], colsets[i])
    
    for j,subname in enumerate(subscore_names):
        
        idx_set=[i for i, x in enumerate(col_setid) if x==subname]
        
        preval=pre.iloc[:,np.array(idx_set)+1].sum(axis=1);
        postval=post.iloc[:,np.array(idx_set)+1].sum(axis=1);
        
        maxval=len(idx_set)*5
        
        tempdf= pd.DataFrame(data={'UID':pre['Unique ID'],
                                   'pre': preval/maxval,
                                   'post': postval/maxval}).melt(id_vars=['UID'],)
        tempdf['conference']=name;
        # sns.lineplot(x="variable", 
        #              y="value",
        #              hue='conference',
        #              seed=1,
        #              palette=[palette_flxsus[i]],
        #              data=tempdf,
        #              ax=ax[i][j],
        #              err_style="bars",
        #              markers=True,
        #              ci=95,
        #              err_kws={'capsize':3},
        #              legend=None)
        
        sns.violinplot(data=tempdf, 
                        ax=ax[i][j],
                        x='conference',
                        y="value", 
                        hue="variable",
                        cut=0,
                        split=True, 
                        inner=None, 
                        fill=False,
                        legend=False)
        ax[i][j].get_legend().set_visible(False)
        p = pg.wilcoxon(preval, postval, alternative='two-sided')
        
        ax[i][j].text(0.5,0.1,"p-val \n{:.1e}".format(p['p-val'][0]),ha='center',fontsize=10)
        
        if j==0:
            ax[i][j].set_ylabel(name,fontsize=12)
            
        if i==0:
            ax[i][j].set_title(subname,fontsize=12)
        
        # sns.histplot(data=tempdf,
        #              y="value",
        #              hue='variable',
        #              bins=np.arange(0,maxval+2)-0.5,
        #              alpha=0.5,
        #              ax=ax[i][j])
        
        # tempdf['score']=subname;
        # tempdf['conference']=name;
        
        # fig,ax = plt.subplots()
        
        # sns.boxplot(x="variable", y="value",data=tempdf,)
        # # sns.violinplot(data=tempdf, 
        # #                ax=ax,
        # #                x='ax',
        # #                y="value", 
        # #                hue="variable",
        # #                cut=0,
        # #                split=True, 
        # #                inner=None, 
        # #                fill=False,)
        



# ax[i][j].set_xlim(-0.2,1.2)
ax[i][j].set_ylim(0,1.05)

os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
# if savefig: fig.savefig('AMEC_FLXSUS_virtual/all_subscores.jpeg',dpi=300);
os.chdir(homedir)

sys.exit()
# sns.despine(fig=fig,bottom=True, trim)

### one way to do it, one by one


for i,name in enumerate(set_names):
    
    col_names, pre, post= prepost_data(pre23, postsets[i], colsets[i])
    
    for subname in subscore_names:
        
        idx_set=[i for i, x in enumerate(col_setid) if x==subname]
        
        preval=pre.iloc[:,np.array(idx_set)+1].sum(axis=1);
        postval=post.iloc[:,np.array(idx_set)+1].sum(axis=1);
        
        tempdf= pd.DataFrame(data={'UID':pre['Unique ID'],
                                   'pre': preval,
                                   'post': postval}).melt(id_vars=['UID'],)
        tempdf['ax']=1;
        
        fig,ax = plt.subplots()
        
        sns.boxplot(x="variable", y="value",data=tempdf,)
        # sns.violinplot(data=tempdf, 
        #                ax=ax,
        #                x='ax',
        #                y="value", 
        #                hue="variable",
        #                cut=0,
        #                split=True, 
        #                inner=None, 
        #                fill=False,)
        
    sys.exit()
    
    
    


sys.exit()

fig, ax = prepost_plot(pre23, post23_FLNSUS, col_neuro)
ax.set_title('FLNSUS 2023 Pre/Post Data')
os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
if savefig: fig.savefig('AMEC_FLXSUS_virtual/Fig_Wilcoxon_2023_FLNSUS.jpeg',dpi=300,bbox_inches='tight');
os.chdir(homedir)

fig, ax = prepost_plot(pre23, post23_FLOSUS, col_ortho)
ax.set_title('FLOSUS 2023 Pre/Post Data')
os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
if savefig: fig.savefig('AMEC_FLXSUS_virtual/Fig_Wilcoxon_2023_FLOSUS.jpeg',dpi=300,bbox_inches='tight');
os.chdir(homedir)

fig, ax = prepost_plot(pre23, post23_FLPSUS, col_plastics)
ax.set_title('FLPSUS 2023 Pre/Post Data')
os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
if savefig: fig.savefig('AMEC_FLXSUS_virtual/Fig_Wilcoxon_2023_FLPSUS.jpeg',dpi=300,bbox_inches='tight');
os.chdir(homedir)

sys.exit()

# =============================================================================
# Plot different 2023 specialty data on the map (presurvey)
# =============================================================================

# load in the world map
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))    

# init the figure    
fig, axs=plt.subplots(figsize=(10,10),ncols=1,nrows=3,)

col_check=['Select your level of agreement for the following statements - I can become a neurosurgeon',
           'Select your level of agreement for the following statements - I can become an orthopaedic surgeon',
           'Select your level of agreement for the following statements - I can become a plastic surgeon']
# set_df=[post23_FLNSUS,post23_FLOSUS,post23_FLPSUS]
set_names=['Neurosurgery','Orthopaedic Surgery','Plastic Surgery']

UID_pre=list()

for i, col_name in enumerate(col_check):
    ax=axs[i]
    
    # create a world plot on that axis
    world.plot(ax=ax,color='#CCCCCC',)
    
    # plot boundaries between countries
    world.boundary.plot(color=[0.5,0.5,0.5],linewidth=0.5,ax=ax,)
    # set figure metadata
    
    df1=pre23.loc[pre23[col_name].isnull()==False,['Unique ID','Latitude','Longitude']]

    UID_pre.append(df1['Unique ID'])

    ax.scatter(df1.loc[:,'Longitude'],
                df1.loc[:,'Latitude'],
                s=10,
                marker='o',
                facecolors=palette_wong[i],
                edgecolors=palette_wong[i],
                linewidths=1,
                alpha=1,
                label=set_names[i])

    ax.set_axis_off()
    ax.set_title(set_names[i])
plt.suptitle('Pre-Survey Map',fontsize=20)
plt.tight_layout()

os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
if savefig: fig.savefig('AMEC_FLXSUS_virtual/pre_map_2023.jpeg',dpi=300);
os.chdir(homedir)


# =============================================================================
# get states temp for abstract
# =============================================================================
# get states
# predf=pd.DataFrame(data=None,index=pre23['Unique ID'],columns=['State'])
# for uid in predf.index:
#      predf.loc[uid,'State']=idfile.loc[idfile['Unique ID']==uid,'Country'].values[0]

locs=pd.DataFrame(data=None,index=UID_pre[1],columns=['State', 'Country'])
for uid in locs.index:
      locs.loc[uid,:]=idfile.loc[idfile['Unique ID']==uid,['State','Country']].values[0]


# =============================================================================
# plots of pre and post race, ethnicity, gender, sexual orientation
# =============================================================================

post_set=[post23_FLNSUS,post23_FLOSUS,post23_FLPSUS]

col_race = ['Race - Other',
            'Race - Multiracial',
            'Race - Prefer not to answer',
            'Race - American Indian or Alaska Native',
            'Race - Asian or Pacific Islander',
            'Race - Black or African American',
            'Race - White or Caucasian',]
            
            
names_race=[i.split(' - ', 1)[1] for i in col_race]

col_ethnicity=['Ethnicity']
names_ethnicity=["Hispanic or Latino",
                 "Not Hispanic or Latino",
                 "Prefer not to answer"][::-1]

col_gender=['Gender']
names_gender=["Female",
              "Male",
              "Non-binary",
              "Gender neutral",
              "Transgender",
              "Prefer not to answer",][::-1]

col_orientation=['Sexual Orientation']
names_orientation=["Heterosexual",
                   "Lesbian",
                   "Bisexual",
                   "Gay",
                   "Queer",
                   "Asexual",
                   "Pansexual",
                   "Prefer not to answer",
                   "Other",][::-1]


colset=col_race + col_ethnicity + col_gender + col_orientation


def getcounts(col,names,df):

    x = list()
    for name in names:
        x.append(df[col].squeeze().str.startswith(name).sum())
    
    x=np.array(x)
    y = np.arange(len(x))
    
    return x,y
    
precolor=palette_wong[1]
postcolor=palette_wong[2]
delt=0.2

for i, symp_name in enumerate(set_names):
    
    # get pre dataframe for specific symposium
    predf=pd.DataFrame(data=None,
                       index=UID_pre[i].values,
                       columns=colset)
    
    for uid in UID_pre[i].values:
        predf.loc[uid,colset]=pre23.loc[pre23['Unique ID']==uid,colset].values
    
    predf=predf.dropna()

    # get pre dataframe for specific symposium
    postdf=pd.DataFrame(data=None,
                       index=UID_post[i].values,
                       columns=colset)
    
    df_post_temp=idfile
    
    for uid in UID_post[i].values:
        postdf.loc[uid,colset]=df_post_temp.loc[df_post_temp['Unique ID']==uid,colset].values
    
    postdf=postdf.dropna()
    
    # get race pre
    x_race_pre=predf[col_race].sum().values
    yl_race_pre=names_race
    y_race_pre=np.arange(len(x_race_pre))
    
    # get race post
    x_race_post=postdf[col_race].sum().values
    yl_race_post=names_race
    y_race_post=np.arange(len(x_race_post))
    
    # get ethnicity pre   
    x_ethnicity_pre,y_ethnicity_pre=getcounts(col_ethnicity,names_ethnicity,predf)
    # get ethnicity post
    x_ethnicity_post,y_ethnicity_post=getcounts(col_ethnicity,names_ethnicity,postdf)
    
    # get gender pre   
    x_gender_pre,y_gender_pre=getcounts(col_gender,names_gender,predf)
    # get gender post
    x_gender_post,y_gender_post=getcounts(col_gender,names_gender,postdf)
   
    # get orientation pre   
    x_orientation_pre,y_orientation_pre=getcounts(col_orientation,names_orientation,predf)
    # get orientation post
    x_orientation_post,y_orientation_post=getcounts(col_orientation,names_orientation,postdf)


    fig, axs=plt.subplots(figsize=(5,12),nrows=4,ncols=1,sharex=True,
                          gridspec_kw={'height_ratios':[7,3,6,9]})
    
    ## race
    axs[0].barh(y=y_race_pre-delt,
             width=x_race_pre/len(predf)*100,
             height=delt*1.8,
             color=precolor)
    axs[0].barh(y=y_race_post+delt,
             width=x_race_post/len(postdf)*100,
             height=delt*1.8,
             color=postcolor)
    axs[0].set_yticks(y_race_post,yl_race_post)
    f_obs = x_race_post
    f_exp = x_race_pre/np.sum(x_race_pre)*np.sum(x_race_post)
    f_obs=f_obs[f_exp>0]# get rid of zeros in expression
    f_exp=f_exp[f_exp>0]

    chip=chisquare(f_obs=f_obs, f_exp=f_exp)[1]
    axs[0].set_title('Race, chi2 p = {0:.3f}'.format(chip))

    
    # ethnicity
    axs[1].barh(y=y_ethnicity_pre-delt,
             width=x_ethnicity_pre/len(predf)*100,
             height=delt*1.8,
             color=precolor)
    axs[1].barh(y=y_ethnicity_post+delt,
             width=x_ethnicity_post/len(postdf)*100,
             height=delt*1.8,
             color=postcolor)
    axs[1].set_yticks(y_ethnicity_post,names_ethnicity)
    #chi2 test
    f_obs = x_ethnicity_post
    f_exp = x_ethnicity_pre/len(predf)*len(postdf)
    f_obs=f_obs[f_exp>0]# get rid of zeros in expression
    f_exp=f_exp[f_exp>0]

    chip=chisquare(f_obs=f_obs, f_exp=f_exp)[1]
    axs[1].set_title('Ethnicity, chi2 p = {0:.3f}'.format(chip))
    
    # gender
    axs[2].barh(y=y_gender_pre-delt,
             width=x_gender_pre/len(predf)*100,
             height=delt*1.8,
             color=precolor)
    axs[2].barh(y=y_gender_post+delt,
             width=x_gender_post/len(postdf)*100,
             height=delt*1.8,
             color=postcolor)
    axs[2].set_yticks(y_gender_post,names_gender)
    #chi2 test
    f_obs = x_gender_post
    f_exp = x_gender_pre/len(predf)*len(postdf)
    f_obs=f_obs[f_exp>0]# get rid of zeros in expression
    f_exp=f_exp[f_exp>0]
    
    chip=chisquare(f_obs=f_obs, f_exp=f_exp)[1]    
    axs[2].set_title('Gender, chi2 p = {0:.3f}'.format(chisquare(f_obs=f_obs, f_exp=f_exp)[1]))
    
    # orientation
    axs[3].barh(y=y_orientation_pre-delt,
             width=x_orientation_pre/len(predf)*100,
             height=delt*1.8,
             color=precolor,
             label='pre-survey (n={})'.format(len(predf)))
    axs[3].barh(y=y_orientation_post+delt,
             width=x_orientation_post/len(postdf)*100,
             height=delt*1.8,
             color=postcolor,
             label='post-survey (n={})'.format(len(postdf)))
    axs[3].set_yticks(y_orientation_post,names_orientation)
    #chi2 test
    f_obs = x_orientation_post
    f_exp = x_orientation_pre/len(predf)*len(postdf)
    f_obs=f_obs[f_exp>0]# get rid of zeros in expression
    f_exp=f_exp[f_exp>0]
    axs[3].set_title('Sexual Orientation, chi2 p = {0:.3f}'.format(chisquare(f_obs=f_obs, f_exp=f_exp)[1]))
    
    axs[3].set_xlabel('% of total')
    axs[3].set_xlim(0,100)
    
    axs[3].legend(loc='lower center',bbox_to_anchor=(0.5,-0.5))
    
    plt.suptitle(symp_name,fontsize=20)
    
    # plt.tight_layout()

    os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
    if savefig: fig.savefig('AMEC_FLXSUS_virtual/prepost_pct_'+symp_name+'.jpeg',dpi=300,bbox_inches='tight');
    os.chdir(homedir)



for i, symp_name in enumerate(set_names):
    
    # get pre dataframe for specific symposium
    predf=pd.DataFrame(data=None,
                       index=UID_pre[i].values,
                       columns=colset)
    
    for uid in UID_pre[i].values:
        predf.loc[uid,colset]=pre23.loc[pre23['Unique ID']==uid,colset].values
    
    predf=predf.dropna()

    # get pre dataframe for specific symposium
    postdf=pd.DataFrame(data=None,
                       index=UID_post[i].values,
                       columns=colset)
    
    df_post_temp=idfile
    
    for uid in UID_post[i].values:
        postdf.loc[uid,colset]=df_post_temp.loc[df_post_temp['Unique ID']==uid,colset].values
    
    postdf=postdf.dropna()
    
    # get race pre
    x_race_pre=predf[col_race].sum().values
    yl_race_pre=names_race
    y_race_pre=np.arange(len(x_race_pre))
    
    # get race post
    x_race_post=postdf[col_race].sum().values
    yl_race_post=names_race
    y_race_post=np.arange(len(x_race_post))
    
    # get ethnicity pre   
    x_ethnicity_pre,y_ethnicity_pre=getcounts(col_ethnicity,names_ethnicity,predf)
    # get ethnicity post
    x_ethnicity_post,y_ethnicity_post=getcounts(col_ethnicity,names_ethnicity,postdf)
    
    # get gender pre   
    x_gender_pre,y_gender_pre=getcounts(col_gender,names_gender,predf)
    # get gender post
    x_gender_post,y_gender_post=getcounts(col_gender,names_gender,postdf)
   
    # get orientation pre   
    x_orientation_pre,y_orientation_pre=getcounts(col_orientation,names_orientation,predf)
    # get orientation post
    x_orientation_post,y_orientation_post=getcounts(col_orientation,names_orientation,postdf)


    fig, axs=plt.subplots(figsize=(5,12),nrows=4,ncols=1,sharex=True,
                          gridspec_kw={'height_ratios':[7,3,6,9]})
    
    # race
    p=axs[0].barh(y=y_race_pre-delt,
             width=x_race_pre,
             height=delt*1.8,
             color=precolor)
    axs[0].bar_label(p,fontsize=10)
    p=axs[0].barh(y=y_race_post+delt,
             width=x_race_post,
             height=delt*1.8,
             color=postcolor)
    axs[0].bar_label(p,fontsize=10)
    axs[0].set_yticks(y_race_post,yl_race_post)
    axs[0].set_title('Race')
    axs[0].legend(loc='lower right')
    
    # ethnicity
    p=axs[1].barh(y=y_ethnicity_pre-delt,
             width=x_ethnicity_pre,
             height=delt*1.8,
             color=precolor)
    axs[1].bar_label(p,fontsize=10)
    p=axs[1].barh(y=y_ethnicity_post+delt,
             width=x_ethnicity_post,
             height=delt*1.8,
             color=postcolor)
    axs[1].bar_label(p,fontsize=10)
    axs[1].set_yticks(y_ethnicity_post,names_ethnicity)
    axs[1].set_title('Ethnicity')
    
    # gender
    p=axs[2].barh(y=y_gender_pre-delt,
             width=x_gender_pre,
             height=delt*1.8,
             color=precolor)
    axs[2].bar_label(p,fontsize=10)
    p=axs[2].barh(y=y_gender_post+delt,
             width=x_gender_post,
             height=delt*1.8,
             color=postcolor)
    axs[2].bar_label(p,fontsize=10)
    axs[2].set_yticks(y_gender_post,names_gender)
    axs[2].set_title('Gender')
    
    # gender
    p=axs[3].barh(y=y_orientation_pre-delt,
             width=x_orientation_pre,
             height=delt*1.8,
             color=precolor,
             label='pre-survey (n={})'.format(len(predf)))
    axs[3].bar_label(p,fontsize=10)
    p=axs[3].barh(y=y_orientation_post+delt,
             width=x_orientation_post,
             height=delt*1.8,
             color=postcolor,
             label='post-survey (n={})'.format(len(postdf)))
    axs[3].bar_label(p,fontsize=10)
    axs[3].set_yticks(y_orientation_post,names_orientation)
    axs[3].set_title('Sexual Orientation')
    
    axs[3].set_xlabel('count')
    # axs[3].set_xlim(0,100)
    
    axs[3].legend(loc='lower center',bbox_to_anchor=(0.5,-0.5))
    
    plt.suptitle(symp_name,fontsize=20)
    
    # plt.tight_layout()

    os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
    if savefig: fig.savefig('AMEC_FLXSUS_virtual/prepost_count_'+symp_name+'.jpeg',dpi=300,bbox_inches='tight');
    os.chdir(homedir)
        

# =============================================================================
# pre and post hometown size, education level, family income, academic stage
# =============================================================================

col_hometown=['Hometown']
names_hometown=['Large City (>150,000 people)',
       'Small City (50,000 - 150,000 people)',
       'Suburban town (10,000 - 50,000 people)',
       'Small town (2,500 - 10,000 people)', 
       'Rural (< 2,500 people)'][::-1]

col_highestfamedu=['What is the highest level of education someone in your immediate family (Mom/Dad/Brother/Sister) has completed?']
names_highestfamedu=['Some high school or less',
                     'High school diploma or GED', 
                     'Some college, but no degree',
                     'Associates or technical degree',
                     'Bachelor’s degree',
                     'Graduate or professional degree',
                     'Prefer not to say',][::-1]

col_family_income=["What is your family's approximate yearly income (in US Dolllars)?"]
names_family_income=['Less than $10,000', 
                     '$10,000 - $19,999', 
                     '$20,000 - $29,999', 
                     '$30,000 - $39,999',
                     '$40,000 - $49,999',
                     '$50,000 - $59,999', 
                     '$60,000 - $69,999', 
                     '$70,000 - $79,999',
                     '$80,000 - $89,999',
                     '$90,000 - $99,999',
                     '$100,000 - $149,999',
                     'More than $150,000',][::-1]

col_academic_stage=["Academic Stage"]
names_academic_stage=['High school freshman',
                      'High school sophomore',
                      'High school junior',
                      'High school senior', 
                      'College freshman', 
                      'College sophomore',
                      'College junior', 
                      'College senior', 
                      'College graduate/Post-baccalaureate', 
                      'Other',][::-1]

colset=col_hometown + col_highestfamedu + col_family_income + col_academic_stage
nameset=[names_hometown,names_highestfamedu,names_family_income,names_academic_stage]
supset=['Hometown Size','Highest Family Education','Family Income', 'Academic Stage']

for i, symp_name in enumerate(set_names):
    
    # get pre dataframe for specific symposium
    predf=pd.DataFrame(data=None,
                       index=UID_pre[i].values,
                       columns=colset)
    
    for uid in UID_pre[i].values:
        predf.loc[uid,colset]=pre23.loc[pre23['Unique ID']==uid,colset].values
    
    predf=predf.dropna()

    # get pre dataframe for specific symposium
    postdf=pd.DataFrame(data=None,
                       index=UID_post[i].values,
                       columns=colset)
    
    for uid in UID_post[i].values:
        # do this off predf dataframe
        if (predf.index==uid).sum()>0:
            postdf.loc[uid,colset]=predf.loc[uid,colset].values
    
    postdf=postdf.dropna()
    
    fig, axs=plt.subplots(figsize=(5,12),nrows=4,ncols=1,sharex=True,
                          gridspec_kw={'height_ratios':[5,7,12,10]})
    
    for j,cols in enumerate(colset):
        xpre,ypre=getcounts(cols,nameset[j],predf)
        xpost,ypost=getcounts(cols,nameset[j],postdf)
        
        axs[j].barh(y=ypre-delt,
                 width=xpre/len(predf)*100,
                 height=delt*1.8,
                 color=precolor,
                 label='pre-survey (n={})'.format(len(predf)))
        
        axs[j].barh(y=ypost+delt,
                 width=xpost/len(postdf)*100,
                 height=delt*1.8,
                 color=postcolor,
                 label='post-survey (n={})'.format(len(postdf)))
        
        # replce $ in string
        name_og=nameset[j]
        name_nu=[sub.replace("$","\$") for sub in name_og]
        
        axs[j].set_yticks(ypost,name_nu)
        f_obs = xpost
        f_exp = xpre/np.sum(xpre)*np.sum(xpost)
        f_obs=f_obs[f_exp>0]# get rid of zeros in expression
        f_exp=f_exp[f_exp>0]

        chip=chisquare(f_obs=f_obs, f_exp=f_exp)[1]
        axs[j].set_title(supset[j]+', chi2 p = {0:.3f}'.format(chip))
    
    axs[3].set_xlabel('% of total')
    axs[3].set_xlim(0,100)
    
    axs[3].legend(loc='lower center',bbox_to_anchor=(0.5,-0.5))
    
    plt.suptitle(symp_name,fontsize=20)
    
    os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
    if savefig: fig.savefig('AMEC_FLXSUS_virtual/prepost_other_pct_'+symp_name+'.jpeg',dpi=300,bbox_inches='tight');
    os.chdir(homedir)
    
    fig, axs=plt.subplots(figsize=(5,12),nrows=4,ncols=1,sharex=True,
                          gridspec_kw={'height_ratios':[5,7,12,10]})
    
    for j,cols in enumerate(colset):
        xpre,ypre=getcounts(cols,nameset[j],predf)
        xpost,ypost=getcounts(cols,nameset[j],postdf)
        
        p = axs[j].barh(y=ypre-delt,
                 width=xpre,
                 height=delt*1.8,
                 color=precolor,
                 label='pre-survey (n={})'.format(len(predf)))
        axs[j].bar_label(p,fontsize=10)
        p = axs[j].barh(y=ypost+delt,
                 width=xpost,
                 height=delt*1.8,
                 color=postcolor,
                 label='post-survey (n={})'.format(len(postdf)))
        axs[j].bar_label(p,fontsize=10)
        # replce $ in string
        name_og=nameset[j]
        name_nu=[sub.replace("$","\$") for sub in name_og]
        axs[j].set_yticks(ypost,name_nu)
        axs[j].set_title(supset[j])
    
    axs[3].set_xlabel('count')
    axs[3].legend(loc='lower center',bbox_to_anchor=(0.5,-0.5))
    
    plt.suptitle(symp_name,fontsize=20)
    
    os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
    if savefig: fig.savefig('AMEC_FLXSUS_virtual/prepost_other_'+symp_name+'.jpeg',dpi=300,bbox_inches='tight');
    os.chdir(homedir)

# =============================================================================
# Pre Perceptions of Medicine versus Surgery
# =============================================================================

col_selfbelief=['Select your level of agreement for the following statements - I will become a doctor',
                'Select your level of agreement for the following statements - I can become a neurosurgeon',
                'Select your level of agreement for the following statements - I can become an orthopaedic surgeon',
                'Select your level of agreement for the following statements - I can become a plastic surgeon']

col_conception=['Select your level of agreement for the following statements - Medicine is a good field for minorities and women',
                'Select your level of agreement for the following statements - Neurosurgery is a good field for minorities and women',
                'Select your level of agreement for the following statements - Orthopaedic Surgery is a good field for minorities and women',
                'Select your level of agreement for the following statements - Plastic Surgery is a good field for minorities and women']

col_familiarity=['Select your level of agreement for the following statements - I am familiar with the career pathway to become a doctor',
                 'Select your level of agreement for the following statements - I am familiar with the career pathway to become a neurosurgeon',
                 'Select your level of agreement for the following statements - I am familiar with the career pathway to become an orthopaedic surgeon',
                 'Select your level of agreement for the following statements - I am familiar with the career pathway to become a plastic surgeon'];

col_specialty=['Neurosurgery','Orthopaedic\nSurgery','Plastic\nSurgery'];

colset=[col_selfbelief,col_conception,col_familiarity];
nameset=['Self Belief',
         'Good field for minorities and women',
         'Familarity with Career Path']

for i, cols in enumerate(colset):
    
    fig, ax=plt.subplots(figsize=(6,6),ncols=1,nrows=1,);
    
    for j, comp in enumerate(col_specialty):
        col2get=[cols[x] for x in [0,j+1]]
        
        tempdf=pre23[col2get].dropna()

        tempdf=tempdf.replace({'Strongly agree': 5, 
                                   'Somewhat agree': 4,
                                   'Neither agree nor disagree': 3,
                                   'Somewhat disagree': 2,
                                   'Strongly disagree': 1,})
    
        stats=pg.wilcoxon(tempdf.iloc[:,0],
                          tempdf.iloc[:,1],
                          alternative='two-sided')
        
        pval=stats['p-val'][0]
        
        if j==2:
            ax.plot(j+1,np.mean(tempdf.iloc[:,0]),'xk',label='general medicine');
            ax.plot(j+1,np.mean(tempdf.iloc[:,1]),'or',label='subspecialty');
        else:
            ax.plot(j+1,np.mean(tempdf.iloc[:,1]),'or');
            ax.plot(j+1,np.mean(tempdf.iloc[:,0]),'xk');
            
        ax.text(j+1,5.1,'p = {0:.6f}'.format(pval),ha='center');
        
    ax.set_ylim(0,6)
    ax.set_yticks(np.arange(1,6),['Strongly\ndisagree','Somewhat\ndisagree',
                        'Neither agree\nnor disagree','Somewhat\nagree',
                        'Strongly\nagree'])    
    ax.grid(axis = 'x',linewidth=0.5)
    ax.grid(axis = 'y',linewidth=0.5)        
    
    ax.set_xticks(np.arange(1,4),col_specialty)  
    ax.legend(loc='lower center')
    ax.set_xlim(0.5,3.5)
    ax.set_title('Pre-Survey: ' + nameset[i])
    
    
    os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
    if savefig: fig.savefig('AMEC_FLXSUS_virtual/pre_perceptionsVmed_'+nameset[i]+'.jpeg',dpi=300,bbox_inches='tight');
    os.chdir(homedir)
    
postdatas=[post23_FLNSUS,post23_FLOSUS,post23_FLPSUS]
for i, cols in enumerate(colset):
    
    fig, ax=plt.subplots(figsize=(6,6),ncols=1,nrows=1,);
    
    for j, comp in enumerate(col_specialty):
        col2get=[cols[x] for x in [0,j+1]]
        
        ladida=postdatas[j]
        tempdf=ladida[col2get].dropna()

        tempdf=tempdf.replace({'Strongly agree': 5, 
                                   'Somewhat agree': 4,
                                   'Neither agree nor disagree': 3,
                                   'Somewhat disagree': 2,
                                   'Strongly disagree': 1,})
    
        stats=pg.wilcoxon(tempdf.iloc[:,0],
                          tempdf.iloc[:,1],
                          alternative='two-sided')
        
        pval=stats['p-val'][0]
        
        if j==2:
            ax.plot(j+1,np.mean(tempdf.iloc[:,0]),'xk',label='general medicine');
            ax.plot(j+1,np.mean(tempdf.iloc[:,1]),'or',label='subspecialty');
        else:
            ax.plot(j+1,np.mean(tempdf.iloc[:,1]),'or');
            ax.plot(j+1,np.mean(tempdf.iloc[:,0]),'xk');
            
        ax.text(j+1,5.1,'p = {0:.6f}'.format(pval),ha='center');
        
    ax.set_ylim(0,6)
    ax.set_yticks(np.arange(1,6),['Strongly\ndisagree','Somewhat\ndisagree',
                        'Neither agree\nnor disagree','Somewhat\nagree',
                        'Strongly\nagree'])    
    ax.grid(axis = 'x',linewidth=0.5)
    ax.grid(axis = 'y',linewidth=0.5)        
    
    ax.set_xticks(np.arange(1,4),col_specialty)  
    ax.legend(loc='lower center')
    ax.set_xlim(0.5,3.5)
    ax.set_title('Post-Survey: ' + nameset[i])
    
    
    os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
    if savefig: fig.savefig('AMEC_FLXSUS_virtual/post_perceptionsVmed_'+nameset[i]+'.jpeg',dpi=300,bbox_inches='tight');
    os.chdir(homedir)
    
# =============================================================================
# Symposium Ratings and Feedback
# =============================================================================

col_rating=['Symposium Rating']
names_rating=np.arange(0,11)

col_future=['Would you attend a future symposium?']
names_future=['Definitely yes', 
                     'Probably yes', 
                     'Might or might not',
                     'Probably no', 
                     'Definitely no',][::-1]

col_rec=['Would you recommend the event to a family member, friend, or colleague?']
names_rec=['Definitely yes', 
                     'Probably yes', 
                     'Might or might not',
                     'Probably no', 
                     'Definitely no',][::-1]

colset=col_rating + col_future + col_rec
nameset=[names_rating,names_future,names_rec]
supset=['Symposium Rating','Attend Future Symposium','Recommend Symposium']

postdatas=[post23_FLNSUS,post23_FLOSUS,post23_FLPSUS]
postspecialty=['Neurosurgery','Orthopaedic Surgery','Plastic Surgery']


def getcounts2(col,names,df):

    x = list()
    for name in names:
        x.append((df[col]==name).sum())
    
    x=np.array(x)
    y = np.arange(len(x))
    
    return x,y

for i, sup in enumerate(supset):
    col=colset[i]
    name=nameset[i]    
    
    
    fig, ax = plt.subplots(figsize=(8,4),
                           nrows=1,
                           ncols=1,)
                           # sharex=True)
    
    for j, specialty in enumerate(postspecialty):
        tempdf=postdatas[j]
        spec=postspecialty[j]        

        y,x=getcounts2(col,name,tempdf)
        
        ax.bar(x=x-0.3+0.3*j,
                 height=y/np.sum(y)*100,
                 width=0.25,
                 color=palette_wong[j+1],
                 label=specialty
                 )
        for k in range(len(y)):
            if y[k]>0:
                ax.text(x=x[k]-0.3+0.3*j,
                        y=y[k]/np.sum(y)*100+1,
                        s=y[k],
                        fontsize=10,
                        ha='center',
                        )
        
    ax.set_xticks(x,name)
    wrap_labels_x(ax,10,break_long_words=False)        
    ax.legend()
    ax.set_title(sup)
    ax.set_ylabel('% of Participants')
    
    os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
    if savefig: fig.savefig('AMEC_FLXSUS_virtual/Feedback_'+sup+'.jpeg',dpi=300,bbox_inches='tight');
    os.chdir(homedir)
