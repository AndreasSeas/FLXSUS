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
# also have openpyxl

# =============================================================================
# Set init parameters
# =============================================================================
savefig=True
deprecate = False

# =============================================================================
# Load the data
# =============================================================================
homedir=os.getcwd()
datadir='/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/Database';

os.chdir(datadir)

pre21=pd.read_excel("FLNSUS2021Pre_Filtered(05-Database).xlsx")
post21=pd.read_excel("FLNSUS2021Post_Filtered(05-Database).xlsx")
pre22=pd.read_excel("FLNSUS2022Pre_Filtered(05-Database).xlsx");
post22=pd.read_excel("FLNSUS2022Post_Filtered(05-Database).xlsx")
mid23=pd.read_excel("FLNSUS_Winter_2023_Followup(05-Database).xlsx")
idfile=pd.read_excel("IDFile.xlsx")

dflist=[pre21,post21,pre22,post22,mid23];
dfname=['presurvey 2021',
        'postsurvey 2021',
        'presurvey 2022',
        'postsurvey 2022',
        'mid-year check-in 2023'];

os.chdir(homedir)

# =============================================================================
# Definitions
# =============================================================================

# =============================================================================
# map figure with 2021 v.s. 2022
# =============================================================================

# https://onelinerhub.com/python-matplotlib/how-to-fill-countries-with-colors-using-world-map

world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
# from https://www.naturalearthdata.com/downloads/110m-cultural-vectors/

plt.rcParams['font.size'] = '12'
plt.rcParams['font.family'] = 'serif'

fig, ax=plt.subplots(figsize=(14,7),ncols=1,nrows=1,)
world.plot(ax=ax,color='#CCCCCC',)
world.boundary.plot(color=[0.5,0.5,0.5],linewidth=0.5,ax=ax,)

sz=15
alph=0.5
ax.scatter(pre21.loc[:,'Longitude'],
           pre21.loc[:,'Latitude'],
           sz,
           'red',
           alpha=alph,
           label='2021 Participants')

ax.scatter(pre22.loc[:,'Longitude'],
           pre22.loc[:,'Latitude'],
           sz,
           'blue',
           alpha=alph,
           label='2022 Participants')

ax.legend(loc='lower center',ncol=5,bbox_to_anchor=[0.5,-0.02])

plt.tight_layout()
plt.axis('off')

os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
if savefig: fig.savefig('Figures/Fig_map_21v22_v1.png',dpi=600);
os.chdir(homedir)

sys.exit()

# =============================================================================
# New map figure
# =============================================================================

import geopandas
from shapely.geometry import Point
# https://onelinerhub.com/python-matplotlib/how-to-fill-countries-with-colors-using-world-map

world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
# from https://www.naturalearthdata.com/downloads/110m-cultural-vectors/

all_coords=pd.DataFrame(data=None,columns=['Unique ID','Geometry']);
for df in dflist:
    df['Geometry']=np.nan;
    for i in df.index:
        df.loc[i,'Geometry']=Point(df.loc[i,'Longitude'],df.loc[i,'Latitude'])
        
    all_coords=pd.concat([all_coords,df[['Unique ID','Geometry']]])

all_coords=all_coords.reset_index(drop=True);
all_coords['Country']=np.nan

# go thru each country
for i in range(len(world)):
    # print(world.loc[i,'name'])
    geom=world.loc[i,'geometry']
    for j in range(len(all_coords)):
        pt=all_coords.loc[j,'Geometry']
        if geom.contains(pt):
            all_coords.loc[j,'Country']=world.loc[i,'name'];
        # print(geom.contains(pt))
        
# make sure to only keep unique countries
world['count']=0;
unique_coords=pd.DataFrame(data=None,columns=['Country'],index=all_coords['Unique ID'].unique());

countmult=0
for i, idx in enumerate(unique_coords.index):

    ctemp=all_coords.loc[all_coords['Unique ID']==idx,'Country']
    
    for loc in ctemp.unique():
        world.loc[world.name == loc,'count']=world.loc[world.name == loc,'count']+1
    
    if len(ctemp.unique())==1:
        unique_coords.loc[idx,'Country']=ctemp.iloc[0]
        
    else:
        unique_coords.loc[idx,'Country']="multiple"
        # print(ctemp)
        # countmult=countmult+1

# world['count']=0;

# polygon.contains(point)

plt.rcParams['font.size'] = '12'
plt.rcParams['font.family'] = 'serif'

fig, ax=plt.subplots(figsize=(16,8),ncols=1,nrows=1,)
world.plot(column='count',legend=True,ax=ax,vmin=0,vmax=10,cmap='Blues',
           legend_kwds={"label": "# Participants", "orientation": "vertical"},)

# import matplotlib.colors as colors
# world.plot(column='count',legend=True,ax=ax,vmin=1,vmax=10,cmap='Blues',
#            norm=colors.LogNorm(vmin=1,vmax=world['count'].max()))
world[world.name == 'United States of America'].plot(color=[0.5,0.5,0.5],ax=ax)

world.boundary.plot(color=[0.5,0.5,0.5],linewidth=0.5,ax=ax,)
# plt.scatter(all_coords.Geometry)

# world[world.name == 'Ukraine'].plot(color='yellow',ax=ax)

# plt.show()

# markerlist=['x','o','x','o','.'];
colorlist=['#e70843','#90052a','#658b6c','#094614','#391164'];

for i,df_ in enumerate(dflist):
    # print(df_.shape)
    x_=df_.loc[:,'Longitude']
    y_=df_.loc[:,'Latitude']
    # ax.plot(x_,y_,markerlist[i],label=dfname[i],);
    ax.scatter(x_,y_,50,colorlist[i],alpha=0.3,label=dfname[i],)

ax.legend(loc='lower center',ncol=5,bbox_to_anchor=[0.5,-0.1])


plt.tight_layout()
plt.axis('off')

os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
if savefig: fig.savefig('Figures/Fig_map_v3.png',dpi=600);
os.chdir(homedir)

sys.exit()

# # =============================================================================
# # Collate Race Data Across years
# # =============================================================================

# col_race=[col for col in idfile if col.startswith('Race')]

# df_race=pd.DataFrame(data=None,index=dfname, columns=col_race)
# df_race_pct=pd.DataFrame(data=None,index=dfname, columns=col_race)

# for i,df_ in enumerate(dflist):
    
#     uid=df_.loc[:,'Unique ID']
#     temp_i=idfile.loc[:,'Unique ID'].isin(uid)
#     numpt=len(uid)
    
#     for j,col in enumerate(col_race):
        
#         df_race.iloc[i,j]=(idfile.loc[temp_i,col]==True).sum()
#         df_race_pct.iloc[i,j]=(idfile.loc[temp_i,col]==True).sum()/numpt

# df_race.insert(0, 'survey', 0)
# df_race.loc[:,'survey']=df_race.index;
# df_race_melt=df_race.melt(id_vars='survey');
# df_race_melt['variable']=df_race_melt['variable'].str.replace('Race - ','')

# fig,ax = plt.subplots(figsize=(10,6))
# s=sns.histplot(data=df_race_melt,
#                x='survey',
#                weights='value',
#                hue='variable',
#                multiple='stack',
#                stat='count',
#                ax=ax,
#                palette='colorblind')

# sns.move_legend(s, 
#                 "lower center", 
#                 bbox_to_anchor=(0.5,1),
#                 ncol=3,
#                 title=None)

# ax.set_xticklabels(['presurvey\n2021',
#                     'postsurvey\n2021',
#                     'presurvey\n2022',
#                     'postsurvey\n2022',
#                     'mid-year check-in\n2023'])
# os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
# if savefig: fig.savefig('Figures/Fig_race_v1.png',dpi=600);
# os.chdir(homedir)

# =============================================================================
# Collate Race Across Years with X2
# =============================================================================
# in this case, are not including individuals who are multiracial in their 
# other race identities in order to allow X2
col_race=[col for col in idfile if col.startswith('Race')]
df_race_pct=pd.DataFrame(data=None,index=dfname, columns=col_race)

for i,df_ in enumerate(dflist):
    
    uid=df_.loc[:,'Unique ID']
    temp_i=idfile.loc[:,'Unique ID'].isin(uid)
    numpt=len(uid)
    
    temp_df=idfile.loc[temp_i,:]
    
    # remove multiracial from other categories
    for col in col_race[:-1]:
        temp_df.loc[temp_df['Race - Multiracial']==True,col]=False
    
    # mark empty race columns as "Other"
    for row in temp_df.index:
        if temp_df.loc[row,col_race].sum()==0:
            temp_df.loc[row,'Race - Prefer not to answer']=True
        # temp_df.loc[temp_df['Race - Multiracial']==True,col]=False
    
    # temp_i=idfile.loc[:,'Unique ID'].isin(uid)
    
    for j,col in enumerate(col_race):
        
        # df_race.iloc[i,j]=(idfile.loc[temp_i,col]==True).sum()
        df_race_pct.iloc[i,j]=(temp_df.loc[:,col]==True).sum()/numpt*100

df_race_pct.insert(0, 'survey', 0)
df_race_pct.loc[:,'survey']=df_race_pct.index;
df_race_melt=df_race_pct.melt(id_vars='survey');
df_race_melt['variable']=df_race_melt['variable'].str.replace('Race - ','')


fig,ax = plt.subplots(figsize=(10,6))
s=sns.histplot(data=df_race_melt,
               x='survey',
               weights='value',
               hue='variable',
               multiple='stack',
               ax=ax,
               palette='colorblind')

sns.move_legend(s, 
                "lower center", 
                bbox_to_anchor=(0.5,1),
                ncol=3,
                title=None)

ax.set_ylabel('Percent By Survey')
ax.set_xlabel('')
ax.set_xticklabels(['presurvey\n2021',
                    'postsurvey\n2021',
                    'presurvey\n2022',
                    'postsurvey\n2022',
                    'mid-year check-in\n2023'])
plt.tight_layout()
os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
if savefig: fig.savefig('Figures/Fig_race_v2.png',dpi=600);
os.chdir(homedir)


# =============================================================================
# Collate Ethnicity Data Across years
# =============================================================================
# col_ethnicity=pre22.Ethnicity.unique()
col_ethnicity=pre22['Ethnicity'].str.strip().unique()

# df_race_pct=pd.DataFrame(data=None,index=dfname, columns=col_race)
df_ethnicity=pd.DataFrame(data=None,index=dfname,columns=col_ethnicity);

for i,df_ in enumerate(dflist):
    
    uid=df_.loc[:,'Unique ID']
    numpt=len(uid)
    temp_i=idfile.loc[:,'Unique ID'].isin(uid)
    temp_df=pd.DataFrame(data=None,columns=['survey','Ethnicity'])
    temp_df.Ethnicity=idfile.loc[temp_i,'Ethnicity']
    
    # if some are nan, replace with prefer not to answer
    temp_df['Ethnicity']=temp_df['Ethnicity'].replace(np.nan, "Prefer not to answer")
    
    temp_df['Ethnicity']=temp_df['Ethnicity'].str.strip();# ensure strip
    
    for col in df_ethnicity.columns:
        df_ethnicity.loc[dfname[i],col]=(temp_df.Ethnicity==col).sum()/numpt*100
    
    # temp_df.Ethnicity=idfile.loc[temp_i,'Ethnicity']
    # temp_df.survey=dfname[i]
    # df_ethnicity=pd.concat([df_ethnicity,temp_df])

# df_ethnicity=df_ethnicity.reset_index()

df_ethnicity.insert(0, 'survey', 0)
df_ethnicity.loc[:,'survey']=df_race_pct.index;
df_race_melt=df_ethnicity.melt(id_vars='survey');
df_race_melt['variable']=df_race_melt['variable'].str.replace('Race - ','')


fig,ax = plt.subplots(figsize=(10,6))
s=sns.histplot(data=df_race_melt,
               x='survey',
               weights='value',
               hue='variable',
               multiple='stack',
               ax=ax,
               palette='colorblind')

sns.move_legend(s, 
                "lower center", 
                bbox_to_anchor=(0.5,1),
                ncol=3,
                title=None)

ax.set_ylabel('Percent By Survey')
ax.set_xlabel('')
ax.set_xticklabels(['presurvey\n2021',
                    'postsurvey\n2021',
                    'presurvey\n2022',
                    'postsurvey\n2022',
                    'mid-year check-in\n2023'])

plt.tight_layout()
os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
if savefig: fig.savefig('Figures/Fig_ethnicity_v2.png',dpi=600);
os.chdir(homedir)

# # =============================================================================
# # Collate Ethnicity Data Across years (deprecated)
# # =============================================================================

# df_ethnicity=pd.DataFrame(data=None,columns=['survey','Ethnicity']);

# for i,df_ in enumerate(dflist):
    
#     uid=df_.loc[:,'Unique ID']
#     temp_i=idfile.loc[:,'Unique ID'].isin(uid)
#     temp_df=pd.DataFrame(data=None,columns=['survey','Ethnicity'])
#     temp_df.Ethnicity=idfile.loc[temp_i,'Ethnicity']
#     temp_df.survey=dfname[i]
#     df_ethnicity=pd.concat([df_ethnicity,temp_df])

# df_ethnicity=df_ethnicity.reset_index()

# fig,ax = plt.subplots(figsize=(10,6))
# s=sns.histplot(data=df_ethnicity,
#              x='survey',
#              hue='Ethnicity',
#              stat='count',
#              multiple='stack',
#              palette='colorblind')

# sns.move_legend(s, 
#                 "lower center", 
#                 bbox_to_anchor=(0.5,1),
#                 ncol=3,
#                 title=None)

# ax.set_xticklabels(['presurvey\n2021',
#                     'postsurvey\n2021',
#                     'presurvey\n2022',
#                     'postsurvey\n2022',
#                     'mid-year check-in\n2023']);

# os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
# if savefig: fig.savefig('Figures/Fig_ethnicity_v1.png',dpi=600);
# os.chdir(homedir)

# =============================================================================
# Collate Gender Data Across years
# =============================================================================

col_gender=pre22['Gender'].str.strip().unique()


df_gender=pd.DataFrame(data=None,index=dfname,columns=col_gender);

for i,df_ in enumerate(dflist):
    
    uid=df_.loc[:,'Unique ID']
    numpt=len(uid)
    temp_i=idfile.loc[:,'Unique ID'].isin(uid)
    temp_df=pd.DataFrame(data=None,columns=['survey','Gender'])
    temp_df.Gender=idfile.loc[temp_i,'Gender']
    
    # if some are nan, replace with prefer not to answer
    temp_df['Gender']=temp_df['Gender'].replace(np.nan, "Prefer not to answer")
    
    temp_df['Gender']=temp_df['Gender'].str.strip();# ensure strip
    
    for col in df_gender.columns:
        print(col)
        df_gender.loc[dfname[i],col]=(temp_df.Gender==col).sum()/numpt*100
    
df_gender.insert(0, 'survey', 0)
df_gender.loc[:,'survey']=df_gender.index;
df_gender_melt=df_gender.melt(id_vars='survey');
df_gender_melt['variable']=df_gender_melt['variable'].str.replace('Race - ','')


fig,ax = plt.subplots(figsize=(10,6))
s=sns.histplot(data=df_gender_melt,
               x='survey',
               weights='value',
               hue='variable',
               multiple='stack',
               ax=ax,
               palette='colorblind')

sns.move_legend(s, 
                "lower center", 
                bbox_to_anchor=(0.5,1),
                ncol=3,
                title=None)

ax.set_ylabel('Percent By Survey')
ax.set_xlabel('')
ax.set_xticklabels(['presurvey\n2021',
                    'postsurvey\n2021',
                    'presurvey\n2022',
                    'postsurvey\n2022',
                    'mid-year check-in\n2023'])

plt.tight_layout()
os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
if savefig: fig.savefig('Figures/Fig_gender_v2.png',dpi=600);
os.chdir(homedir)

# # =============================================================================
# # Collate Gender Data Across years - deprecated
# # =============================================================================

# df_gender=pd.DataFrame(data=None,columns=['survey','Gender']);

# for i,df_ in enumerate(dflist):
    
#     uid=df_.loc[:,'Unique ID']
#     temp_i=idfile.loc[:,'Unique ID'].isin(uid)
#     temp_df=pd.DataFrame(data=None,columns=['survey','Gender'])
#     temp_df.Gender=idfile.loc[temp_i,'Gender']
#     temp_df.survey=dfname[i]
#     df_gender=pd.concat([df_gender,temp_df])

# df_gender=df_gender.reset_index()

# fig,ax = plt.subplots(figsize=(10,6))
# s=sns.histplot(data=df_gender,
#              x='survey',
#              hue='Gender',
#              stat='count',#
#              multiple='stack',
#              palette='colorblind',)

# sns.move_legend(s, 
#                 "lower center", 
#                 bbox_to_anchor=(0.5,1),
#                 ncol=3,
#                 title=None)

# ax.set_xticklabels(['presurvey\n2021',
#                     'postsurvey\n2021',
#                     'presurvey\n2022',
#                     'postsurvey\n2022',
#                     'mid-year check-in\n2023']);
# os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
# if savefig: fig.savefig('Figures/Fig_gender_v1.png',dpi=600);
# os.chdir(homedir)

# # =============================================================================
# # Collate Sexual Orientation Data Across years - deprecated
# # =============================================================================

# df_orientation=pd.DataFrame(data=None,columns=['survey','Sexual Orientation']);


# for i,df_ in enumerate(dflist):
    
#     uid=df_.loc[:,'Unique ID']
#     temp_i=idfile.loc[:,'Unique ID'].isin(uid)
#     temp_df=pd.DataFrame(data=None,columns=['survey','Sexual Orientation'])
#     temp_df['Sexual Orientation']=idfile.loc[temp_i,'Sexual Orientation']
#     temp_df.survey=dfname[i]
#     df_orientation=pd.concat([df_orientation,temp_df])

# df_orientation=df_orientation.reset_index()

# fig,ax = plt.subplots(figsize=(10,6))
# s=sns.histplot(data=df_orientation,
#              x='survey',
#              hue='Sexual Orientation',
#              stat='count',#
#              multiple='stack',
#              palette='colorblind',)

# sns.move_legend(s, 
#                 "lower center", 
#                 bbox_to_anchor=(0.5,1),
#                 ncol=3,
#                 title=None)

# ax.set_xticklabels(['presurvey\n2021',
#                     'postsurvey\n2021',
#                     'presurvey\n2022',
#                     'postsurvey\n2022',
#                     'mid-year check-in\n2023']);
# os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
# if savefig: fig.savefig('Figures/Fig_SexualOrientation_v1.png',dpi=600);
# os.chdir(homedir)

# =============================================================================
# Collate Sexual Orientation Data Across years
# =============================================================================

col_orientation=pre22['Sexual Orientation'].str.strip().unique()

df_orientation=pd.DataFrame(data=None,index=dfname,columns=col_orientation);

for i,df_ in enumerate(dflist):
    
    uid=df_.loc[:,'Unique ID']
    numpt=len(uid)
    temp_i=idfile.loc[:,'Unique ID'].isin(uid)
    temp_df=pd.DataFrame(data=None,columns=['survey','Sexual Orientation'])
    temp_df['Sexual Orientation']=idfile.loc[temp_i,'Sexual Orientation']
    
    # if some are nan, replace with prefer not to answer
    temp_df['Sexual Orientation']=temp_df['Sexual Orientation'].replace(np.nan, "Prefer not to answer")
    
    temp_df['Sexual Orientation']=temp_df['Sexual Orientation'].str.strip();# ensure strip
    
    for col in df_orientation.columns:
        # print(col)
        df_orientation.loc[dfname[i],col]=(temp_df['Sexual Orientation']==col).sum()/numpt*100
    
df_orientation.insert(0, 'survey', 0)
df_orientation.loc[:,'survey']=df_orientation.index;
df_orientation_melt=df_orientation.melt(id_vars='survey');
# df_orientation_melt['variable']=df_orientation_melt['variable'].str.replace('Race - ','')


fig,ax = plt.subplots(figsize=(10,6))
s=sns.histplot(data=df_orientation_melt,
               x='survey',
               weights='value',
               hue='variable',
               multiple='stack',
               ax=ax,
               palette='colorblind')

sns.move_legend(s, 
                "lower center", 
                bbox_to_anchor=(0.5,1),
                ncol=3,
                title=None)

ax.set_ylabel('Percent By Survey')
ax.set_xlabel('')
ax.set_xticklabels(['presurvey\n2021',
                    'postsurvey\n2021',
                    'presurvey\n2022',
                    'postsurvey\n2022',
                    'mid-year check-in\n2023'])

plt.tight_layout()
os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
if savefig: fig.savefig('Figures/Fig_SexualOrientation_v2.png',dpi=600);
os.chdir(homedir)
sys.exit()
# # =============================================================================
# # Collaged Income Data in 2022
# # =============================================================================
# 
# fig,ax = plt.subplots(figsize=(10,6))

# financial_order=["Less than $10,000",
#                  "\$10,000 - $19,999",
#                  "\$20,000 - $29,999",
#                  "\$30,000 - $39,999",
#                  "\$40,000 - $49,999",
#                  "\$50,000 - $59,999",
#                  "\$60,000 - $69,999",
#                  "\$70,000 - $79,999",
#                  "\$80,000 - $89,999",
#                  "\$90,000 - $99,999",
#                  "\$100,000 - $149,999",
#                  "More than $150,000"];

# s=sns.countplot(data=pre22,
#              y="What is your family's approximate yearly income (in US Dolllars)?",
#              order=financial_order,color=(0.00392156862745098, 0.45098039215686275, 0.6980392156862745));
# os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
# if savefig: fig.savefig('Figures/Fig_IncomeData.png',dpi=600);
# os.chdir(homedir)


# # =============================================================================
# # Collate Income Data in 2022
# # =============================================================================
# # get MSQ data first, from 2022
# refdir='/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/ReferenceData';
# os.chdir(refdir)
# msq_income=pd.read_excel('MSQ_Table6_Q24_2022.xlsx')
# os.chdir(homedir)

# fig,ax = plt.subplots(figsize=(10,6))

# financial_order=["Less than $10,000",
#                  "\$10,000 - $19,999",
#                  "\$20,000 - $29,999",
#                  "\$30,000 - $39,999",
#                  "\$40,000 - $49,999",
#                  "\$50,000 - $59,999",
#                  "\$60,000 - $69,999",
#                  "\$70,000 - $79,999",
#                  "\$80,000 - $89,999",
#                  "\$90,000 - $99,999",
#                  "\$100,000 - $149,999",
#                  "More than $150,000"];
# financial_min=[0,
#                10000,
#                20000,
#                30000,
#                40000,
#                50000,
#                60000,
#                70000,
#                80000,
#                90000,
#                100000,
#                150000,]
# financial_max=[10000,
#                20000,
#                30000,
#                40000,
#                50000,
#                60000,
#                70000,
#                80000,
#                90000,
#                100000,
#                150000,
#                200000]

# financial_df=pd.DataFrame(data=None,index=financial_order,columns=['numeach','Min','Max']);
# financial_df['Min']=financial_min;
# financial_df['Max']=financial_max;


# tempcol=pre22["What is your family's approximate yearly income (in US Dolllars)?"];
# numpt=len(pre22);
# for row in financial_order:
#     financial_df.loc[row,'numeach']=(tempcol==row).sum()/numpt*100
    
# ax.bar(financial_df.Min,financial_df.numeach,financial_df.Max-financial_df.Min,
#        edgecolor='gray',color='b',alpha=0.5,align='edge')

# ax.bar(msq_income['min'],msq_income[2022],msq_income['max']-msq_income['min'],
#        edgecolor='gray',color='r',align='edge',alpha=0.5)



# os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
# if savefig: fig.savefig('Figures/Fig_IncomeData_v2.png',dpi=600);
# os.chdir(homedir)

# =============================================================================
# Collate Income Data in 2022 - v2 with overlapping bars
# =============================================================================
# get MSQ data first, from 2022
refdir='/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/ReferenceData';
os.chdir(refdir)
msq_income=pd.read_excel('MSQ_Table6_Q24_2022.xlsx')
msq_income['Avg'] = (msq_income['min']+msq_income['max'])/2
os.chdir(homedir)

fig,ax = plt.subplots(figsize=(10,6))

financial_order=["Less than $10,000",
                 "$10,000 - $19,999",
                 "$20,000 - $29,999",
                 "$30,000 - $39,999",
                 "$40,000 - $49,999",
                 "$50,000 - $59,999",
                 "$60,000 - $69,999",
                 "$70,000 - $79,999",
                 "$80,000 - $89,999",
                 "$90,000 - $99,999",
                 "$100,000 - $149,999",
                 "More than $150,000"];

financial_min=[0,
               10000,
               20000,
               30000,
               40000,
               50000,
               60000,
               70000,
               80000,
               90000,
               100000,
               150000,]
financial_max=[10000,
               20000,
               30000,
               40000,
               50000,
               60000,
               70000,
               80000,
               90000,
               100000,
               150000,
               200000]

financial_df=pd.DataFrame(data=None,index=financial_order,columns=['numeach','Min','Max']);
financial_df['Min']=financial_min;
financial_df['Max']=financial_max;
financial_df['Avg']=(financial_df['Max']+financial_df['Min'])/2;

tempcol=pre22["What is your family's approximate yearly income (in US Dolllars)?"];
numpt=len(pre22);
for row in financial_order:
    financial_df.loc[row,'numeach']=(tempcol==row).sum()/numpt*100

aligned_mins=[0, 50000,100000,150000];
aligned_max=[50000,100000,150000,600000];

aligned_df=pd.DataFrame(data=None,columns=['Min','Max','FLNSUS','National'])
aligned_df['Min']=aligned_mins
aligned_df['Max']=aligned_max
aligned_df['order']=["Less than $50,000",
                 "\$50,000 - $99,999",
                 "\$100,000 - $149,999",
                 "More than $150,000"];
for i in aligned_df.index:
    
    flnsus_binary=(aligned_df.loc[i,'Min'] <= financial_df['Avg'].astype(int)) & (financial_df['Avg'].astype(int) <= aligned_df.loc[i,'Max']);
    aligned_df.loc[i,'FLNSUS']=financial_df.loc[flnsus_binary,'numeach'].sum()
    national_binary=(aligned_df.loc[i,'Min'] <= msq_income['Avg'].astype(int)) & (msq_income['Avg'].astype(int) <= aligned_df.loc[i,'Max']);
    aligned_df.loc[i,'National']=msq_income.loc[national_binary,2022].sum();

financial_df=aligned_df.melt(id_vars=['order'],value_vars=['FLNSUS','National'])

s=sns.barplot(data=financial_df,y='order',x='value',hue='variable',ax=ax)
ax.set_ylabel('')
ax.set_xlabel('Percent of Participants')
sns.move_legend(s, loc='center right', title=None)

plt.tight_layout()
os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
if savefig: fig.savefig('Figures/Fig_IncomeData_v3.png',dpi=600);
os.chdir(homedir)

## Do stats for chi2
# https://rowannicholls.github.io/python/statistics/hypothesis_testing/chi_squared.html#independence-test
#


from scipy.stats import chisquare
from scipy.stats import chi2
n_National=9278;# from the MSQ data
n_FLNSUS=len(pre22);

obs_National=np.ceil(aligned_df['National'].values*n_National/100).astype(int);
obs_FLNSUS=np.ceil(aligned_df['FLNSUS'].values*n_FLNSUS/100).astype(int);
observations = np.array([obs_National,obs_FLNSUS])
row_totals = np.array([np.sum(observations, axis=1)])
col_totals = np.array([np.sum(observations, axis=0)])
n = np.sum(observations)

# Calculate the expected observations
expected = np.dot(row_totals.T, col_totals) / n
# Calculate the chi-square test statistic
chisq, p = chisquare(observations, expected)
chisq = np.sum(chisq)
# Degrees of freedom
rows = observations.shape[0]
cols = observations.shape[1]
df = (rows - 1) * (cols - 1)
# Convert chi-square test statistic to p-value
p = 1 - chi2.cdf(chisq, df)
print(f'p = {p:3.5f}')

# =============================================================================
# Pre-Post 2021
# =============================================================================
col_perceptions=[col for col in post21 if col.startswith('Select your level of agreement for the following statements - ')]
col_names=[i.split(' - ', 1)[1] for i in col_perceptions]

col_order=['I will get into medical school',
           'I will become a doctor',
           'I can become a neurosurgeon',
           'I have the ability to shadow neurosurgical procedures',
           'I am familiar with the career pathway to become a neurosurgeon',
           'I have the institutional support and resources to become a neurosurgeon',
           'I am connected to mentors that can help me become a neurosurgeon',
           'I know the day-to-day responsibilities of a neurosurgeon',
           'I can list at least three subspecialties of neurosurgery',
           'Neurosurgery is a good field for minorities and women',
           'I have seen or met a Woman neurosurgeon',
           'I have seen or met a Black neurosurgeon',
           'I have seen or met a Latinx neurosurgeon',
           'Neurosurgeons are intimidating',
           'Neurosurgeons have a good work-life balance',
           'Neurosurgeons have reasonable work hours',
           "Neurosurgeons improve their patients' quality of life"];

df_pre=pre21;
df_post=post21;

uid_pre=set(df_pre['Unique ID']);
uid_post=set(df_post['Unique ID']);

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

fig, ax=plt.subplots(figsize=(12,5),ncols=1,nrows=1,);
bonf=1;

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
        pcolor='orange'
    elif stats['p-val'][0]<0.05/bonf:
        pcolor='green'
    else:
        pcolor='grey'
    
    ax.plot(np.mean(post.loc[:,col]),idx,'o',color=pcolor);
    ax.text(5.1,idx,"{0:.3f}".format(stats['p-val'][0]),
            verticalalignment='center',color=pcolor)

ax.set_yticks(np.arange(0,len(col_names)));
ax.set_yticklabels(col_names);
ax.set_xticks(np.arange(1,6));
ax.set_xticklabels(['Strongly\ndisagree','Somewhat\ndisagree',
                    'Neither agree\nnor disagree','Somewhat\nagree',
                    'Strongly\nagree'])    
ax.grid(axis = 'x',linewidth=0.5)
ax.grid(axis = 'y',linewidth=0.5)        

ax.set_title('FLNSUS 2021 Pre/Post Data')

plt.tight_layout()
os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
if savefig: fig.savefig('Figures/Fig_Wilcoxon_2021_v2.png',dpi=600);
os.chdir(homedir)
# =============================================================================
# Pre-Post 2022
# =============================================================================
col_perceptions=[col for col in post21 if col.startswith('Select your level of agreement for the following statements - ')]
col_names=[i.split(' - ', 1)[1] for i in col_perceptions]

col_order=['I will get into medical school',
           'I will become a doctor',
           'I can become a neurosurgeon',
           'I have the ability to shadow neurosurgical procedures',
           'I am familiar with the career pathway to become a neurosurgeon',
           'I have the institutional support and resources to become a neurosurgeon',
           'I am connected to mentors that can help me become a neurosurgeon',
           'I know the day-to-day responsibilities of a neurosurgeon',
           'I can list at least three subspecialties of neurosurgery',
           'Neurosurgery is a good field for minorities and women',
           'I have seen or met a Woman neurosurgeon',
           'I have seen or met a Black neurosurgeon',
           'I have seen or met a Latinx neurosurgeon',
           'Neurosurgeons are intimidating',
           'Neurosurgeons have a good work-life balance',
           'Neurosurgeons have reasonable work hours',
           "Neurosurgeons improve their patients' quality of life"];


df_pre=pre22;
df_post=post22;

uid_pre=set(df_pre['Unique ID']);
uid_post=set(df_post['Unique ID']);

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

fig, ax=plt.subplots(figsize=(12,5),ncols=1,nrows=1,);
bonf=1;

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
        pcolor='orange'
    elif stats['p-val'][0]<0.05/bonf:
        pcolor='green'
    else:
        pcolor='grey'
    
    ax.plot(np.mean(post.loc[:,col]),idx,'o',color=pcolor);
    ax.text(5.1,idx,"{0:.3f}".format(stats['p-val'][0]),
            verticalalignment='center',color=pcolor)

ax.set_yticks(np.arange(0,len(col_names)));
ax.set_yticklabels(col_names);
ax.set_xticks(np.arange(1,6));
ax.set_xticklabels(['Strongly\ndisagree','Somewhat\ndisagree',
                    'Neither agree\nnor disagree','Somewhat\nagree',
                    'Strongly\nagree'])    
ax.grid(axis = 'x',linewidth=0.5)
ax.grid(axis = 'y',linewidth=0.5)        

ax.set_title('FLNSUS 2022 Pre/Post Data')

plt.tight_layout()
os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
if savefig: fig.savefig('Figures/Fig_Wilcoxon_2022_v2.png',dpi=600);
os.chdir(homedir)

# =============================================================================
# Pre-Post 2021 and 2022
# =============================================================================
col_perceptions=[col for col in post21 if col.startswith('Select your level of agreement for the following statements - ')]
col_names=[i.split(' - ', 1)[1] for i in col_perceptions]

col_order=['I will get into medical school',
           'I will become a doctor',
           'I can become a neurosurgeon',
           'I have the ability to shadow neurosurgical procedures',
           'I am familiar with the career pathway to become a neurosurgeon',
           'I have the institutional support and resources to become a neurosurgeon',
           'I am connected to mentors that can help me become a neurosurgeon',
           'I know the day-to-day responsibilities of a neurosurgeon',
           'I can list at least three subspecialties of neurosurgery',
           'Neurosurgery is a good field for minorities and women',
           'I have seen or met a Woman neurosurgeon',
           'I have seen or met a Black neurosurgeon',
           'I have seen or met a Latinx neurosurgeon',
           'Neurosurgeons are intimidating',
           'Neurosurgeons have a good work-life balance',
           'Neurosurgeons have reasonable work hours',
           "Neurosurgeons improve their patients' quality of life"];

# 2021 first
df_pre=pre21;
df_post=post21;

uid_pre=set(df_pre['Unique ID']);
uid_post=set(df_post['Unique ID']);

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

fig, ax=plt.subplots(figsize=(15,5),ncols=1,nrows=1,);
bonf=1;

for idx,col in enumerate(col_perceptions):
    stats=pg.wilcoxon(pre.loc[:,col],
                      post.loc[:,col], 
                      alternative='two-sided')

    # print(col_names[idx]);
    # print('    p-val = ',stats['p-val'].values[0])
   
    ax.plot(np.mean(pre.loc[:,col]),idx+0.2,'xk');
    ax.plot([np.mean(pre.loc[:,col]),
                     np.mean(post.loc[:,col])],[idx+0.2,idx+0.2],'-',color='k');
    
    if stats['p-val'][0]<0.001/bonf:
        pcolor='red'
    elif stats['p-val'][0]<0.01/bonf:
        pcolor='orange'
    elif stats['p-val'][0]<0.05/bonf:
        pcolor='green'
    else:
        pcolor='grey'
    
    ax.plot(np.mean(post.loc[:,col]),idx+0.2,'o',color=pcolor);
    ax.text(5.1,idx,"{0:.3f}".format(stats['p-val'][0]),
            verticalalignment='center',color=pcolor)


df_pre=pre22;
df_post=post22;

uid_pre=set(df_pre['Unique ID']);
uid_post=set(df_post['Unique ID']);

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

# fig, ax=plt.subplots(figsize=(12,5),ncols=1,nrows=1,);
bonf=1;

for idx,col in enumerate(col_perceptions):
    stats=pg.wilcoxon(pre.loc[:,col],
                      post.loc[:,col], 
                      alternative='two-sided')

    # print(col_names[idx]);
    # print('    p-val = ',stats['p-val'].values[0])
   
    ax.plot(np.mean(pre.loc[:,col]),idx-0.2,'xk');
    ax.plot([np.mean(pre.loc[:,col]),
                     np.mean(post.loc[:,col])],[idx-0.2,idx-0.2],'-',color='b');
    
    if stats['p-val'][0]<0.001/bonf:
        pcolor='red'
    elif stats['p-val'][0]<0.01/bonf:
        pcolor='orange'
    elif stats['p-val'][0]<0.05/bonf:
        pcolor='green'
    else:
        pcolor='grey'
    
    ax.plot(np.mean(post.loc[:,col]),idx-0.2,'o',color=pcolor);
    ax.text(5.7,idx,"{0:.3f}".format(stats['p-val'][0]),
            verticalalignment='center',color=pcolor)

ax.text(5.1,idx+1,'2021',color='k',fontweight='bold')
ax.text(5.7,idx+1,'2022',color='b',fontweight='bold')

ax.set_yticks(np.arange(0,len(col_names)));
ax.set_yticklabels(col_names);
ax.set_xticks(np.arange(1,6));
ax.set_xticklabels(['Strongly\ndisagree','Somewhat\ndisagree',
                    'Neither agree\nnor disagree','Somewhat\nagree',
                    'Strongly\nagree'])    
ax.grid(axis = 'x',linewidth=0.5)
ax.grid(axis = 'y',linewidth=0.5)        

ax.set_title('FLNSUS 2021 and 2022 Pre/Post Data')

plt.tight_layout()
os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
if savefig: fig.savefig('Figures/Fig_Wilcoxon_2021_2022_v1.png',dpi=600);
os.chdir(homedir)

sys.exit()

# =============================================================================
# Post 2021 --> Check in 2023; check patency
# =============================================================================
col_perceptions=[col for col in post21 if col.startswith('Select your level of agreement for the following statements - ')]
col_names=[i.split(' - ', 1)[1] for i in col_perceptions]

df_pre=post21;
df_post=mid23;

uid_pre=set(df_pre['Unique ID']);
uid_post=set(df_post['Unique ID']);

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

fig, ax=plt.subplots(figsize=(12,5),ncols=1,nrows=1,);
bonf=1;

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
        pcolor='orange'
    elif stats['p-val'][0]<0.05/bonf:
        pcolor='green'
    else:
        pcolor='grey'
    
    ax.plot(np.mean(post.loc[:,col]),idx,'o',color=pcolor);
    ax.text(5.1,idx,"{0:.3f}".format(stats['p-val'][0]),
            verticalalignment='center',color=pcolor)

ax.set_yticks(np.arange(0,len(col_names)));
ax.set_yticklabels(col_names);
ax.set_xticks(np.arange(1,6));
ax.set_xticklabels(['Strongly\ndisagree','Somewhat\ndisagree',
                    'Neither agree\nnor disagree','Somewhat\nagree',
                    'Strongly\nagree'])    
ax.grid(axis = 'x',linewidth=0.5)
ax.grid(axis = 'y',linewidth=0.5)        

ax.set_title('Post 21 --> Mid 2023')

plt.tight_layout()
os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
if savefig: fig.savefig('Figures/Fig_Wilcoxon_post21_mid23.png',dpi=600);
os.chdir(homedir)

# =============================================================================
# Post 2022 --> Check in 2023; check patency
# =============================================================================
col_perceptions=[col for col in post21 if col.startswith('Select your level of agreement for the following statements - ')]
col_names=[i.split(' - ', 1)[1] for i in col_perceptions]

df_pre=post22;
df_post=mid23;

uid_pre=set(df_pre['Unique ID']);
uid_post=set(df_post['Unique ID']);

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

fig, ax=plt.subplots(figsize=(12,5),ncols=1,nrows=1,);
bonf=1;

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
        pcolor='orange'
    elif stats['p-val'][0]<0.05/bonf:
        pcolor='green'
    else:
        pcolor='grey'
    
    ax.plot(np.mean(post.loc[:,col]),idx,'o',color=pcolor);
    ax.text(5.1,idx,"{0:.3f}".format(stats['p-val'][0]),
            verticalalignment='center',color=pcolor)

ax.set_yticks(np.arange(0,len(col_names)));
ax.set_yticklabels(col_names);
ax.set_xticks(np.arange(1,6));
ax.set_xticklabels(['Strongly\ndisagree','Somewhat\ndisagree',
                    'Neither agree\nnor disagree','Somewhat\nagree',
                    'Strongly\nagree'])    
ax.grid(axis = 'x',linewidth=0.5)
ax.grid(axis = 'y',linewidth=0.5)        

ax.set_title('Post 22 --> Mid 2023')

plt.tight_layout()
os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
if savefig: fig.savefig('Figures/Fig_Wilcoxon_post22_mid23.png',dpi=600);
os.chdir(homedir)

# =============================================================================
# Compare across time - newest version, June 11
# =============================================================================
# https://stats.stackexchange.com/questions/584656/confidence-intervals-for-likert-items
# http://rstudio-pubs-static.s3.amazonaws.com/300786_136029ae2bce4ab2a40caaef34ed62c0.html
# https://github.com/nmalkin/plot-likert/blob/master/docs/lots_of_random_figures.ipynb
# http://faculty.nps.edu/rdfricke/OA4109/Lecture%209-1%20--%20Introduction%20to%20Survey%20Analysis.pdf


uid_all=idfile['Unique ID'];
col_idx=['FLNSUS 21 Pre',
          'FLNSUS 21 Post',
          'FLNSUS 22 Pre',
          'FLNSUS 22 Post',
          'FLXSUS 23 Midterm Jan'];

col_perceptions=[col for col in post21 if col.startswith('Select your level of agreement for the following statements - ')]
# using 2021 so that have values throughout all FLNSUS surveys
col_names=[i.split(' - ', 1)[1] for i in col_perceptions]


for i,col in enumerate(col_perceptions):# iterate through questions

    df_col=pd.DataFrame(data=np.nan,index=uid_all,columns=dfname)
    
    for j, uid in enumerate(uid_all):# iterate through people
        
        for k, df_ in enumerate(dflist):#iterate through years/survey
             
            if idfile.loc[idfile['Unique ID']==uid,col_idx[k]].values[0]:
                 
                if len(df_.loc[df_['Unique ID']==uid,col].values)>0:
                    df_col.loc[uid,dfname[k]]=df_.loc[df_['Unique ID']==uid,col].values[0]
    
    df_col=df_col.replace({'Strongly agree': 5, 
                    'Somewhat agree': 4,
                    'Neither agree nor disagree': 3,
                    'Somewhat disagree': 2,
                    'Strongly disagree': 1,})
    
    
    # sys.exit()
    ## only people at all years

    all_years=df_col
    # all_years=df_col.dropna()
    all_years=all_years.melt(ignore_index=False);
    all_years['uid']=all_years.index;
    all_years=all_years.reset_index();
    
    stats=pg.friedman(data=df_col)
    print(col_names[i])
    print(stats)
    
    fig, ax=plt.subplots(figsize=(10,6))
    
    sns.lineplot(data=all_years,
                  x='variable',
                  y='value',
                  # hue='uid',
                  legend=False,
                  ax=ax)
    ax.set_yticks([1,2,3,4,5]);
    
    ax.set_yticklabels(['Strongly disagree',
                          'Somewhat Disagree',
                          'Neither agree nor disagree',
                          'Somewhat agree',
                          'Strongly agree']);
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(col_names[i])
    
    
    if stats['p-unc'].values[0] <0.05:

        
        stats2=pg.pairwise_tests(data=all_years,dv='value',between='variable',parametric=False)
        
        
    
        
        
        os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
        stats2.to_csv("Figures/Perceptions_lineplot_withstats("+col_names[i]+").csv")
        print(col_names[i])
        os.chdir(homedir)
        
        # sys.exit()
    os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
    fig.savefig("Figures/Perceptions_lineplot_withstats_v2("+col_names[i]+") (p_friedman = %.4f).png" % stats['p-unc'].values[0],dpi=600);
    os.chdir(homedir)
        
    # sys.exit()
    
    # os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
    # fig.savefig("Figures/Perceptions_lineplot_withstats("+col_names[i]+").png",dpi=600);
    # os.chdir(homedir)
    
# os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
# fig.savefig("Figures/Perceptions_lineplot_average_all.png",dpi=600);
# os.chdir(homedir)



# =============================================================================
# Compare across time
# =============================================================================
# https://stats.stackexchange.com/questions/584656/confidence-intervals-for-likert-items
# http://rstudio-pubs-static.s3.amazonaws.com/300786_136029ae2bce4ab2a40caaef34ed62c0.html
# https://github.com/nmalkin/plot-likert/blob/master/docs/lots_of_random_figures.ipynb
# http://faculty.nps.edu/rdfricke/OA4109/Lecture%209-1%20--%20Introduction%20to%20Survey%20Analysis.pdf


uid_all=idfile['Unique ID'];
col_idx=['FLNSUS 21 Pre',
          'FLNSUS 21 Post',
          'FLNSUS 22 Pre',
          'FLNSUS 22 Post',
          'FLXSUS 23 Midterm Jan'];

col_perceptions=[col for col in post21 if col.startswith('Select your level of agreement for the following statements - ')]
# using 2021 so that have values throughout all FLNSUS surveys
col_names=[i.split(' - ', 1)[1] for i in col_perceptions]

fig, axs=plt.subplots(figsize=(37,20,),nrows=4,ncols=5,sharex=True,sharey=True);
from itertools import chain
l = chain.from_iterable(zip(*axs))
l=list(l)


for i,col in enumerate(col_perceptions):# iterate through questions

    df_col=pd.DataFrame(data=np.nan,index=uid_all,columns=dfname)
    
    for j, uid in enumerate(uid_all):# iterate through people
        
        for k, df_ in enumerate(dflist):#iterate through years/survey
            
            if idfile.loc[idfile['Unique ID']==uid,col_idx[k]].values[0]:
                
                if len(df_.loc[df_['Unique ID']==uid,col].values)>0:
                    df_col.loc[uid,dfname[k]]=df_.loc[df_['Unique ID']==uid,col].values[0]
    
    df_col=df_col.replace({'Strongly agree': 5, 
                    'Somewhat agree': 4,
                    'Neither agree nor disagree': 3,
                    'Somewhat disagree': 2,
                    'Strongly disagree': 1,})
    
    
    # sys.exit()
    ## only people at all years
    # fig, ax=plt.subplots(figsize=(10,6))
    all_years=df_col
    # all_years=df_col.dropna()
    all_years=all_years.melt(ignore_index=False);
    all_years['uid']=all_years.index;
    all_years=all_years.reset_index();
    
    ax=l[i]
    
    sns.lineplot(data=all_years,
                  x='variable',
                  y='value',
                  # hue='uid',
                  legend=False,
                  ax=ax)
    ax.set_yticks([1,2,3,4,5]);
    
    ax.set_yticklabels(['Strongly disagree',
                          'Somewhat Disagree',
                          'Neither agree nor disagree',
                          'Somewhat agree',
                          'Strongly agree']);
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(col_names[i])
    
os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
fig.savefig("Figures/Perceptions_lineplot_average_all.png",dpi=600);
os.chdir(homedir)

# =============================================================================
# Look at data over all years
# =============================================================================

uid_all=idfile['Unique ID'];
col_idx=['FLNSUS 21 Pre',
          'FLNSUS 21 Post',
          'FLNSUS 22 Pre',
          'FLNSUS 22 Post',
          'FLXSUS 23 Midterm Jan'];

col_perceptions=[col for col in post21 if col.startswith('Select your level of agreement for the following statements - ')]
# using 2021 so that have values throughout all FLNSUS surveys
col_names=[i.split(' - ', 1)[1] for i in col_perceptions]

for i,col in enumerate(col_perceptions):# iterate through questions
    
    

    df_col=pd.DataFrame(data=np.nan,index=uid_all,columns=dfname)
    
    for j, uid in enumerate(uid_all):# iterate through people
        
        for k, df_ in enumerate(dflist):#iterate through years/survey
            
            if idfile.loc[idfile['Unique ID']==uid,col_idx[k]].values[0]:
                
                if len(df_.loc[df_['Unique ID']==uid,col].values)>0:
                    df_col.loc[uid,dfname[k]]=df_.loc[df_['Unique ID']==uid,col].values[0]
    
    df_col=df_col.replace({'Strongly agree': 5, 
                    'Somewhat agree': 4,
                    'Neither agree nor disagree': 3,
                    'Somewhat disagree': 2,
                    'Strongly disagree': 1,})
    
    stats=pg.friedman(data=df_col)
    if stats['p-unc'].values[0] <0.05:

        all_years=df_col
        # all_years=df_col.dropna()
        all_years=all_years.melt(ignore_index=False);
        all_years['uid']=all_years.index;
        all_years=all_years.reset_index();
        
        stats2=pg.pairwise_tests(data=all_years,dv='value',between='variable',parametric=False)
        sys.exit()

    

    
    
    
    sys.exit()
    ## only people at all years
    fig, ax=plt.subplots(figsize=(10,6))

    
    sns.lineplot(data=all_years,
                  x='variable',
                  y='value',
                  hue='uid',
                  legend=False,
                  ax=ax)
    ax.set_yticks([1,2,3,4,5]);
    
    ax.set_yticklabels(['Strongly disagree',
                          'Somewhat Disagree',
                          'Neither agree nor disagree',
                          'Somewhat agree',
                          'Strongly agree']);
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(col_names[i])
    
    os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
    fig.savefig("Figures/Perceptions_lineplot("+col_names[i]+").png",dpi=600);
    os.chdir(homedir)
    
    ## One way to make a figure
    # df_col_melt=df_col.melt(ignore_index=True);
    
    # fig, ax=plt.subplots(figsize=(10,6))
    
    # sns.histplot(data=df_col_melt,
    #              x='variable',
    #              y='value',
    #              common_norm=True,
    #              cbar=True,
    #              ax=ax,
    #              bins=[0.5,1.5,2.5,3.5,4.5,5.5])
    # # ax.set_box_aspect(1)
    
    # ax.set_xticklabels(['presurvey\n2021',
    #                     'postsurvey\n2021',
    #                     'presurvey\n2022',
    #                     'postsurvey\n2022',
    #                     'mid-year check-in\n2023']);
    
    # ax.set_yticklabels(['','Strongly disagree',
    #                     'Somewhat Disagree',
    #                     'Neither agree nor disagree',
    #                     'Somewhat agree',
    #                     'Strongly agree']);
    # ax.set_xlabel("")
    # ax.set_ylabel("")
    # ax.set_title(col_names[i])
    
    # if savefig: fig.savefig("Figures/Perceptions("+col_names[i]+").png",dpi=600);
    

# # =============================================================================
# # Look at data over all years
# # =============================================================================


# uid_all=idfile['Unique ID'];
# col_idx=['FLNSUS 21 Pre',
#          'FLNSUS 21 Post',
#          'FLNSUS 22 Pre',
#          'FLNSUS 22 Post',
#          'FLXSUS 23 Midterm Jan'];

# col_perceptions=[col for col in post21 if col.startswith('Select your level of agreement for the following statements - ')]
# # using 2021 so that have values throughout all FLNSUS surveys
# col_names=[i.split(' - ', 1)[1] for i in col_perceptions]

# for i,col in enumerate(col_perceptions):# iterate through questions
    
    

#     df_col=pd.DataFrame(data=np.nan,index=uid_all,columns=dfname)
    
#     for j, uid in enumerate(uid_all):# iterate through people
        
#         for k, df_ in enumerate(dflist):#iterate through years/survey
            
#             if idfile.loc[idfile['Unique ID']==uid,col_idx[k]].values[0]:
                
#                 if len(df_.loc[df_['Unique ID']==uid,col].values)>0:
#                     df_col.loc[uid,dfname[k]]=df_.loc[df_['Unique ID']==uid,col].values[0]
    
#     df_col=df_col.replace({'Strongly agree': 5, 
#                     'Somewhat agree': 4,
#                     'Neither agree nor disagree': 3,
#                     'Somewhat disagree': 2,
#                     'Strongly disagree': 1,})
    
#     ## only people at all years
#     fig, ax=plt.subplots(figsize=(10,6))
#     all_years=df_col
#     # all_years=df_col.dropna()
#     all_years=all_years.melt(ignore_index=False);
#     all_years['uid']=all_years.index;
#     all_years=all_years.reset_index();
    
#     sns.lineplot(data=all_years,
#                  x='variable',
#                  y='value',
#                  hue='uid',
#                  legend=False,
#                  ax=ax)
#     ax.set_yticks([1,2,3,4,5]);
    
#     ax.set_yticklabels(['Strongly disagree',
#                           'Somewhat Disagree',
#                           'Neither agree nor disagree',
#                           'Somewhat agree',
#                           'Strongly agree']);
#     ax.set_xlabel("")
#     ax.set_ylabel("")
#     ax.set_title(col_names[i])
    
#     os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
#     fig.savefig("Figures/Perceptions_lineplot("+col_names[i]+").png",dpi=600);
#     os.chdir(homedir)
    
#     ## One way to make a figure
#     # df_col_melt=df_col.melt(ignore_index=True);
    
#     # fig, ax=plt.subplots(figsize=(10,6))
    
#     # sns.histplot(data=df_col_melt,
#     #              x='variable',
#     #              y='value',
#     #              common_norm=True,
#     #              cbar=True,
#     #              ax=ax,
#     #              bins=[0.5,1.5,2.5,3.5,4.5,5.5])
#     # # ax.set_box_aspect(1)
    
#     # ax.set_xticklabels(['presurvey\n2021',
#     #                     'postsurvey\n2021',
#     #                     'presurvey\n2022',
#     #                     'postsurvey\n2022',
#     #                     'mid-year check-in\n2023']);
    
#     # ax.set_yticklabels(['','Strongly disagree',
#     #                     'Somewhat Disagree',
#     #                     'Neither agree nor disagree',
#     #                     'Somewhat agree',
#     #                     'Strongly agree']);
#     # ax.set_xlabel("")
#     # ax.set_ylabel("")
#     # ax.set_title(col_names[i])
    
#     # if savefig: fig.savefig("Figures/Perceptions("+col_names[i]+").png",dpi=600);
    
    
# =============================================================================
# look at rating of symposium
# =============================================================================

mid23_replaced=mid23.replace({'Strongly agree': 5, 
                               'Somewhat agree': 4,
                               'Neither agree nor disagree': 3,
                               'Somewhat disagree': 2,
                               'Strongly disagree': 1,})

fig, ax=plt.subplots(figsize=(8,6))
sns.histplot(data=mid23_replaced,
             y='What factors have influenced your career goals? - FLNSUS 2021 and/or 2022',
             hue='Which FLNSUS year did you attend?',
             multiple='stack',
             bins=[0.5,1.5,2.5,3.5,4.5,5.5],
             ax=ax)
ax.set_yticks([1,2,3,4,5]);
 
ax.set_yticklabels(['Strongly disagree',
                       'Somewhat Disagree',
                       'Neither agree \nnor disagree',
                       'Somewhat agree',
                       'Strongly agree']);

ax.set_ylabel("")
ax.set_title("Did FLNSUS 2021 or 2022 Influence Your Career Goals?")
os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
if savefig: fig.savefig("Figures/FLNSUS_impact.png",dpi=600);
os.chdir(homedir)

# =============================================================================
# Look at ratings of symposia
# =============================================================================
rating_21=post21['Symposium Rating'];
rating_22=post22['Symposium Rating'];
rating_21=pd.DataFrame(data=rating_21,columns=['Symposium Rating']);
rating_21['Symposium']='FLNSUS 2021'
rating_22=pd.DataFrame(data=rating_22,columns=['Symposium Rating']);
rating_22['Symposium']='FLNSUS 2022'

ratings=pd.concat([rating_21,rating_22]).reset_index();

fig, ax=plt.subplots(figsize=(8,6))
sns.histplot(data=ratings,
             x='Symposium Rating',
             hue='Symposium',
             multiple='stack',
             bins=np.arange(0,11)-0.5,
             ax=ax)

os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
if savefig: fig.savefig("Figures/Fig_rating.png",dpi=600);
os.chdir(homedir)

# =============================================================================
# Try NLP
# =============================================================================
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
# nltk.download('all')# only needed first time

mid23_temp=mid23[['Unique ID','Text - How has FLNSUS impacted your life?']]
mid23_temp=mid23_temp.dropna()
mid23_temp=mid23_temp.reset_index()

def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())

    # Remove stop words
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Join the tokens back into a string
    processed_text = ' '.join(lemmatized_tokens)

    return processed_text


mid23_temp['proc_text']=np.nan
for i in range(len(mid23_temp)):
    mid23_temp.loc[i,'proc_text']=preprocess_text(mid23_temp.loc[i,'Text - How has FLNSUS impacted your life?'])


analyzer = SentimentIntensityAnalyzer()

mid23_temp['sentiment']=np.nan
for i in range(len(mid23_temp)):
    scores = analyzer.polarity_scores(mid23_temp.loc[i,'proc_text'])
    
    mid23_temp.loc[i,'sentiment']=scores['compound']
    
fig, ax=plt.subplots(figsize=(8,6))
sns.histplot(data=mid23_temp,
             x='sentiment',
             bins=np.arange(-1,1.2,0.2),ax=ax)

os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
if savefig: fig.savefig("Figures/Fig_sentiment.png",dpi=600);
os.chdir(homedir)