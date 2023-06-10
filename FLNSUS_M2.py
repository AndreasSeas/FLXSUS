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
# also have openpyxl

# =============================================================================
# Set init parameters
# =============================================================================
savefig=False

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
# sys.exit()
# =============================================================================
# Map Figure - create a map showing all the different participants
# =============================================================================
fig, ax=plt.subplots(figsize=(16,6),ncols=1,nrows=1,)

# m = Basemap(width=12000000,height=9000000,projection='lcc',
            # resolution='l',lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.)
# m = Basemap(projection='ortho',lon_0=-105,lat_0=40,resolution='c')

m = Basemap(projection='robin',lon_0=0,resolution='c')
m.drawcoastlines()
m.drawcountries()
m.fillcontinents(color=[0.8,0.8,0.8],lake_color='aqua')
m.drawmapboundary(fill_color='aqua')

# markerlist=['xr','.r','xb','.b','og'];
markerlist=['x','o','x','o','.'];
colorlist=['r','r','b','b','k'];

for i,df_ in enumerate(dflist):
    print(df_.shape)
    x_,y_=m(df_.loc[:,'Longitude'],df_.loc[:,'Latitude']);
    # ax.plot(x_,y_,markerlist[i],label=dfname[i],);
    ax.scatter(x_,y_,50,colorlist[i],markerlist[i],alpha=0.3,label=dfname[i],)

ax.legend()
os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
if savefig: fig.savefig('Figures/Fig_map_v1.png',dpi=600);
os.chdir(homedir)
# =============================================================================
# Collate Race Data Across years
# =============================================================================
col_race=[col for col in idfile if col.startswith('Race')]

df_race=pd.DataFrame(data=None,index=dfname, columns=col_race)
df_race_pct=pd.DataFrame(data=None,index=dfname, columns=col_race)

for i,df_ in enumerate(dflist):
    
    uid=df_.loc[:,'Unique ID']
    temp_i=idfile.loc[:,'Unique ID'].isin(uid)
    numpt=len(uid)
    
    for j,col in enumerate(col_race):
        
        df_race.iloc[i,j]=(idfile.loc[temp_i,col]==True).sum()
        df_race_pct.iloc[i,j]=(idfile.loc[temp_i,col]==True).sum()/numpt

df_race.insert(0, 'survey', 0)
df_race.loc[:,'survey']=df_race.index;
df_race_melt=df_race.melt(id_vars='survey');
df_race_melt['variable']=df_race_melt['variable'].str.replace('Race - ','')

fig,ax = plt.subplots(figsize=(10,6))
s=sns.histplot(data=df_race_melt,
               x='survey',
               weights='value',
               hue='variable',
               multiple='stack',
               stat='count',
               ax=ax,
               palette='colorblind')

sns.move_legend(s, 
                "lower center", 
                bbox_to_anchor=(0.5,1),
                ncol=3,
                title=None)

ax.set_xticklabels(['presurvey\n2021',
                    'postsurvey\n2021',
                    'presurvey\n2022',
                    'postsurvey\n2022',
                    'mid-year check-in\n2023'])
os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
if savefig: fig.savefig('Figures/Fig_race_v1.png',dpi=600);
os.chdir(homedir)
# =============================================================================
# Collate Ethnicity Data Across years
# =============================================================================

df_ethnicity=pd.DataFrame(data=None,columns=['survey','Ethnicity']);

for i,df_ in enumerate(dflist):
    
    uid=df_.loc[:,'Unique ID']
    temp_i=idfile.loc[:,'Unique ID'].isin(uid)
    temp_df=pd.DataFrame(data=None,columns=['survey','Ethnicity'])
    temp_df.Ethnicity=idfile.loc[temp_i,'Ethnicity']
    temp_df.survey=dfname[i]
    df_ethnicity=pd.concat([df_ethnicity,temp_df])

df_ethnicity=df_ethnicity.reset_index()

fig,ax = plt.subplots(figsize=(10,6))
s=sns.histplot(data=df_ethnicity,
             x='survey',
             hue='Ethnicity',
             stat='count',
             multiple='stack',
             palette='colorblind')

sns.move_legend(s, 
                "lower center", 
                bbox_to_anchor=(0.5,1),
                ncol=3,
                title=None)

ax.set_xticklabels(['presurvey\n2021',
                    'postsurvey\n2021',
                    'presurvey\n2022',
                    'postsurvey\n2022',
                    'mid-year check-in\n2023']);

os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
if savefig: fig.savefig('Figures/Fig_ethnicity_v1.png',dpi=600);
os.chdir(homedir)

# =============================================================================
# Collate Gender Data Across years
# =============================================================================

df_gender=pd.DataFrame(data=None,columns=['survey','Gender']);

for i,df_ in enumerate(dflist):
    
    uid=df_.loc[:,'Unique ID']
    temp_i=idfile.loc[:,'Unique ID'].isin(uid)
    temp_df=pd.DataFrame(data=None,columns=['survey','Gender'])
    temp_df.Gender=idfile.loc[temp_i,'Gender']
    temp_df.survey=dfname[i]
    df_gender=pd.concat([df_gender,temp_df])

df_gender=df_gender.reset_index()

fig,ax = plt.subplots(figsize=(10,6))
s=sns.histplot(data=df_gender,
             x='survey',
             hue='Gender',
             stat='count',#
             multiple='stack',
             palette='colorblind',)

sns.move_legend(s, 
                "lower center", 
                bbox_to_anchor=(0.5,1),
                ncol=3,
                title=None)

ax.set_xticklabels(['presurvey\n2021',
                    'postsurvey\n2021',
                    'presurvey\n2022',
                    'postsurvey\n2022',
                    'mid-year check-in\n2023']);
os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
if savefig: fig.savefig('Figures/Fig_gender_v1.png',dpi=600);
os.chdir(homedir)

# =============================================================================
# Collate Sexual Orientation Data Across years
# =============================================================================

df_orientation=pd.DataFrame(data=None,columns=['survey','Sexual Orientation']);


for i,df_ in enumerate(dflist):
    
    uid=df_.loc[:,'Unique ID']
    temp_i=idfile.loc[:,'Unique ID'].isin(uid)
    temp_df=pd.DataFrame(data=None,columns=['survey','Sexual Orientation'])
    temp_df['Sexual Orientation']=idfile.loc[temp_i,'Sexual Orientation']
    temp_df.survey=dfname[i]
    df_orientation=pd.concat([df_orientation,temp_df])

df_orientation=df_orientation.reset_index()

fig,ax = plt.subplots(figsize=(10,6))
s=sns.histplot(data=df_orientation,
             x='survey',
             hue='Sexual Orientation',
             stat='count',#
             multiple='stack',
             palette='colorblind',)

sns.move_legend(s, 
                "lower center", 
                bbox_to_anchor=(0.5,1),
                ncol=3,
                title=None)

ax.set_xticklabels(['presurvey\n2021',
                    'postsurvey\n2021',
                    'presurvey\n2022',
                    'postsurvey\n2022',
                    'mid-year check-in\n2023']);
os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
if savefig: fig.savefig('Figures/Fig_SexualOrientation_v1.png',dpi=600);
os.chdir(homedir)
# =============================================================================
# Collate Income Data in 2022
# =============================================================================

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

s=sns.countplot(data=pre22,
             y="What is your family's approximate yearly income (in US Dolllars)?",
             order=financial_order,color=(0.00392156862745098, 0.45098039215686275, 0.6980392156862745));
os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
if savefig: fig.savefig('Figures/Fig_IncomeData.png',dpi=600);
os.chdir(homedir)

# =============================================================================
# Pre-Post 2021
# =============================================================================
col_perceptions=[col for col in post21 if col.startswith('Select your level of agreement for the following statements - ')]
col_names=[i.split(' - ', 1)[1] for i in col_perceptions]

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
if savefig: fig.savefig('Figures/Fig_Wilcoxon_2021_v1.png',dpi=600);
os.chdir(homedir)
# =============================================================================
# Pre-Post 2022
# =============================================================================
col_perceptions=[col for col in post22 if col.startswith('Select your level of agreement for the following statements - ')]
col_names=[i.split(' - ', 1)[1] for i in col_perceptions]

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
if savefig: fig.savefig('Figures/Fig_Wilcoxon_2022_v1.png',dpi=600);
os.chdir(homedir)


# =============================================================================
# Post 2021 --> Check in 2023; check patency
# =============================================================================
col_perceptions=[col for col in post22 if col.startswith('Select your level of agreement for the following statements - ')]
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

ax.set_title('FLNSUS 2022 Pre/Post Data')

plt.tight_layout()
os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
if savefig: fig.savefig('Figures/Fig_Wilcoxon_post22_mid23.png',dpi=600);
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
    
    ## only people at all years
    fig, ax=plt.subplots(figsize=(10,6))
    all_years=df_col
    # all_years=df_col.dropna()
    all_years=all_years.melt(ignore_index=False);
    all_years['uid']=all_years.index;
    all_years=all_years.reset_index();
    
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