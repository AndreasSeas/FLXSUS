#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 23:15:26 2023

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
import sys, os
import nltk
import tqdm
import geopandas
from shapely.geometry import Point

# =============================================================================
# Map Comparing Two Years
# =============================================================================
# https://onelinerhub.com/python-matplotlib/how-to-fill-countries-with-colors-using-world-map
# from https://www.naturalearthdata.com/downloads/110m-cultural-vectors/

def map2yrs(df1,df2, label1, label2,figsize,markersize,marker1,marker2,alpha1,alpha2,linewidth,color1,color2):
    # load in the world map
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))    

    # init the figure    
    fig, ax=plt.subplots(figsize=figsize,ncols=1,nrows=1,)
    
    # create a world plot on that axis
    world.plot(ax=ax,color='#CCCCCC',)
    # plot boundaries between countries
    world.boundary.plot(color=[0.5,0.5,0.5],linewidth=0.5,ax=ax,)
    
    # set figure metadata
    
    # ax.plot(x = df1.loc[:,'Longitude'],
    #         y = df1.loc[:,'Latitude'],
    #         fmt = 'o',
    #         markerfacecolor=None,
    #         markeredgecolor='k',
    #         )
    
    ax.scatter(df1.loc[:,'Longitude'],
                df1.loc[:,'Latitude'],
                s=markersize,
                marker=marker1,
                facecolors=color1,
                edgecolors=color1,
                linewidths=linewidth,
                alpha=alpha1,
                label=label1)
    
    ax.scatter(df1.loc[:,'Longitude'],
                df1.loc[:,'Latitude'],
                s = markersize,
                marker=marker2,
                facecolors=color2,
                edgecolors=color2,
                linewidths=linewidth,
                alpha=alpha2,
                label=label1)
    
    ax.legend(loc='lower center',ncol=5,bbox_to_anchor=[0.5,-0.02])
    
    plt.tight_layout()
    plt.axis('off')

    return ax


def map2yrs_panels(df1,df2, label1, label2,figsize,markersize,marker1,marker2,alpha1,alpha2,linewidth1,linewidth2,color1,color2,facecolor1,facecolor2):
    
    # https://basemaptutorial.readthedocs.io/en/latest/backgrounds.html
    
    fig, ax=plt.subplots(figsize=figsize,ncols=2,nrows=1,)
    # NAmap = Basemap(projection='ortho',lat_0=45,lon_0=-100,resolution='l',ax=ax)
    # NAmap = Basemap(llcrnrlon=190,llcrnrlat=0,urcrnrlon=0,urcrnrlat=59.5,
    #         resolution='l',projection='cass',lat_0=14.5, lon_0=-82.0)

    NAmap = Basemap(projection = 'ortho',lat_0=0,lon_0=-100,ax=ax[0],resolution='i')
    
    # NAmap = Basemap(width=12000000,height=9000000,
    #         resolution='l',projection='eqdc',\
    #         lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.)
    NAmap.drawcoastlines(linewidth=0.5,ax=ax[0])
    NAmap.drawcountries(linewidth=0.25,ax=ax[0])
    NAmap.fillcontinents(color='#CCCCCC',lake_color='aqua',ax=ax[0])
    NAmap.drawmapboundary(fill_color='aqua',ax=ax[0])
    
    x, y = NAmap(df1.loc[:,'Longitude'], df1.loc[:,'Latitude'])
    NAmap.scatter(x,y,edgecolors=color1,ax=ax[0],alpha=alpha1,marker=marker1,facecolors=facecolor1,linewidth=linewidth1)
    
    x, y = NAmap(df2.loc[:,'Longitude'], df2.loc[:,'Latitude'])
    NAmap.scatter(x,y,edgecolors=color2,ax=ax[0],alpha=alpha2,marker=marker2,facecolors=facecolor2,linewidth=linewidth2)
    
    
    NAmap = Basemap(projection = 'ortho',lat_0=0,lon_0=60,ax=ax[1])
    
    # NAmap = Basemap(width=12000000,height=9000000,
    #         resolution='l',projection='eqdc',\
    #         lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.)
    NAmap.drawcoastlines(linewidth=1,ax=ax[1])
    NAmap.drawcountries(linewidth=0.25,ax=ax[1])
    NAmap.fillcontinents(color='#CCCCCC',lake_color='aqua',ax=ax[1])
    NAmap.drawmapboundary(fill_color='aqua',ax=ax[1])
    
    x, y = NAmap(df1.loc[:,'Longitude'], df1.loc[:,'Latitude'])
    NAmap.scatter(x,y,edgecolors=color1,ax=ax[1],alpha=alpha1,marker=marker1,facecolors=facecolor1,linewidth=linewidth1,label=label1)
    
    x, y = NAmap(df2.loc[:,'Longitude'], df2.loc[:,'Latitude'])
    NAmap.scatter(x,y,edgecolors=color2,ax=ax[1],alpha=alpha2,marker=marker2,facecolors=facecolor2,linewidth=linewidth2,label=label2)
    
    # plt.legend(loc='lower center',bbox_to_anchor=[0,-0.02])
    handles, labels = ax[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',bbox_to_anchor=(0.5,0))
    
    plt.tight_layout()
    # plt.axis('off')
    return fig,ax



def map2yrs_1panel(df1,df2, label1, label2,figsize,markersize,marker1,marker2,alpha1,alpha2,linewidth1,linewidth2,color1,color2,facecolor1,facecolor2):
    
    # https://basemaptutorial.readthedocs.io/en/latest/backgrounds.html
    
    fig, ax=plt.subplots(figsize=figsize,ncols=1,nrows=1,)
    
    NAmap = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,\
            llcrnrlon=-180,urcrnrlon=180,resolution='i')
    #Basemap(projection = 'ortho',lat_0=0,lon_0=-100,ax=ax,resolution='i')
    
    NAmap.drawcoastlines(linewidth=0.5,ax=ax)
    NAmap.drawcountries(linewidth=0.25,ax=ax)
    NAmap.fillcontinents(color='#CCCCCC',lake_color='white',ax=ax)
    NAmap.drawmapboundary(fill_color='white',ax=ax)
    
    x, y = NAmap(df1.loc[:,'Longitude'], df1.loc[:,'Latitude'])
    NAmap.scatter(x,y,edgecolors=color1,ax=ax,alpha=alpha1,marker=marker1,facecolors=facecolor1,linewidth=linewidth1,label=label1)
    
    x, y = NAmap(df2.loc[:,'Longitude'], df2.loc[:,'Latitude'])
    NAmap.scatter(x,y,edgecolors=color2,ax=ax,alpha=alpha2,marker=marker2,facecolors=facecolor2,linewidth=linewidth2,label=label2)
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',bbox_to_anchor=(0.5,0))
    
    plt.tight_layout()
    plt.axis('off')
    
    
    return fig,ax
    