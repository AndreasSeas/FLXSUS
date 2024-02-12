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
import io

# =============================================================================
# load in the required data
# =============================================================================
def city_state_country(row):
    coord = f"{row['Latitude']}, {row['Longitude']}"
    location = geolocator.reverse(coord, exactly_one=True)
    address = location.raw['address']
    city = address.get('city', '')
    state = address.get('state', '')
    country = address.get('country', '')
    row['city'] = city
    row['state'] = state
    row['country'] = country
    return row

homedir=os.getcwd()
datadir='/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/Database';

os.chdir(datadir)

idfile=pd.read_excel("IDFile.xlsx")

os.chdir(homedir)

# =============================================================================
# Get latlong dataframe
# =============================================================================

LL=idfile[['Unique ID','Latitude','Longitude']]
LL['city']= None
LL['state']= None
LL['country']= None

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

geolocator = Nominatim(user_agent="application")

reverse = RateLimiter(geolocator.reverse, min_delay_seconds=0.1)

lala=(((idfile.State=='unk') | (idfile.State.isnull())) & ~idfile.Latitude.isnull())

indices=[i for i, x in enumerate(lala) if x]

for i in indices:
    location = reverse((LL.loc[i,'Latitude'],LL.loc[i,'Longitude']), 
                       language='en', exactly_one=True)    
    address = location.raw['address']
    LL.loc[i,'city'] = address.get('city', '')
    LL.loc[i,'state'] = address.get('state', '') 
    LL.loc[i,'country'] = address.get('country', '')
    

    print(i)

LL.to_csv('LL.csv')
