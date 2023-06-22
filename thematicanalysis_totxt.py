#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 13:40:25 2023

@author: as822
"""

import pandas as pd
import os

df=pd.read_excel('ThematicAnalysis.xlsx');

homedir=os.getcwd()

# os.chdir('/Users/as822/Documents/GitHub/FLXSUS/rawtxt')
f=open("allthemes.txt","w")
for i,txt in enumerate(df.words):
    
    f.write(txt)
    f.write('\n')
    f.write('\n')
    f.write('===============================')
    f.write('\n')
    f.write('\n')
    
f.close()
# os.chdir(homedir)
    