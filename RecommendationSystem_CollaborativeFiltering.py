#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 15:41:24 2022

@author: nitinsinghal
"""

#recommendation system, user-based (user-item) collaborative filtering

import pandas as pd
import numpy as np
from scipy.spatial import distance

# Read data
data = pd.read_csv('./Q4Data.csv')
data.rename(columns={'Unnamed: 0':'User'}, inplace=True)

# Matrix normalization
data.set_index('User', inplace=True)
data_mean = data.mean(axis=1)
print(data_mean)

data_norm = data.sub(data_mean, axis=0)
print(data_norm)

# Define a focal user and a target artist
focal_user, target_artist = "Bob", "Joker"

# Calculate the similarities between the focal user and the other users (i.e., neighbors)
sim_dict = {}

for i in range(len(data_norm)):
    if(data_norm.index[i]==focal_user):
        continue
    else:
        a,b = [],[]
        for j in range(len(data_norm.loc[focal_user])):
            if(not(np.isnan(data_norm.loc[focal_user][j]) or np.isnan(data_norm.iloc[i,j]))):
                a.append(data_norm.loc[focal_user][j])
                b.append(data_norm.iloc[i][j])
                sim_dict[data_norm.index[i]] = 1.0-distance.cosine(np.array(a),np.array(b))

print(sim_dict)
# Calculate the predicted score of the focal user on the target artist
# CF (i.e., 𝑟𝑓𝑖=𝑟𝑓̅+ Σ𝑠𝑖𝑚𝑓,𝑗∗𝑟𝑗𝑖Σ|𝑠𝑖𝑚𝑓,𝑗| 
#(𝑓=𝑓𝑜𝑐𝑎𝑙 𝑢𝑠𝑒𝑟;𝑖= 𝑡𝑎𝑟𝑔𝑒𝑡 𝑖𝑡𝑒𝑚;𝑟𝑓̅=𝑎𝑣𝑒𝑟𝑎𝑔𝑒 𝑟𝑎𝑡𝑖𝑛𝑔 𝑜𝑓 𝑓𝑜𝑐𝑎𝑙 𝑢𝑠𝑒𝑟;𝑠𝑖𝑚𝑓,𝑗= 𝑠𝑖𝑚𝑖𝑙𝑎𝑟𝑖𝑡𝑦 𝑏𝑒𝑡𝑤𝑒𝑒𝑛 𝑓𝑜𝑐𝑎𝑙 𝑢𝑠𝑒𝑟 𝑎𝑛𝑑 𝑢𝑠𝑒𝑟 𝑗;𝑟𝑗𝑖=𝑟𝑎𝑡𝑖𝑛𝑔 𝑜𝑓 𝑢𝑠𝑒𝑟 𝑗 𝑜𝑛 𝑖𝑡𝑒𝑚 𝑖)

rf_ = data_mean.loc[focal_user]

sum_sim_fj=0
sum_sim_fj_rji =0
for i in range(len(data)):
    if(data.index[i]!=focal_user):
        if(not(np.isnan(data.loc[data.index[i],target_artist]))):
            sum_sim_fj += abs(sim_dict[data.index[i]])
            sum_sim_fj_rji += sim_dict[data.index[i]] * data.loc[data.index[i],target_artist]

cf_score = rf_ + (sum_sim_fj_rji / sum_sim_fj)
print('CF score: ',cf_score)

# Output: CF score:  1.448431361543539






























