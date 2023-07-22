#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 11:49:38 2023

@author: christinadelta
"""

# Version 2 of modelling analyses of the NMA project 2023 

# ---------------------

# add libraries 
import numpy as np
from pyddm import Sample
import pyddm as ddm
import pyddm.plot
from pyddm import Model, Fittable
from pyddm.functions import fit_adjust_model, display_model
from pyddm.models import NoiseConstant, BoundConstant, OverlayChain, OverlayNonDecision, OverlayPoissonMixture
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os, requests
from matplotlib import colors
from matplotlib import rcParams
from matplotlib.colors import ListedColormap
from numpy import pi
from copy import copy
from scipy.stats import pearsonr, zscore

# load data
with open("data01_direction4priors.csv", "r") as f:
    data = pd.read_csv(f) 
    
# function that converts the x and y positions to degrees 
def get_cartesian_to_deg(
    x: np.ndarray, y: np.ndarray, signed: bool
) -> np.ndarray:
    """convert cartesian coordinates to
    angles in degree
    Args:
        x (np.ndarray): x coordinate
        y (np.ndarray): y coordinate
        signed (boolean): True (signed) or False (unsigned)
    Usage:
        .. code-block:: python
            import numpy as np
            from bsfit.nodes.cirpy.utils import get_cartesian_to_deg
            x = np.array([1, 0, -1, 0])
            y = np.array([0, 1, 0, -1])
            degree = get_cartesian_to_deg(x,y,False)
            # Out: array([  0.,  90., 180., 270.])
    Returns:
        np.ndarray: angles in degree
    """
    # convert to radian (ignoring divide by 0 warning)
    with np.errstate(divide="ignore"):
        degree = np.arctan(y / x)

    # convert to degree and adjust based
    # on quadrant
    for ix in range(len(x)):
        if (x[ix] >= 0) and (y[ix] >= 0):
            degree[ix] = degree[ix] * 180 / np.pi
        elif (x[ix] == 0) and (y[ix] == 0):
            degree[ix] = 0
        elif x[ix] < 0:
            degree[ix] = degree[ix] * 180 / np.pi + 180
        elif (x[ix] >= 0) and (y[ix] < 0):
            degree[ix] = degree[ix] * 180 / np.pi + 360

    # if needed, convert signed to unsigned
    if not signed:
        degree[degree < 0] = degree[degree < 0] + 360
    return degree


# convert cartesian positions to degrees
# we convert cartesian estimates into a degree
data["estimate_deg"] = np.round(get_cartesian_to_deg(data["estimate_x"].values, data["estimate_y"].values, False))

# difference between prior and showed direction
data["prior_minus_current_t"] = abs(data["prior_mean"] - data["motion_direction"])


# Add estimated reaction times ( for first 5 subjects)

# est_rt = trial_time(t+1)-trial_time(t)-1s(fixation)-0.3s(motion display)-0.1s(confirmation)-0.1s(feedback)

# Calculate the 'est_rt' by subtracting the current row's 'trial_time' from the next row's 'trial_time'
data['est_rt'] = data.groupby('run_id')['trial_time'].shift(-1) - data['trial_time'] - 1 - 0.3 - 0.1 - 0.1

# select est rt > 0
data_rt = data[data["est_rt"] >= 0]

# Replace NaN values in 'reaction_times' with 'est_rt' values
data_rt["reaction_time"].fillna(data_rt["est_rt"], inplace=True)

# subset not na values
data_rt = data_rt[data_rt["reaction_time"].notna()]

# Assuming 'data_rt' DataFrame contains 'est_rt' and 'reaction_time' columns
est_rt_values = data_rt['est_rt']
reaction_time_values = data_rt['reaction_time']

# Create a figure and axis for the plot
fig, ax = plt.subplots()

# Plot the histogram for 'est_rt'
ax.hist(est_rt_values, bins=30, alpha=0.5, label='est_rt')

# Plot the histogram for 'reaction_time' and set a different color
ax.hist(reaction_time_values, bins=30, alpha=0.5, label='reaction_time', color='orange')

# Set labels and legend
ax.set_xlabel('Response Time')
ax.set_ylabel('Frequency')
ax.legend()

# Show the plot
plt.show()

print(est_rt_values.describe())
print(reaction_time_values.describe())

## perform hypothesis testing
# Perform hypothesis test for Pearson correlation coefficient
correlation, p_value = pearsonr(data_rt['prior_minus_current_t'], data_rt['reaction_time'])
print(correlation, p_value)

#Regression line
sns.regplot(x='prior_minus_current_t', y='reaction_time', color='red', scatter = False,
                 scatter_kws={"color": "blue",
                              "alpha": 0.2},
            line_kws={"color": "red"}, ci=95, data=data_rt)

# Add labels and title
plt.xlabel('|Prior - Current direction|')
plt.ylabel('RT (s)')
plt.title('RT as a function of conflict between prior and current direction')

# Display the plot
plt.show()


# remove trials where the prior and direction are too close
# create a list of possible absolute differences to consider
differences = [*range(5, 20, 5)]

# 5-10 seems to work
for diff in differences:
  print(f"Difference = {diff}")
  nb_trial = data_rt[data_rt["prior_minus_current_t"] < diff]
  percent_trial = (len(nb_trial) / len(data_rt)) *100
  print(f"Percent of trials excluded = {percent_trial}")
  
  
# comment the lines below not to exclude the differences that are too small
minimal_difference = 10

data_rt = data_rt[data_rt["prior_minus_current_t"] >= minimal_difference]

# DDM data preprocessing
# create the midpoint between prior and showed direction
data_rt["midpoint"] = (data_rt["motion_direction"] + 225) / 2

#sanity check
data_rt["midpoint"].describe()

# we create a variable named choice used to categorize whether subject used on strategy or the other
# if midpoint --> used prior (strategy 0)
conds = [
    (data_rt['midpoint'] > 225) & (data_rt['estimate_deg'] > data_rt['midpoint']),
    (data_rt['midpoint'] > 225) & (data_rt['estimate_deg'] <= data_rt['midpoint']),
    (data_rt['midpoint'] <= 225) & (data_rt['estimate_deg'] < data_rt['midpoint']),
    (data_rt['midpoint'] <= 225) & (data_rt['estimate_deg'] >= data_rt['midpoint'])
]
choices = [1, 0, 1, 0]
data_rt['choice'] = np.select(conds, choices)

# sanity check
print(data_rt.iloc[10:20].loc[:, ['motion_direction', 'estimate_deg', 'midpoint', 'choice']])

# compute zscore for each strength parameter and then the difference
data_rt['prior_std_norm'] = zscore(1/data_rt["prior_std"])
data_rt['motion_coherence_norm'] = zscore(data_rt["motion_coherence"])

# compute
data_rt['strength_diff'] = np.round(data_rt['motion_coherence_norm'] - data_rt['prior_std_norm'])

# we compute mean choice values as a function of the difference in strength (should be at 0.5)
mean_vals = data_rt.groupby('strength_diff')['choice'].mean()

print(mean_vals)

# create choice history variable 
data_rt["choice_hist"] = data_rt["choice"]
data_rt["choice_hist"] = data_rt.choice_hist.shift(1)

# convert all first trials of each session to nan
first_rows = data_rt.groupby(['session_id', 'subject_id']).head(1).index
data_rt.loc[first_rows, 'choice_hist'] = None 

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

#### Prepare parameters and variables for ddm 

###### VERSION 1 (Model 1)

# subset one subject 
data_rt["subject_id"].head(10)

# extract data for subject 7
sub_df_rts = data_rt[data_rt["subject_id"]==1]
# sub_df_rts = data_rt

len(sub_df_rts)

# run ddm version 1 with one sub

# Create a sample object from our data.  This is the standard input
# format for fitting procedures.  Since RT and correct/error are
# both mandatory columns, their names are specified by command line
# arguments.
data_sample = Sample.from_pandas_dataframe(sub_df_rts, rt_column_name="reaction_time", choice_column_name="choice")

# create conditions
conditions = ["strength_diff", "subject_id", "choice"]

# create class of version 1 pyDDM
class DriftCoherence(ddm.models.Drift):
    name = "Drift depends linearly on coherence"
    required_parameters = ["driftdiff"] # <-- Parameters we want to include in the model
    required_conditions = ["strength_diff"] # <-- Task parameters ("conditions"). Should be the same name as in the sample.

    # We must always define the get_drift function, which is used to compute the instantaneous value of drift.
    def get_drift(self, conditions, **kwargs):
        return self.driftdiff * conditions['strength_diff']
    
# fit the model
model_rs = Model(name='Laquitaine data, drift varies with strategy used',
                 drift=DriftCoherence(driftdiff=Fittable(minval=0, maxval=20)),
                 noise=NoiseConstant(noise=1),
                 bound=BoundConstant(B=Fittable(minval=.1, maxval=1.5)),
                 # Since we can only have one overlay, we use
                 # OverlayChain to string together multiple overlays.
                 # They are applied sequentially in order.  OverlayNonDecision
                 # implements a non-decision time by shifting the
                 # resulting distribution of response times by
                 # `nondectime` seconds.
                 overlay=OverlayChain(overlays=[OverlayNonDecision(nondectime=Fittable(minval=0, maxval=.4)),
                                                OverlayPoissonMixture(pmixturecoef=.02,
                                                                      rate=1)]),
                 dx=.001, dt=.0001, T_dur=5.1)

fit_model_rs = fit_adjust_model(sample=data_sample, model=model_rs, verbose=False)

# display model output 
display_model(fit_model_rs)

# let's access the parameters 
fit_model_rs.parameters()

# plot the fit 



























