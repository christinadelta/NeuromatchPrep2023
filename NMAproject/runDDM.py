#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 06:08:33 2023

@author: christinadelta
"""

# Running the pyDDM toolbox to the Bayesian switching observer task of Laquitaine & Gardner (2018)

# ---------------------------

# Import libraries
import numpy as np
from pyddm import Sample
import pyddm as ddm
import pyddm.plot
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


# Load data 
with open("data01_direction4priors.csv", "r") as f:
    data = pd.read_csv(f) 

"""
with open("project_NMA2023.csv", "r") as f:
    data = pd.read_csv(f) """
    
# define figure settings
rcParams['figure.figsize'] = [20, 4]
rcParams['font.size'] = 11
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['figure.autolayout'] = True

# inspect data
data.head()
len(data) # what is the length?

# covert positions to degrees 
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


# convert subject estimate cartesian coordinates to degrees
estimates_deg = get_cartesian_to_deg(data["estimate_x"].values, data["estimate_y"].values, False)
estimates_deg = np.round(estimates_deg)
data["estimate_deg"] = estimates_deg

# difference between prior and showed direction
data["prior_minus_current_t"] = abs(data["prior_mean"] - data["motion_direction"])

# use the estimated rt only for participants that had missing rts (1-5)
# first compute estimated rts 
# Calculate the 'est_rt' by subtracting the current row's 'trial_time' from the next row's 'trial_time'
data['est_rt_all'] = data.groupby('run_id')['trial_time'].shift(-1) - data['trial_time'] - 1 - 0.3 - 0.1 - 0.1

tmp = data.iloc[0:36441, 18]
tmp2 = data.iloc[36441:83213, 7]

# merge the 2 tmp subsets
frames = [tmp, tmp2]
result = pd.concat(frames)

data['est_rt'] = result

# Set 'None' for the last row of each run_id
last_rows = data.groupby('run_id').tail(1).index
data.loc[last_rows, 'est_rt'] = None

# make conversions of data and compute new variables for DDM
data_rt = data[data["est_rt"].notna()] # first remove nan rows

## Remove trials where prior and current direction are too close
# create a list of possible absolute differences to consider
differences = [*range(5, 20, 5)]

for diff in differences:
  print(f"Difference = {diff}")
  nb_trial = data_rt[data_rt["prior_minus_current_t"] < diff]
  percent_trial = (len(nb_trial) / len(data_rt)) *100
  print(f"Percent of trials excluded = {percent_trial}")
  
# From these results, excluding absolute differences < 10° seems sensible as it only excludes around 16.5% of the trials.
  
# comment the lines below not to exclude the differences that are too small
minimal_difference = 10

data_rt = data_rt[data_rt["prior_minus_current_t"] >= minimal_difference]

# Competition and RTs
# Perform hypothesis test for Pearson correlation coefficient
correlation, p_value = pearsonr(data_rt['prior_minus_current_t'], data_rt['est_rt'])
print(correlation, p_value)

#Regression line

sns.regplot(x='prior_minus_current_t', y='est_rt', color='red', scatter = False,
                 scatter_kws={"color": "blue",
                              "alpha": 0.2},
            line_kws={"color": "red"}, ci=95, data=data_rt)

# Add labels and title
plt.xlabel('|Prior - Current direction|')
plt.ylabel('RT (s)')
plt.title('RT as a function of conflict between prior and current direction')

# Display the plot
plt.show()
  
  
# create the midpoint between prior and showed direction
data_rt["midpoint"] = (data_rt["motion_direction"] + 225) / 2

#sanity check
data_rt["midpoint"].describe()

# we create a variable named choice used to categorize whether subject used on strategy or the other
conds = [
    (data_rt['midpoint'] > 225) & (data_rt['estimate_deg'] > data_rt['midpoint']),
    (data_rt['midpoint'] > 225) & (data_rt['estimate_deg'] <= data_rt['midpoint']),
    (data_rt['midpoint'] <= 225) & (data_rt['estimate_deg'] < data_rt['midpoint']),
    (data_rt['midpoint'] <= 225) & (data_rt['estimate_deg'] >= data_rt['midpoint'])
]
choices = [1, 0, 1, 0]
data_rt['choice'] = np.select(conds, choices)

# compute the choice history (prediction of next trial based on previous trial's choice)
# Set 'None' for the first row of each session_id
data_rt["choice_hist"] = data_rt["choice"]

# convert all first trials of each session to nan
first_rows = data_rt.groupby('session_id')["subject_id"].head(1).index
data_rt.loc[first_rows, 'choice_hist'] = None 



# compute zscore for each strength parameter and then the difference
data_rt['prior_std_norm'] = zscore(1/data_rt["prior_std"])
data_rt['motion_coherence_norm'] = zscore(data_rt["motion_coherence"])

# compute
data_rt['strength_diff'] = np.round(data_rt['motion_coherence_norm'] - data_rt['prior_std_norm'])

# we compute mean choice values as a function of the difference in strength (should be at 0.5)
mean_vals = data_rt.groupby('strength_diff')['choice'].mean()

## now for DDM version 1
# choose subject
# extract data for subject 7
sub_df_rts = data_rt[data_rt["subject_id"]==1]

# len(sub_df_rts)

# Create a sample object from our data.  This is the standard input
# format for fitting procedures.  Since RT and correct/error are
# both mandatory columns, their names are specified by command line
# arguments.
data_sample = Sample.from_pandas_dataframe(sub_df_rts, rt_column_name="estimate_roughly_reaction_time", choice_column_name="choice")

# create conditions
conditions = ["strength_diff", "subject_id", "choice"]

# create the ddm class
class DriftCoherence(ddm.models.Drift):
    name = "Drift depends linearly on coherence"
    required_parameters = ["driftdiff"] # <-- Parameters we want to include in the model
    required_conditions = ["strength_diff"] # <-- Task parameters ("conditions"). Should be the same name as in the sample.

    # We must always define the get_drift function, which is used to compute the instantaneous value of drift.
    def get_drift(self, conditions, **kwargs):
        return self.driftdiff * conditions['strength_diff']
    
# fit the model
from pyddm import Model, Fittable
from pyddm.functions import fit_adjust_model, display_model
from pyddm.models import NoiseConstant, BoundConstant, OverlayChain, OverlayNonDecision, OverlayPoissonMixture

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
                 dx=.001, dt=.01, T_dur=7)

# Fitting this will also be fast because PyDDM can automatically
# determine that DriftCoherence will allow an analytical solution.
fit_model_rs = fit_adjust_model(sample=data_sample, model=model_rs, verbose=False)

# check the fit
display_model(fit_model_rs)

# plot
pyddm.plot.plot_fit_diagnostics(model=fit_model_rs, sample=data_sample)

# -------------------------------
# DDM version 2 -- remove rts

# choose subject
# extract data for subject 7
sub_df_rts = data_rt[data_rt["subject_id"]==8]

# Remove short and long RTs, as in 10.1523/JNEUROSCI.4684-04.2005.
# This is not strictly necessary, but is performed here for
# compatibility with this study.
sub_df_rts = sub_df_rts[sub_df_rts["estimate_roughly_reaction_time"] > .1] # Remove trials less than 100ms
sub_df_rts = sub_df_rts[sub_df_rts["estimate_roughly_reaction_time"] < 1.65] # Remove trials greater than 1650ms

# Create a sample object from our data.  This is the standard input
# format for fitting procedures.  Since RT and correct/error are
# both mandatory columns, their names are specified by command line
# arguments.
data_sample = Sample.from_pandas_dataframe(sub_df_rts, rt_column_name="estimate_roughly_reaction_time", choice_column_name="choice")

# create conditions
conditions = ["strength_diff", "subject_id", "choice"]

# create the ddm class
class DriftCoherence(ddm.models.Drift):
    name = "Drift depends linearly on coherence"
    required_parameters = ["driftdiff"] # <-- Parameters we want to include in the model
    required_conditions = ["strength_diff"] # <-- Task parameters ("conditions"). Should be the same name as in the sample.

    # We must always define the get_drift function, which is used to compute the instantaneous value of drift.
    def get_drift(self, conditions, **kwargs):
        return self.driftdiff * conditions['strength_diff']
    
# fit the model
from pyddm import Model, Fittable
from pyddm.functions import fit_adjust_model, display_model
from pyddm.models import NoiseConstant, BoundConstant, OverlayChain, OverlayNonDecision, OverlayPoissonMixture

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
                 dx=.001, dt=.01, T_dur=2)

# Fitting this will also be fast because PyDDM can automatically
# determine that DriftCoherence will allow an analytical solution.
fit_model_rs = fit_adjust_model(sample=data_sample, model=model_rs, verbose=False)


# check the fit
display_model(fit_model_rs)

# plot
pyddm.plot.plot_fit_diagnostics(model=fit_model_rs, sample=data_sample)



### testing stuff
subjects = 12
first_rows = [] # init emptylist

for sub in range(subjects):
    
    this_sub = sub + 1
    print(this_sub)
    

    sub_rows = data_rt[data_rt["subject_id"] == this_sub]
    tmp_rows = data_rt.groupby('session_id').head(1)
    tmp = list(tmp_rows["Index"])

    # append indexes of sessions' first rows
    first_rows.append(tmp)  
    
    






