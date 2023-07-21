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
with open("project_NMA2023.csv", "r") as f:
    df_data = pd.read_csv(f)
    
# define figure settings
rcParams['figure.figsize'] = [20, 4]
rcParams['font.size'] = 11
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['figure.autolayout'] = True

# inspect data
df_data.head()
len(df_data) # what is the length?

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
estimates_deg = get_cartesian_to_deg(df_data["estimate_x"].values, df_data["estimate_y"].values, False)
estimates_deg = np.round(estimates_deg)

df_data["estimate_deg"] = estimates_deg

# make conversions of data and compute new variables for DDM
data_rt = df_data[df_data["estimate_roughly_reaction_time"].notna()] # first remove nan rows

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

















