# fanova library 예제

import numpy as np
import pandas as pd

from fanova import fANOVA
import fanova.visualizer

import ConfigSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

import os
path = '/Users/songmi/Desktop/NutellaCoder/fanova/examples' # os.path.dirname(os.path.realpath(__file__))

# directory in which you can find all plots
plot_dir = path + '/example_data/test_plots'

# artificial dataset (here: features)
features = np.loadtxt(path + '/example_data/diabetes_features.csv', delimiter=",")
responses = np.loadtxt(path + '/example_data/diabetes_responses.csv', delimiter=",")
df_x = pd.DataFrame(features[0:443, 0:3], columns=['0','1','2']) #,'3', '4','5','6','7','8','9'])

# config space
pcs = list(zip(np.min(df_x, axis=0), np.max(df_x, axis=0)))
cs = ConfigSpace.ConfigurationSpace()
#for i in range(len(pcs)):
for i in range(3):
	cs.add_hyperparameter(UniformFloatHyperparameter("%i" %i, pcs[i][0], pcs[i][1]))

# create an instance of fanova with trained forest and ConfigSpace
f = fANOVA(X = df_x, Y = responses, config_space=cs)

# marginal of particular parameter:
dims = ('0', '1', '2')
res = f.quantify_importance(dims)
print(res)

# getting the 10 most important pairwise marginals sorted by importance
best_margs = f.get_most_important_pairwise_marginals(n=2)
print(best_margs)