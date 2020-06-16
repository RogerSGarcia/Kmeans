# convert -delay 40 -loop 0 iter_*.png xclara_3.gif
import os
import numpy as np
import random
import json
import matplotlib.pyplot as plt
import pandas as pd
import csv

def tofile(filename, xdata, ydata):
    # Creating pandas dataframe from numpy array
    dataset_df = pd.DataFrame({'x': xdata, 'y': ydata})
    # export pandas df to csv
    dataset_df.to_csv(filename, index = False, header = True)


filename = "data/xclara.csv"

dataframe = pd.read_csv(filename)
points = dataframe.get(['x', 'y']).values
# xdata = points[:,0]  
# ydata = points[:,1]  
ydata = points[:,1]  * np.random.uniform(low=-1.4, high=1.41, size=(len(dataframe,)))
xdata = points[:,0]  + np.random.uniform(low=2.5, high=3.41, size=(len(dataframe,)))

plt.scatter(xdata, ydata, c='black', s=7)
plt.show()

filename = "data/points_3.csv"
# tofile(filename, xdata, ydata)