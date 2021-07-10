import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import bar_chart_race as bcr
import os
import sys
import warnings

warnings.filterwarnings('ignore')
plt.style.use('ggplot')



if __name__ == "__main__":

	df = pd.read_csv('./DATA/archive/MLTollsStackOverflow.csv')
	df['month'] = pd.to_datetime(df['month'], format='%y-%b')
	df = df.set_index('month').sort_index()

	bcr.bar_chart_race(df=df, filename='./VIDEOS/race.mp4', title='StackOverflow Question Toll over the year 2009-2019', orientation='h', 
		n_bars=20, figsize=(12, 8))



