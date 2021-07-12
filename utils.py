import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as pyo

from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from collections import defaultdict

class MovingAverageFilter:

	"""
	Wrapper class for the application of 
	Moving Average Filter on a data.

	Parameters:
	-----------
	win_len : int
		Represents the number of timesteps to consider for calculating the Moving Average

	Returns:
	--------
	Moving Average of the data

	"""
	def __init__(self, win_len):

		self.win_len = win_len
		self.data = list()

	def step(self, x):
		self.data.append(x)

		if len(self.data) > self.win_len:
			self.data.pop(0)

	def current_state(self):
		return np.mean(self.data)


def apply_moving_average_filter(data, win_len=5):

	"""
	Function to apply the Moving Average Filter on the data

	Parameters:
	-----------
	data : array-like
		Represents the data with which we are working.

	win_len : int, optional; default:5
		Represents the number of timesteps to consider for calculating the Moving Average

	Returns:
	--------
	smoothed_vals : array-like
		Represents the moving average of the data

	"""
	ma = MovingAverageFilter(win_len=win_len)
	
	smoothed_vals = list()

	for i in range(len(data)):
		ma.step(data[i])
		if len(ma.data) != win_len:
			smoothed_vals.append(np.nan)
		else:
			smoothed_vals.append(ma.current_state())

	return smoothed_vals

def plot_interactive(data, cols, title='Interactive Plot'):

	"""
	Function to plot interactive plot for the data

	Parameters:
	----------
	data : pandas.DataFrame
		Represents the dataset.

	cols : array-like
		Represents the data to be plotted.

	title : str, optional; default:'Interactive Plot'
		Represents the title for the plot


	Returns:
	--------
	fig : plotly.Figure
		Represents the interactive plot

	"""


	layout = dict(autosize=False, width=900, title=title,
		xaxis=dict(

			rangeslider = dict(visible=True),
			rangeselector=dict(

				buttons = list([

					dict(count=1, label='1y', step='year', stepmode='backward'),
					dict(count=3, label='3y', step='year', stepmode='backward'),
					dict(count=5, label='5y', step='year', stepmode='backward'),
					dict(step='all')

					])

				)
			),template='plotly_white')

	fig = go.Figure(layout=layout)
	for col in cols:

		orgn = go.Scatter(name=f"{col}",
			x=data.index,
			y=data[col],
			mode='lines',
			line=dict(width=3))
		fig.add_trace(orgn)

	return fig



def multiple_distribution_plots(data, cols, title='Distribution Plots'):

	fig, ax = plt.subplots(figsize=(12, 8))
	
	for col in cols:
		sns.distplot(data[col], hist=False, label=f'{col}', ax=ax)

	ax.set_xlabel('Values')
	ax.legend()

	return fig


def box_dist(data, col, title='Distribution Plot'):

	fig, ax = plt.subplots(2, 1, figsize=(12, 8))
	sns.boxplot(x=data[col], ax=ax[0])
	sns.distplot(data[col], ax=ax[1])
	fig.suptitle(title)

	return fig



def interactive_pie_chart(data):
	labels = ['python', 'r', 'matlab']
	value_2009 = [data.loc['2009':'2009', col].values[0] for col in labels]
	value_2019 = [data.loc['2019':'2019', col].values[0] for col in labels]

	fig = make_subplots(1, 2, specs=[[dict(type='domain'), dict(type='domain')]], subplot_titles=['2009', '2019'])

	fig.add_trace(go.Pie(labels=labels, values=value_2009, scalegroup='one', name='Stackoverflow Question Toll 2009', hole=.3), 1, 1)
	fig.add_trace(go.Pie(labels=labels, values=value_2019, scalegroup='one', name='Stackoverflow Question Toll 2019', hole=.3), 1, 2)

	fig.update_layout(title_text='Stack Overflow Question Toll of Python, R, and Matlab', width=900)
	return fig


def adfuller_test(data, trace=False):
	"""
	Perform the Augmented-Dickey Fuller Test on the data

	Parameters:
	-----------
	data : pandas.DataFrame
		Represents the data on which the test needs to be performed.

	trace : bool, optional; default:False
		Represents whether to showcase the results of the test

	Returns:
	-------
	res : pandas.Series
		Represents the result obtained from the test

	is_stn : bool
		Represents whether the test considers the series to be stationary or not.

	"""

	test = adfuller(data)
	res = pd.Series(test[0:4], index=['ADF-Test Statistics', 'p-values', 'No of Lags Used', 'No of Observations'], name='ADF results')

	for k, v in test[4].items():

		res[f'critical-value {k}'] = v

	if trace:
		print(res.to_string())

		if test[1] <= 0.05:
			print('The ADF-Test considers the series to be stationary')

		else:
			print('The ADF-Test considers the series to be non-stationary')


	is_stn = test[1] <= 0.05

	return res, is_stn


def kpss_test(data, reg='c', trace=False):

	"""
	Function to compute the KPSS Test on the data.

	Parameters:
	-----------
	data : pandas.Series
		Represents the data on which the test needs to be performed.

	reg : str, optional; default:'c'
		Represents the null hypothesis for the test. Possible values
		`c` or `ct`

	trace : bool, optional; default:False
		Represents whether to showcase the results or not.


	Returns:
	--------
	res : pandas.Series
		Represents the results of the test

	is_stn : bool
		Represents the series is stationary or not as per the KPSS Test.

	"""

	test = kpss(data, regression=reg)
	res = pd.Series(test[0:3], index=['KPSS-Test Stat', 'p-values', 'No of Lags'], name='KPSS results')

	for k, v in test[3].items():
		res[f'critical-value {k}'] = v

	if trace:
		print(res.to_string())

		if test[1] > 0.05:
			print('The series as per the KPSS Test is Stationary')

		else:
			print('The series as per the KPSS Test is Non-Stationary')

	is_stn = test[1] > 0.05 

	return res, is_stn

