import numpy as np
import pandas as pd

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
