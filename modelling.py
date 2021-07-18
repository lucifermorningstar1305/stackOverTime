import numpy as np
import pandas as pd

from pmdarima import auto_arima
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults


def find_best_fit(data, seasonal=False, trace=True):

	"""
	Function to find the best set of parameters
	for fitting the ARIMA based model.

	Parameters:
	-----------
	data : pandas.Series
		Represents the data on which the ARIMA model needs to be applied.

	seasonal : bool, optional; default:True
		Represents whether there are any seasonality in the data.

	trace : bool, optional; default:True
		Represents whether to showcase the best parameters.

	Returns:
	--------
	best_params : dict
		Represents the best set of orders for the ARIMA model.

	"""
	best_params = dict()

	stepwise_fit = auto_arima(data, seasonal=seasonal, trace=trace, start_p=0, start_q=0, max_p=5, max_q=5)
	best_params['order'] = stepwise_fit.get_params()['order']

	return best_params


def arima_fit(data, order, filename=None):

	"""
	Function to fit the ARIMA model on the data
	
	Parameters:
	-----------
	data : pandas.Series
		Represents the data which needs to be fitted.

	order : tuple
		Represents the order for the ARIMA

	filename : str, optional; default:None
		Represents the filename to save the ARIMA model.

	Returns:
	-------
	model_fit : statsmodels.tsa.arima_model.ARIMA
		Represents the ARIMA model
	"""

	model = ARIMA(data, order=order)
	model_fit = model.fit()
	
	if filename is not None:
		model_fit.save(filename)

	return model_fit


def __getnewargs__(self):
	return ((self.endog), (self.k_lags, self.k_diff, self.k_ma))



def load_arima(filename):

	"""
	Function to load the ARIMA model 

	Parameters:
	-----------
	filename : str	
		Represents the filename of the arima model

	Returns:
	-----------
	model : statsmodels.tsa.arima_model.ARIMA
		Represents the ARIMA model

	""" 
	ARIMA.__getnewargs__ = __getnewargs__
	model = ARIMAResults.load(filename)


def arima_forecast(model, start, end):
	return model.predict(start, end, typ='levels')




