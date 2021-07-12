import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as pyo
import streamlit as st
import os
import sys
import warnings

from statsmodels.tsa.seasonal import seasonal_decompose
from plotly.subplots import make_subplots

from utils import apply_moving_average_filter, plot_interactive, multiple_distribution_plots,\
 box_dist, interactive_pie_chart, adfuller_test, kpss_test

warnings.filterwarnings('ignore')
plt.style.use('ggplot')

if __name__ == "__main__":

	df = pd.read_csv('./DATA/archive/MLTollsStackOverflow.csv')
	df['month'] = pd.to_datetime(df['month'], format='%y-%b')
	df = df.set_index('month').sort_index()

	st.set_page_config(page_title='StackOverflow')

	st.markdown(""" 
		<head>
		</head>
		<body>
		<div>
		<img src="data:image/png;base64,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" style='float: left;' alt="Stack-Overflow">
		<h2 style='text-align:left; font-size:48px; font-weight:bold; color:#f6a733;'> Stack Overflow Topics</h2>
		<p style='text-algin:center; font-size:14px; color:'> An in-depth analysis of the Stack Overflow Questions Toll Series </p>
		</div>
		</body>
		""", unsafe_allow_html=True)
	
	radio = st.sidebar.radio("Navigation", ["Home", "Data Insights", "Know Specific Data", "Statistical Tests"])

	if radio == "Home":
		st.header("Top 100 records of the StackOverflow Dataset")
		cols = ["nltk", "spacy", "python", "r"]
		st_ms = st.multiselect("Topics", df.columns.tolist(), default=cols)
		st.dataframe(df[st_ms].head(100))

		st.markdown(f""" 

			The above Data-frame represents the top discussion topics and their tolls in StackOverflow from the year 2009-2019.
			The total Number of records in the dataset is {df.shape[0]} and the total number of topics that are being discussed 
			are {df.shape[1]}.
			""")

		st.markdown("The data is recorded at the begining of each month of each year.")
		st.markdown("")

		st.header("Percentage of Data Missing")
		miss_df = pd.DataFrame((df.isna().sum() / df.isna().count()) * 100, columns=["Percentage of Data Missing"])
		st.dataframe(miss_df)

		st.markdown("")
		st.header("Top 20 discussion topics in Stack Overflow over the year 2009-2019")
		st.image('./img/comparison.gif')


	if radio == "Data Insights":

		st.header("Comparison plots")
		cols = ['nltk', 'spacy']
		st_ms = st.multiselect("Topics", df.columns.tolist(), default=cols)
		fig = plot_interactive(df, st_ms)
		st.write(fig)

		# st.markdown("")
		# st.markdown("# Distribution Plot for the data")
		# fig2 = multiple_distribution_plots(df, st_ms)
		# st.pyplot(fig2)

		st.markdown("")
		df_sub1 = df[['python', 'r', 'matlab']].copy()
		df_sub1['year'] = df_sub1.index.year
		df_sub1 = df_sub1.groupby(by=['year']).mean()

		st.header("Comparison between Python, R, and Matlab Question Toll")
		fig2 = interactive_pie_chart(df_sub1)
		st.write(fig2)

		st.markdown("")
		st.header("Rise of Big-Data Libraries")
		fig3 = plot_interactive(df, ['hadoop', 'apache-spark', 'pyspark', 'Dask'])
		st.write(fig3)


	if radio == "Know Specific Data":

		option = st.selectbox('Select the Tag which needs to be analyzed', df.columns.tolist())

		df_sub2 = df[[option]]
		st.header(f'Data Properties of {option} tag')
		st.dataframe(df_sub2.describe().T)

		st.header(f"Distribution of the {option} tag")
		fig4 = box_dist(df_sub2, option, title=f'Distribution Plot of the {option}')
		st.pyplot(fig4)

		st.header(f'Trend and Seasonality in the {option} tag')
		ses_trend_res = seasonal_decompose(df_sub2[option], model='additive')
		fig5, ax5 = plt.subplots(3, 1, figsize=(12, 8))
		ax5[0].plot(ses_trend_res.trend)
		ax5[1].plot(ses_trend_res.seasonal)
		ax5[2].plot(ses_trend_res.resid)

		ax5[0].set_ylabel('Trend')
		ax5[1].set_ylabel('Seasonal')
		ax5[2].set_ylabel('Residual')

		st.pyplot(fig5)

		st.header(f"The pattern of the {option} tag")

		slider_val = st.slider('Window Length', min_value=3, max_value=10)
		
		with st.spinner('Calculating the Moving Average ....'):
			df_sub2['moving_average'] = apply_moving_average_filter(df_sub2[option], win_len=slider_val)

		fig6 = plot_interactive(df_sub2, [option, 'moving_average'], title=f'Pattern in {option} tag')
		st.write(fig6)


	if radio == "Statistical Tests":

		option = st.selectbox("Select the Tag which needs to be tested", df.columns.tolist())

		st.header('Augmented Dickey-Fuller Test Results')
		adtes_res, ad_is_stn = adfuller_test(df[option])
		st.dataframe(adtes_res)
		ad_is_stn = "Stationary" if ad_is_stn else "Non-Stationary"
		st.markdown(f'\n As per the Augmented Dickey Fuller the series for the tag **{option}** is considered to be **{ad_is_stn}**')

		st.markdown("")
		st.header('KPSS Test Results')
		option2 = st.selectbox('Select the Null Hypothesis for the KPSS Test', ['c', 'ct'])
		kpsstest_res, kpss_is_stn = kpss_test(df[option], reg=option2)
		st.dataframe(kpsstest_res)
		kpss_is_stn = "Stationary" if kpss_is_stn else "Non-Stationary"
		st.markdown(f""" 
			As per the KPSS Test on the series for the tag **{option}** is considered to be **{kpss_is_stn}**
			""")
