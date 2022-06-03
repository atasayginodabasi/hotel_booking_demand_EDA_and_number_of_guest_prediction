import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from pmdarima import auto_arima
import warnings
from sklearn.metrics import mean_absolute_percentage_error

warnings.filterwarnings("ignore")

# Reading the data
print(os.listdir(r'C:/Users/ata-d/'))
data = pd.read_csv(r'C:/Users/ata-d/OneDrive/Masaüstü/ML/Datasets/hotel_bookings.csv')
data.info()

# -------------------------------------------------------------------------------------------------------

# Dropping certain columns
data = data.drop(['company', 'agent'], axis=1)
data = data.dropna(subset=['country', 'children', 'arrival_date_week_number'], axis=0)
data = data.reset_index(drop=True)

data['children'] = data['children'].astype(int)

# Checking for the missing values
NaN = data.isna().sum()

# Checking for the missing values after drops
NaN_updated = data.isna().sum()

# -------------------------------------------------------------------------------------------------------

# Converting string month to numerical one (Dec = 12, Jan = 1, etc.)
datetime_object = data['arrival_date_month'].str[0:3]
month_number = np.zeros(len(datetime_object))

# Creating a new column based on numerical representation of the months
for i in range(0, len(datetime_object)):
    datetime_object[i] = datetime.datetime.strptime(datetime_object[i], "%b")
    month_number[i] = datetime_object[i].month

# Float to integer conversion
month_number = pd.DataFrame(month_number).astype(int)

# 3 columns are merged into one
data['arrival_date'] = data['arrival_date_year'].map(str) + '-' + month_number[0].map(str) + '-' \
                       + data['arrival_date_day_of_month'].map(str)

# Dropping already used columns
data = data.drop(['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month',
                  'arrival_date_week_number'], axis=1)

# Converting wrong datatype columns to correct type (object to datetime)
data['arrival_date'] = pd.to_datetime(data['arrival_date'])
data['reservation_status_date'] = pd.to_datetime(data['reservation_status_date'])

# -------------------------------------------------------------------------------------------------------

# Calculating total guests for each record
data['Total Guests'] = data['adults'] + data['children']

# Some data points include zero Total Guests, therefore I dropped them
data = data[data['Total Guests'] != 0]

# Total Number of Days Stayed
data['Total Stays'] = data['stays_in_weekend_nights'] + data['stays_in_week_nights']

dataNoCancel = data[data['is_canceled'] == 0]
dataNoCancel = dataNoCancel.reset_index(drop=True)

# Creating two dataframes include only discrete hotel type
dataResort = data[data['hotel'] == 'Resort Hotel']
dataCity = data[data['hotel'] == 'City Hotel']

data = data.reset_index(drop=True)

# -------------------------------------------------------------------------------------------------------

# Calculating Number of Guests Weekly - Resort Hotel
NumberOfGuests_Resort = dataResort[['arrival_date', 'Total Guests']]
NumberOfGuests_ResortWeekly = dataResort['Total Guests'].groupby(dataResort['arrival_date']).sum()
NumberOfGuests_ResortWeekly = NumberOfGuests_ResortWeekly.resample('w').sum().to_frame()

# Calculating Number of Guests Weekly - City Hotel
NumberOfGuests_City = dataCity[['arrival_date', 'Total Guests']]
NumberOfGuests_CityWeekly = dataCity['Total Guests'].groupby(dataCity['arrival_date']).sum()
NumberOfGuests_CityWeekly = NumberOfGuests_CityWeekly.resample('w').sum().to_frame()

# -------------------------------------------------------------------------------------------------------

# Number of Records by Country (both Cancelled and Showed Up)
'''''''''
country_freq = data['country'].value_counts().to_frame()
country_freq.columns = ['count']
fig = px.choropleth(country_freq, color='count',
                    locations=country_freq.index,
                    hover_name=country_freq.index,
                    color_continuous_scale=px.colors.sequential.Teal)
fig.update_traces(marker=dict(line=dict(color='#000000', width=1)))
fig.update_layout(title_text='Number of Records by Countries',
                  title_x=0.5, title_font=dict(size=30))  # Location and the font size of the main title
fig.show()
'''''''''

# Number of Records Monthly
'''''''''
dataResortMonthly = dataResort['arrival_date'].value_counts()
dataResortMonthly = dataResortMonthly.resample('m').sum().to_frame()

dataCityMonthly = dataCity['arrival_date'].value_counts()
dataCityMonthly = dataCityMonthly.resample('m').sum().to_frame()

fig = go.Figure()
fig.add_trace(go.Scatter(x=dataResortMonthly.index, y=dataResortMonthly['arrival_date'], name="Resort Hotel",
                         hovertext=dataResortMonthly['arrival_date']))
fig.add_trace(go.Scatter(x=dataCityMonthly.index, y=dataCityMonthly['arrival_date'], name="City Hotel",
                         hovertext=dataCityMonthly['arrival_date']))
fig.update_layout(title_text='Number of Records Monthly',
                  title_x=0.5, title_font=dict(size=30))  # Location and the font size of the main title
fig.update_layout(
    xaxis_title="Arrival Date",
    yaxis_title="Number of Records")

fig.show()
'''''''''

# Number of Records Weekly
'''''''''
dataResortWeekly = dataResort['customer_type'].groupby(dataResort['arrival_date']).count()
dataResortWeekly = dataResortWeekly.resample('w').sum().to_frame()

fig = px.line(dataResortWeekly, x=dataResortWeekly.index, y=dataResortWeekly['customer_type'])
fig.update_layout(title_text='Number of Records Weekly - Resort Hotel',
                  title_x=0.5, title_font=dict(size=30))  # Location and the font size of the main title
fig.show()

dataCityWeekly = dataCity['customer_type'].groupby(dataCity['arrival_date']).count()
dataCityWeekly = dataCityWeekly.resample('w').sum().to_frame()

fig = px.line(dataCityWeekly, x=dataCityWeekly.index, y=dataCityWeekly['customer_type'])
fig.update_layout(title_text='Number of Records Weekly - City Hotel',
                  title_x=0.5, title_font=dict(size=30))  # Location and the font size of the main title
fig.show()
'''''''''

# Number of Guests Weekly
'''''''''
fig = go.Figure()
fig.add_trace(go.Scatter(x=NumberOfGuests_ResortWeekly.index, y=NumberOfGuests_ResortWeekly['Total Guests'],
                         name="Resort Hotel",
                         hovertext=NumberOfGuests_ResortWeekly['Total Guests']))

fig.add_trace(go.Scatter(x=NumberOfGuests_CityWeekly.index, y=NumberOfGuests_CityWeekly['Total Guests'],
                         name="City Hotel",
                         hovertext=NumberOfGuests_CityWeekly['Total Guests']))

fig.update_layout(title_text='Number of Guests Weekly',
                  title_x=0.5, title_font=dict(size=30))  # Location and the font size of the main title
fig.update_layout(
    xaxis_title="Arrival Date",
    yaxis_title="Number of Guests")

fig.show()
'''''''''

# Number of Guests by Customer Type
'''''''''
customerTransient = data[data['customer_type'] == 'Transient']
customerContract = data[data['customer_type'] == 'Contract']
customerTransientParty = data[data['customer_type'] == 'Transient-Party']
customerGroup = data[data['customer_type'] == 'Group']

customerTransient = customerTransient.set_index("arrival_date")
customerContract = customerContract.set_index("arrival_date")
customerTransientParty = customerTransientParty.set_index("arrival_date")
customerGroup = customerGroup.set_index("arrival_date")

customerTransientMonthly = customerTransient.resample('m').sum()
customerContract = customerContract.resample('m').sum()
customerTransientParty = customerTransientParty.resample('m').sum()
customerGroup = customerGroup.resample('m').sum()

fig = go.Figure()
fig.add_trace(go.Scatter(x=customerTransientMonthly.index, y=customerTransientMonthly['Total Guests'],
                         name="Transient Guests",
                         ))
fig.add_trace(go.Scatter(x=customerContract.index, y=customerContract['Total Guests'],
                         name="Contract Guests",
                         ))
fig.add_trace(go.Scatter(x=customerTransientParty.index, y=customerTransientParty['Total Guests'],
                         name="Transient-Party Guests",
                         ))
fig.add_trace(go.Scatter(x=customerGroup.index, y=customerGroup['Total Guests'],
                         name="Group Guests",
                         ))
fig.update_layout(title_text='Number of Guests by Customer Type',
                  title_x=0.5, title_font=dict(size=30))  # Location and the font size of the main title
fig.update_layout(
    xaxis_title="Arrival Date",
    yaxis_title="Number of Guests")

fig.show()
'''''''''

# Number of Guests by Market Segment
'''''''''
customerOnline = data[data['market_segment'] == 'Online TA']
customerDirect = data[data['market_segment'] == 'Direct']
customerOffline = data[data['market_segment'] == 'Offline TA/TO']

customerOnline = customerOnline.set_index("arrival_date")
customerDirect = customerDirect.set_index("arrival_date")
customerOffline = customerOffline.set_index("arrival_date")

customerOnline = customerOnline.resample('m').sum()
customerDirect = customerDirect.resample('m').sum()
customerOffline = customerOffline.resample('m').sum()

fig = go.Figure()
fig.add_trace(go.Scatter(x=customerOnline.index, y=customerOnline['Total Guests'],
                         name="Online TA Segment Guests",
                         ))
fig.add_trace(go.Scatter(x=customerDirect.index, y=customerDirect['Total Guests'],
                         name="Direct Segment Guests",
                         ))
fig.add_trace(go.Scatter(x=customerOffline.index, y=customerOffline['Total Guests'],
                         name="Offline TA/TO Segment Guests",
                         ))
fig.update_layout(title_text='Number of Guests by Market Segment',
                  title_x=0.5, title_font=dict(size=30))  # Location and the font size of the main title
fig.update_layout(
    xaxis_title="Arrival Date",
    yaxis_title="Number of Guests")

fig.show()
'''''''''

# Distribution of Market Segment by different Hotel type
'''''''''
fig = px.histogram(data, x="market_segment", color='hotel')
fig.update_layout(barmode='group', xaxis={'categoryorder': 'total descending'})
fig.update_traces(marker=dict(line=dict(color='#000000', width=1)))
fig.update_layout(title_text='Distribution of Market Segment by different Hotel Types',
                  title_x=0.5, title_font=dict(size=20))  # Location and the font size of the main title
fig.show()
'''''''''

# Distribution of the Reservation Status
'''''''''
reservation_status = data['reservation_status'].value_counts()
fig = go.Figure(data=[go.Pie(labels=reservation_status.index, values=reservation_status, opacity=0.8)])
fig.update_traces(textinfo='percent+label', marker=dict(line=dict(color='#000000', width=2)))
fig.update_layout(title_text='Distribution of the Reservation Status', title_x=0.5, title_font=dict(size=32))
fig.show()
'''''''''

# Distribution of Room Type and ADR
'''''''''
fig = px.box(data_frame=dataNoCancel, x='reserved_room_type', y='adr', color='hotel')
fig.update_layout(title_text='Distribution of Room Type and ADR',
                  title_x=0.5, title_font=dict(size=20))
fig.update_traces(marker=dict(line=dict(color='#000000', width=1)))
fig.show()
'''''''''

# Distribution of Room Meal and ADR
'''''''''
fig = px.box(data_frame=dataNoCancel, x='meal', y='adr', color='hotel')
fig.update_traces(marker=dict(line=dict(color='#000000', width=1)))
fig.update_layout(title_text='Distribution of Meal by ADR',
                  title_x=0.5, title_font=dict(size=20))
fig.show()
'''''''''

# Did the Guests get the Reserved Room?
'''''''''
dataResort['TookReservedRoom'] = np.where(dataResort['reserved_room_type'] == dataResort['assigned_room_type'],
                                          'Yes', 'No')
dataCity['TookReservedRoom'] = np.where(dataCity['reserved_room_type'] == dataCity['assigned_room_type'],
                                        'Yes', 'No')

fig = go.Figure(data=[go.Pie(labels=dataResort['TookReservedRoom'].unique(),
                             values=dataResort['TookReservedRoom'].value_counts(), opacity=0.9)])
fig.update_traces(textinfo='percent+label', marker=dict(line=dict(color='#000000', width=2)))
fig.update_layout(title_text='Did the Guests get the Reserved Room? - Resort Hotel',
                  title_x=0.5, title_font=dict(size=32))
fig.show()

fig = go.Figure(data=[go.Pie(labels=dataCity['TookReservedRoom'].unique(),
                             values=dataCity['TookReservedRoom'].value_counts(), opacity=0.9)])
fig.update_traces(textinfo='percent+label', marker=dict(line=dict(color='#000000', width=2)))
fig.update_layout(title_text='Did the Guests get the Reserved Room? - City Hotel',
                  title_x=0.5, title_font=dict(size=32))
fig.show()
'''''''''

# Density Plot of Number of Days Stayed for different Hotel Types
'''''''''
plt.figure(figsize=(15, 8))
sns.distplot(dataResort['Total Stays'], color='blue')
sns.distplot(dataCity['Total Stays'], color='red')
plt.xlabel("Number of Days Stayed", fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(['Resort Hotel', 'City Hotel'], loc='upper right')
plt.title("Density Plot of Number of Days Stayed for different Hotel Types", fontsize=16)
'''''''''

# Correlation Graph City Hotel
'''''''''
plt.figure(figsize=(15, 8))
correlation = sns.heatmap(dataCity.corr(), vmin=-1, vmax=1, annot=True, linewidths=1, linecolor='black')
correlation.set_title('Correlation Graph of the City Hotel', fontdict={'fontsize': 24})
'''''''''

# Correlation Graph Resort Hotel
'''''''''
plt.figure(figsize=(15, 8))
correlation = sns.heatmap(dataResort.corr(), vmin=-1, vmax=1, annot=True, linewidths=1, linecolor='black')
correlation.set_title('Correlation Graph of the Resort Hotel', fontdict={'fontsize': 24})
'''''''''

# -------------------------------------------------------------------------------------------------------
# ARIMA Model for Predicting Future Number of Guests [CITY HOTEL]

# Dickey-Fuller Test to City Hotel Data
'''
CityWeeklyValues = NumberOfGuests_CityWeekly.values
result_city = adfuller(CityWeeklyValues)
print('ADF Statistic: %f' % result_city[0])
print('p-value: %f' % result_city[1])
print('Critical Values:')
for key, value in result_city[4].items():
    print('\t%s: %.3f' % (key, value))
'''

# Rolling Mean & Rolling Standard Deviation of City Hotel
'''''''''
plt.figure(figsize=(15, 8))
rolling_mean = NumberOfGuests_CityWeekly.rolling(window=4).mean()
rolling_std = NumberOfGuests_CityWeekly.rolling(window=4).std()
plt.plot(NumberOfGuests_CityWeekly, color='blue', label='Original')
plt.plot(rolling_mean, color='red', label='Rolling Mean')
plt.plot(rolling_std, color='black', label='Rolling Std')
plt.legend(loc='upper right')
plt.title('Rolling Mean & Rolling Standard Deviation of the Weekly Number of Guests - City Hotel')
plt.show()
'''''''''

trainCity = NumberOfGuests_CityWeekly[:90]
testCity = NumberOfGuests_CityWeekly[90:]

# Fit auto_arima function to NumberOfGuests_CityWeekly Dataset
'''''''''
stepwise_fit = auto_arima(trainCity['Total Guests'], start_p=1, start_q=1,
                          max_p=3, max_q=3, m=12,
                          start_P=0, seasonal=True,
                          d=None, D=1, trace=True,
                          error_action='ignore',
                          suppress_warnings=True,
                          stepwise=True)

stepwise_fit.summary()
'''''''''

# -------------------------------------------------------------------------------------------------------

modelCity = ARIMA(trainCity['Total Guests'],
                  order=(3, 0, 0),
                  seasonal_order=(2, 1, 0, 12))

resultCity = modelCity.fit()
resultCity.summary()

# Prediction of the Test data
predictionsCityTest = resultCity.predict(90, 113, typ='levels').rename("Predictions")

# Prediction of Number of Guests (Test Data)
'''''''''
testCity['Total Guests'].plot(legend=True)
predictionsCityTest.plot(legend=True)
plt.title('Prediction of Number of Guests for City Hotel (Test Data)', fontsize=16)
plt.xlabel('Arrival Date', fontsize=12)
plt.ylabel('Number of Guests', fontsize=12)
'''''''''

# Prediction of Number of Guests
'''''''''
trainCity['Total Guests'].plot(legend=True)
predictionsCityTest.plot(legend=True)
plt.title('Prediction of Number of Guests for City Hotel', fontsize=16)
plt.xlabel('Arrival Date', fontsize=12)
plt.ylabel('Number of Guests', fontsize=12)
'''''''''

MeanAbsPercentageErrCity_test = mean_absolute_percentage_error(testCity, predictionsCityTest)
print('Test MAPE City Hotel: %f' % MeanAbsPercentageErrCity_test)

# -------------------------------------------------------------------------------------------------------

# Dickey-Fuller Test to City Resort Data
'''
ResortWeeklyValues = NumberOfGuests_ResortWeekly.values
result_resort = adfuller(ResortWeeklyValues)
print('ADF Statistic: %f' % result_resort[0])
print('p-value: %f' % result_resort[1])
print('Critical Values:')
for key, value in result_resort[4].items():
    print('\t%s: %.3f' % (key, value))
'''

# Rolling Mean & Rolling Standard Deviation of Resort Hotel
'''''''''
plt.figure(figsize=(15, 8))
rolling_mean = NumberOfGuests_ResortWeekly.rolling(window=4).mean()
rolling_std = NumberOfGuests_ResortWeekly.rolling(window=4).std()
plt.plot(NumberOfGuests_ResortWeekly, color='blue', label='Original')
plt.plot(rolling_mean, color='red', label='Rolling Mean')
plt.plot(rolling_std, color='black', label='Rolling Std')
plt.legend(loc='upper right')
plt.title('Rolling Mean & Rolling Standard Deviation of the Weekly Number of Guests - Resort Hotel')
plt.show()
'''''''''

trainResort = NumberOfGuests_ResortWeekly[:90]
testResort = NumberOfGuests_ResortWeekly[90:]

# Fit auto_arima function to NumberOfGuests_ResortWeekly Dataset
'''''''''
stepwise_fit = auto_arima(trainResort['Total Guests'], start_p=1, start_q=1,
                          max_p=3, max_q=3, m=12,
                          start_P=0, seasonal=True,
                          d=None, D=1, trace=True,
                          error_action='ignore',
                          suppress_warnings=True,
                          stepwise=True)

stepwise_fit.summary()
'''''''''

modelResort = ARIMA(trainResort['Total Guests'],
                    order=(2, 0, 0),
                    seasonal_order=(2, 1, 0, 12))

resultResort = modelResort.fit()
resultResort.summary()

# Prediction of the Test data
predictionsResortTest = resultResort.predict(90, 113, typ='levels').rename("Predictions")

# Prediction of Number of Guests (Test Data)
'''''''''
testResort['Total Guests'].plot(legend=True)
predictionsResortTest.plot(legend=True)
plt.title('Prediction of Number of Guests for Resort Hotel (Test Data)', fontsize=16)
plt.xlabel('Arrival Date', fontsize=12)
plt.ylabel('Number of Guests', fontsize=12)
'''''''''

# Prediction of Number of Guests
'''''''''
trainResort['Total Guests'].plot(legend=True)
predictionsResortTest.plot(legend=True)
plt.title('Prediction of Number of Guests for Resort Hotel', fontsize=16)
plt.xlabel('Arrival Date', fontsize=12)
plt.ylabel('Number of Guests', fontsize=12)
'''''''''

MeanAbsPercentageErrResort_test = mean_absolute_percentage_error(testResort, predictionsResortTest)
print('Test MAPE Resort Hotel: %f' % MeanAbsPercentageErrResort_test)
