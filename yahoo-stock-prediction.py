import re, io, requests, pandas as pd, numpy as np
import datetime 
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.models import load_model

class YahooStockPrices:
	def __init__(self, initials, output, start_date, end_date):
		self.initials = initials
		self.start_date = datetime.datetime.timestamp(datetime.datetime.strptime(start_date, '%d-%m-%Y'))
		self.end_date = datetime.datetime.timestamp(datetime.datetime.strptime(end_date, '%d-%m-%Y'))
		self.output = io.open(output, 'w', encoding = 'utf-8')
		self.output.write('date,open,high,low,close,volume,adjclose\n')

	def data_download(self):

		self.data = requests.get('https://finance.yahoo.com/quote/{}/history?period1={}&period2={}&interval=1d&filter=history&frequency=1d'.format(self.initials, int(self.end_date), int(self.start_date)))
		self.regex_return = re.findall(r'(?<=prices":\[)(.*)(?=\],"isPending")', str(self.data.content), re.IGNORECASE)[0]
		self.regex_return = self.regex_return.split('{')
		self.regex_return = [i.replace('}', '') for i in self.regex_return if i != '']
 
		for result in self.regex_return:
			try: self.date = str(re.findall(r'(?<="date":)(.*)(?=,"open")', str(result), re.IGNORECASE)[0]).strip()
			except: self.date = ''

			try: self.open = str(re.findall(r'(?<="open":)(.*)(?=,"high")', str(result), re.IGNORECASE)[0]).strip()
			except: self.open = ''

			try: self.high = str(re.findall(r'(?<="high":)(.*)(?=,"low")', str(result), re.IGNORECASE)[0]).strip()
			except: self.high = ''

			try: self.low = str(re.findall(r'(?<="low":)(.*)(?=,"close")', str(result), re.IGNORECASE)[0]).strip()
			except: self.low = ''

			try: self.close = str(re.findall(r'(?<="close":)(.*)(?=,"volume")', str(result), re.IGNORECASE)[0]).strip()
			except: self.close = ''

			try: self.volume = str(re.findall(r'(?<="volume":)(.*)(?=,"adjclose")', str(result), re.IGNORECASE)[0]).strip()
			except: self.volume = ''

			try: self.adjclose = str(re.findall(r'(?<="adjclose":)(.*)(?=,)', str(result), re.IGNORECASE)[0]).strip()
			except: self.adjclose = ''

			if self.date != '':
				self.output.write(self.date + ',' + self.open + ',' + self.high + ',' + self.low + ',' + self.close + ',' + self.volume + ',' + self.adjclose + '\n')

class StockPrediction:

	def __init__(self, fInput):
		self.fInput = pd.read_csv(fInput)
		self.fInput['date'] = pd.to_datetime(self.fInput['date'], unit = 's').dt.date
		self.fInput = self.fInput.sort_index(axis = 0 ,ascending = False)
		self.scaler = MinMaxScaler(feature_range = (0, 1))

	def x_y_preperation(self, dataset, timesteps):
		self.x, self.y = [], []
		for t in range(timesteps, len(dataset)):
			self.x.append(dataset[(t - 60) : t, 0])
			self.y.append(dataset[t, 0])
		self.x, self.y = np.reshape(np.array(self.x), (np.array(self.x).shape[0], np.array(self.x).shape[1], 1)), np.array(self.y)
		return self.x, self.y

	def train_model(self):
		self.close_prices = self.scaler.fit_transform(self.fInput.filter(['close']).values)
		self.X, self.Y = self.x_y_preperation(self.close_prices, 60)

		self.model = Sequential()
		
		self.model.add(LSTM(units = 600, return_sequences = True, input_shape = (self.X.shape[1], 1)))
		self.model.add(Dropout(0.2))
		
		self.model.add(LSTM(units = 60, return_sequences = True))
		self.model.add(Dropout(0.1))

		self.model.add(LSTM(units = 60, return_sequences = False))
		self.model.add(Dropout(0.1))

		self.model.add(Dense(units = 1))
		self.model.compile(optimizer = 'adam', loss = 'mean_squared_error')
		self.model.fit(self.X, self.Y, epochs = 25, batch_size = 10)

		self.model.save('yahoo_model.h5')
		self.model.save_weights('yahoo_model_weight.h5')

class PredictPrice:
	def __init__(self, initials, start_date, h5model):
		self.initials = initials
		self.start_date = start_date
		self.end_date = datetime.datetime.strptime((datetime.datetime.strptime(self.start_date, '%d-%m-%Y') - datetime.timedelta(days = 100)).isoformat(), '%Y-%m-%dT%H:%M:%S').strftime("%d-%m-%Y")
		self.scaler = MinMaxScaler(feature_range = (0, 1))
		self.model = load_model(h5model)

	def x_preparation(self, dataset, timesteps):
		self.x, self.y = [], []
		for t in range(timesteps, len(dataset)):
			self.x.append(dataset[(t - timesteps) : t, 0])
			self.y.append(dataset[t, 0])
		self.x = np.reshape(np.array(self.x), (np.array(self.x).shape[0], np.array(self.x).shape[1], 1))
		return self.x

	def predict(self):
		self.start_date = str(datetime.datetime.strptime(self.start_date, '%d-%m-%Y').strftime('%d-%m-%Y'))
		self.end_date = str(datetime.datetime.strptime(self.end_date, '%d-%m-%Y').strftime('%d-%m-%Y'))
		YahooStockPrices(initials = self.initials,  output = 'prediction_dataset.csv', start_date = self.start_date, end_date = self.end_date).data_download()
		self.dataset = pd.read_csv('prediction_dataset.csv')
		self.close_prices = self.scaler.fit_transform(self.dataset.filter(['close']).values)
		self.close_prices = self.close_prices[0:61]

		self.X = self.x_preparation(self.close_prices, 60)
		self.prediction = self.model.predict(self.X)

		return self.scaler.inverse_transform(self.prediction)

## Download training dataset
# YahooStockPrices(initials = 'AAPL',  output = 'train.csv', start_date = '03-03-2020', end_date = '01-01-2010').data_download()

## Training the model 
# StockPrediction(fInput = 'train.csv').train_model()

## Predict data
y_prediction = PredictPrice(initials = 'AAPL', start_date = '02-03-2020', h5model = 'yahoo_model.h5').predict()
print ('Predicted price : ', y_prediction[0][0])
