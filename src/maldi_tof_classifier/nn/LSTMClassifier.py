import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy

def LSTMClassifier(X, y, lstm_units: int = 32) -> keras.Model:
	"""
	Builds a LSTM classifier model for classifying 1D vectors from raw or
	preprocessed spectra.

	Parameters
	-----------
	X
		Array of multiple spectra (rows).
	y
		Array of one hot encoded classes of spectra.
	lstm_units
		The number of neurons in the LSTM layer.
		
	Returns
	--------
	The compiled LSTM classifier model.
	"""
	model = Sequential()
	model.add(LSTM(lstm_units, input_shape=(X.shape[1], 1)))
	model.add(Dense(y.shape[1], activation="softmax"))

	model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=["accuracy"])

	return model
