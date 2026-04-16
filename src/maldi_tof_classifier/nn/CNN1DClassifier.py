import math
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPool1D, GlobalMaxPool1D
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy

def CNN1D_Classifier(X, y, filters: int = 3, kernel_size: int = 32,
					 use_max_pool: bool = False, dense_units: int = 128
					 ) -> keras.Model:
	"""
	Builds a 1D CNN classifier model for classifying 1D vectors from raw or
	preprocessed spectra.

	Parameters
	-----------
	X
		Array of multiple spectra (rows).
	y
		Array of one hot encoded classes of spectra.
	filters
		Number of convolution filters used.
	kernel_size
		Size of the convolution filters.
	use_max_pool
		Specifies whether a max pooling layer is added before the dense layer.
	dense_units
		Number of units used in densely connected layer.

	Returns
	--------
	The compiled 1D CNN classifier model.
	"""
	model = Sequential()
	model.add(Conv1D(filters=filters, kernel_size=kernel_size,
					 input_shape=(X.shape[1], 1)))
	if use_max_pool:
		model.add(MaxPool1D(pool_size=math.ceil(kernel_size/4)))
	model.add(GlobalMaxPool1D())
	model.add(Dense(dense_units, activation="relu"))
	model.add(Dense(y.shape[1], activation="softmax"))

	model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=["accuracy"])

	return model
