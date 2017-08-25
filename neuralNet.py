from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import numpy as np

X_train = np.load('metadata/x_train_probs.npy')
Y_train = np.load('metadata/y_train_probs.npy')
X_test = np.load('metadata/x_test_probs.npy')
Y_test = np.load('metadata/y_test_probs.npy')

# define base mode
def baseline_model():
    # create model
	model = Sequential()
	model.add(Dense(500, input_dim=125, init='normal', activation='relu'))
	model.add(Dense(150, init='normal', activation='relu'))
	model.add(Dense(50, init='normal', activation='relu'))
	model.add(Dense(1, init='normal'))
	# Compile model
	model.compile(loss='mse', optimizer='rmsprop')
	return model

seed = 7
np.random.seed(seed)
model = baseline_model()

# evaluate model with standardized dataset
model.fit(X_train, Y_train, nb_epoch=75, batch_size=32, verbose=1, validation_data=(X_test,Y_test))

loss = model.evaluate(X_test, Y_test, batch_size=400, verbose=1)
print "Loss: " + str(loss)
