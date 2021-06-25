import pickle, json
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow import keras

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

model = keras.models.load_model('models')

def predict(term, usd_goal, category, blurb, subcategory):
	data = [term, usd_goal, category, blurb, subcategory]
	df = pd.DataFrame(data=[data])
	maxlen = 43 # Somewhat arbitrary at this point, neccessary however
	seq = tokenizer.texts_to_sequences(df[3])
	# seq2 = [i for s in seq for i in s]
	padded_sequence = sequence.pad_sequences(seq, maxlen)
	# Convert text to df columns (inelegant, but effective)
	for j in range(5, 48):
		f = (j-5)
		df[j] = [padded_sequence[0][f]]
	# Drop unencoded text column
	df.drop(columns=[3], inplace=True)
	# Convert df to numpy array, for further conversion to tensor
	arr = df.to_numpy()
	# Reshape array into tensor
	tarr = arr.reshape(1, 47, 1)
	# Convert all types to float32
	tarr = np.asarray(tarr).astype('float32')
	# Save prediction to variable
	y_pred = model.predict(tarr)[0][0]
	# return prediction
	return str(y_pred)
