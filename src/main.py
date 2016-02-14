import classify, vis

# global variables
FRAMES_PER_SONG = 1198

def test_model(data, model):
	genres = ['blues', 'classical', 'country', 'disco', 'hiphop', \
        'jazz', 'metal', 'pop', 'reggae', 'rock']

	# only predict genres for those not used by mask
    predictions = model.predict(data)

    # convert to words
    predictions = [genres[int(genre)-1] for genre in guessed_genres]

    return predictions

def make_mask(n_total, n_mask):
	# select training set using boolean mask of a random subset
    training_indices = random.sample(range(n_total, n_mask))

    mask = np.zeros(n_total, dtype=bool)
    mask[training_indices] = True

    return mask

def classify_1(data):
	# extract strength of dominant key
	key_strength = np.zeros([len(data), FRAMES_PER_SONG])
	for i in range(len(data)):
		key = data[i]['key'][0][0][0][0]
		key_strength[i,:] = data[i]['keystrength'][0][0][key-1,:]

	# extract only mean + std
	key_strength = classify.summarize(key_strength)

	# extract tempo
	tempo = np.asarray([data[i]['tempo'][0][0][0][0] for i in range(len(data))])
	tempo = np.transpose(tempo)

	# combine arrays:
	params = hstack(key_strength, tempo)

	# get correct genre assignment
	classes = np.asarray([data[i]['class'][0][0][0][0] for i in range(len(data))])
	classes = np.transpose(genre)

	# train
	model = simple_classify.build_model(params, classes)

	# test
	return test_model(data, model)



if __name__ == '__main__':
	# load 
	data = simple_classify.load_songs()
