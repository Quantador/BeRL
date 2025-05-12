import pickle

def pref_save(pref_data, name):
    with open(name, 'wb') as f:
        pickle.dump(pref_data, f, pickle.HIGHEST_PROTOCOL)

def pref_load(name):
    with open(name, 'rb') as f:
        return pickle.load(f)