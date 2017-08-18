import numpy as np
import pandas as pd
import os
import tarfile
import hashlib
from six.moves import urllib
import matplotlib.pyplot as plt
from datasets.DownloadPaths import *
import hashlib

def fetch_data(url, local_path, file_name):
    '''
    fetch the housing data from the given url to the given local path
    and extract all.
    ''' 
    if not os.path.isdir(local_path):

        os.makedirs(local_path)
        tgz_path = os.path.join(local_path, file_name)

        urllib.request.urlretrieve(url, tgz_path)
        housing_tgz = tarfile.open(tgz_path)

        housing_tgz.extractall(path=local_path)
        housing_tgz.close()

def load_data(local_path, file_name):
    '''
    turn housing csv file into a pandas data frame
    ''' 
    csv_path = os.path.join(local_path, file_name) 
    return pd.read_csv(csv_path)

def split_train_test(data, test_ratio=0.2, random_seed=42):
    '''
    split data into train and test sets. gets random permutaion of indices
    and extracts a test set from them according to given ratio.
    '''
    np.random.seed(seed=random_seed)
    shuffled_indices = np.random.permutation(len(data)) 
    test_set_size = int(len(data) * test_ratio)

    test_indices = shuffled_indices[:test_set_size] 
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

def test_set_check(identifier, test_ratio, hash):
    '''
    given an id, returns true if the last byte of the hash code of that id
    is less than 256 * test_ratio 
    '''
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    '''
    splits data into train and test set according to an id column. if 
    id doesn't change, this ensures that same instances will remain in the test set
    when the data set updates
    '''
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash)) 
    
    return data.loc[~in_test_set], data.loc[in_test_set]

if __name__ == '__main__':
    fetch_data(HOUSING_URL, HOUSING_PATH, 'housing.tgz')
    housing = load_data(HOUSING_PATH, 'housing.csv')

    # train_set, test_set = split_train_test(housing, 0.2)

    housing_with_id = housing.reset_index() # adds an `index` column 
    train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

    print(housing.head())
    print (housing.info())
    print (housing.describe())
    # housing.hist(bins=50, figsize=(20,15))
    # plt.show()


