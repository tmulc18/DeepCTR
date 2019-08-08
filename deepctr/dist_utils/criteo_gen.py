import json
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr.inputs import  SparseFeat, DenseFeat,get_fixlen_feature_names

def get_total_examples(list_files):
    """Returns the total number of training examples
    from list of csv files.
    """
    total_lines = 0
    for file in list_files:
        with open(file) as f:
            for i, l in enumerate(f):
                pass
        total_lines += i
    return total_lines

class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras.
    Modified from: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """
    def __init__(self, list_files, nexamples, 
        file_index=0, 
        batch_size=256, 
        shuffle=True, 
        dense_meta = "stats/competition-dense-meta.json",
        cat_meta = "stats/competition-categorical-meta.json"
        ):
        'Initialization'
        self.batch_size = batch_size
        self.list_files = list_files
        self.shuffle = shuffle
        self.on_epoch_end()

        self.nexamples = nexamples # number of training examples
        self.num_files = len(list_files)

        # metadata
        self.sparse_features = ['C' + str(i) for i in range(1, 27)]
        self.dense_features = ['I' + str(i) for i in range(1, 14)]

        self.target = ['Label']

        # load meta data
        self.dense_meta = json.load(open(dense_meta,"r"))
        self.cat_meta = json.load(open(cat_meta,"r"))
        self.load_stats()

        print(self.cat_meta)

        # initialize file
        self.file_index = file_index
        self.load_data()


    def load_data(self):
        print("loading new file")
        self.cur_file = self.list_files[self.file_index%self.num_files]
        self.cur_df = pd.read_csv(self.cur_file)

        # data transform.  should remove and do in preprocessing
        self.cur_df[self.sparse_features] = self.cur_df[self.sparse_features].fillna('-1', )
        self.cur_df[self.dense_features] = self.cur_df[self.dense_features].fillna(0, )

        # # 1.Label Encoding for sparse features,and do simple Transformation for dense features
        # for feat in self.sparse_features:
        #     lbe = LabelEncoder()
        #     self.cur_df[feat] = lbe.fit_transform(self.cur_df[feat])
        # mms = MinMaxScaler(feature_range=(0, 1))
        # self.cur_df[self.dense_features] = mms.fit_transform(self.cur_df[self.dense_features])

        # # 2.count #unique features for each sparse field,and record dense feature field name

        # fixlen_feature_columns = [SparseFeat(feat, self.cur_df[feat].nunique())
        #                    for feat in self.sparse_features] + [DenseFeat(feat, 1,)
        #                   for feat in self.dense_features]

        # self.dnn_feature_columns = fixlen_feature_columns
        # self.linear_feature_columns = fixlen_feature_columns

        # fixlen_feature_names = get_fixlen_feature_names(self.linear_feature_columns + self.dnn_feature_columns)

        # done data transform

        self.cur_array = [self.cur_df[name] for name in self.fixlen_feature_names]#self.cur_df.as_matrix()
        self.cur_values = self.cur_df[self.target].values

        self.indexes = np.arange(len(self.cur_df))
        self.index_ = 0


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.nexamples / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[self.index_*self.batch_size:(self.index_+1)*self.batch_size]
        X = [feat[indexes] for feat in self.cur_array]
        y = self.cur_values[indexes]

        self.index_ += 1

        # move onto the next file
        if (self.index_)*self.batch_size > len(self.indexes):
            self.file_index += 1
            self.load_data()

        # Find list of IDs
        #list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        #X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        pass

    def load_stats(self):
        fixlen_feature_columns = [SparseFeat(feat, self.cat_meta[feat])
                           for feat in self.sparse_features] + [DenseFeat(feat, 1,)
                          for feat in self.dense_features]

        self.dnn_feature_columns = fixlen_feature_columns
        self.linear_feature_columns = fixlen_feature_columns

        self.fixlen_feature_names = get_fixlen_feature_names(self.linear_feature_columns + self.dnn_feature_columns)


#    def on_epoch_end(self):
#        'Updates indexes after each epoch'
#        self.indexes = np.arange(len(self.list_IDs))
#        if self.shuffle == True:
#            np.random.shuffle(self.indexes)

#    def __data_generation(self):
#        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
#        # Initialization
#        #X = np.empty((self.batch_size, *self.dim, self.n_channels))
#        #y = np.empty((self.batch_size), dtype=int)
#
#        # Generate data
#        for i, ID in enumerate(list_IDs_temp):
#            # Store sample
#            X[i,] = np.load('data/' + ID + '.npy')
#
#            # Store class
#            y[i] = self.labels[ID]
