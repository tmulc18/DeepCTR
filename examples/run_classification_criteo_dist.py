import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ProgbarLogger, ModelCheckpoint
from tensorflow import keras as keras
import tensorflow as tf
from sklearn.metrics import roc_auc_score

def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)
#from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr.models import DeepFM
#from deepctr.inputs import  SparseFeat, DenseFeat,get_fixlen_feature_names
from deepctr.dist_utils.criteo_gen import DataGenerator, get_total_examples
import argparse
import glob

def main(dataPath,dataPath_val,batch_size):

    # must have list of training files
    files = glob.glob(dataPath+"/*.csv")[::5]

    # validation files
    files_val = glob.glob(dataPath_val+"/*.csv")[::5]
    
    # Count number of examples in training data
    nexs = get_total_examples(files)
    print("Number of training examples: ", nexs)

    nexs_val = get_total_examples(files_val)
    print("Number of validation examples: ", nexs_val)
    
    # Create data generator
    train_gen = DataGenerator(files,nexs,batch_size=batch_size)
    val_gen = DataGenerator(files_val,nexs_val,batch_size=batch_size)


    linear_feature_columns = train_gen.linear_feature_columns
    dnn_feature_columns = train_gen.dnn_feature_columns
   

    # 4.Define Model,train,predict and evaluate
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
    optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999,decay=0.0)
    model.compile(optimizer, "binary_crossentropy",
                  metrics=['binary_crossentropy', auroc], )

    pbar = ProgbarLogger(count_mode='steps', stateful_metrics=None)

    weights_file = "model-5-lr0p001.h5"
    model_checkpoint= ModelCheckpoint(weights_file, monitor="val_binary_crossentropy", save_best_only=True,
                                                  save_weights_only=True, verbose=1)

    history = model.fit_generator(train_gen, epochs=10, verbose=1, steps_per_epoch=nexs/batch_size,
                                  validation_data = val_gen, validation_steps=nexs/batch_size,
                                  callbacks = [model_checkpoint])
    #pred_ans = model.predict(test_model_input, batch_size=256)
    #print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    #print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainPath",default="sample_data/")
    parser.add_argument("--valPath",default="sample_data/")
    parser.add_argument("--batch_size",default=256,type=int)

    args = vars(parser.parse_args())
    dataPath = args["trainPath"]
    dataPath_val = args["valPath"]
    batch_size = args["batch_size"]

    print(dataPath)
    main(dataPath,dataPath_val,batch_size)
