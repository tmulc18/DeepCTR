import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ProgbarLogger
#from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr.models import DeepFM
#from deepctr.inputs import  SparseFeat, DenseFeat,get_fixlen_feature_names
from deepctr.dist_utils.criteo_gen import DataGenerator, get_total_examples
import argparse
import glob

def main(dataPath,batch_size):
    files = glob.glob(dataPath+"*.csv")[0:2]
    
    # Count number of examples in training data
    nexs = get_total_examples(files)
    print("Number of examples: ", nexs)
    
    # Create data generator
    train_gen = DataGenerator(files,nexs)


    linear_feature_columns = train_gen.linear_feature_columns
    dnn_feature_columns = train_gen.dnn_feature_columns
   

    # 4.Define Model,train,predict and evaluate
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
    model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy'], )

    pbar = ProgbarLogger(count_mode='steps', stateful_metrics=None)

    history = model.fit_generator(train_gen, epochs=10, verbose=2, steps_per_epoch=nexs/256 )
    #pred_ans = model.predict(test_model_input, batch_size=256)
    #print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    #print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trainPath")
    parser.add_argument("-batch_size",default=256)

    args = vars(parser.parse_args())
    dataPath = args["trainPath"]
    batch_size = args["batch_size"]


    print(dataPath)
    main(dataPath,batch_size)
