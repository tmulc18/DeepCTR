# Criteo Experiments
We noticed that the original code was unsuitable for the training models on Criteo datasets.  The main issue was that all the data was required to fit in memory, and even with 60 GiB of RAM, we were unable to fit the (smaller) Kaggle Criteo dataset into memory;  the (larger) 1TB Criteo dataset certainly would not fit into memory.  Thus, the main design change we made involved using a data generator to load smaller files into memory.  We also did all datatransformations before hand and saved the relevant statistics for training.  We provide an alternative training script for criteo training that can be used for both the Kaggle data and 1TB data: [`run_classification_criteo_dist.py`](run_classification_criteo_dist.py). All additional code is the [`dist_utils`](/deepctr/dist_utils/) folder.

# Criteo Data

The Kaggle competition data has X total records.  We use 40,340,943 records for training, 916,835 records for validation, and 4,582,840 records for the holdout test data.  Because the data is sorted in chronological order, the test data is taken from the end of the file.

The 1TB data has X total records.  We use X records for training, X records for validation, and X records for the holdout test data.  Days 0 through 18 are used for training, day 19 is used for validation, and days 20 and 21 are used for the test data.

## Download
Download the classic competition data [here](http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/).  There should be two text files: `text.txt` and `train.txt`.

Download the 1TB data [here](https://labs.criteo.com/2013/12/download-terabyte-click-logs-2/).  There are 21 days total, and each day must be downloaded seperately.  Each day is about 14 GiB compressed.

## Processing
### Competition data
1. Run [txt2csv.py](./txt2csv.py) to convert the both the text files (train and test) into a CSV files.
2. Run [split_file.py](/deepctr/dist_utils/split_file.py) to split the train file into managable chunks that will fit in memory.
3. (Optional) store the files in a new location.
4. Process the data using the spark script.

### 1TB data
1. Unzip each file.  *Note: uncompressing each 14 GiB file results in a 48 GiB file.*
2. (Optional) store the files in a new location.
3. Process the data using the spark script.

### Data splits
#### Competition data
We use the entire training data as the training dataset, 2% of the data as a validation dataset, and the last 10% of data as the holdout test dataset.

#### 1TB data
We use days zero through eighteen as the train dataset, day nineteen for the validation dataset, and days twenty and twentyone as the test dataset.

### Misc.
 While we cannot provide the processed data, we do provide the useful statistics that can be used to reproduce our work (of course, you should also be able to reproduce our work if you follow the above steps verbatim, but we make the statstics avaialble for those who don't have access to large computing clusters).  These include the number of unique values for each categorical, the categorical-to-index mapping for each categorical, and the minimum-maximum value pairs for each dense feature.  For the *competition data* these are computed using the training file.  For the *1TB data*, these are computed over days zero through nineteen.
 

