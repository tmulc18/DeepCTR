# Criteo Data

## Download
Download the classic competition data [here](http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/).
Download the 1TB data [here](https://labs.criteo.com/2013/12/download-terabyte-click-logs-2/).

## Processing
### Competition data
1. Run [txt2csv.py](./txt2csv.py) to convert the both the text files (train and test) into a CSV files.
2. Split the train file into managable chunks that will fit in memory.
3. Process the data using the spark script. 

### 1TB data
1. Unzip each file.
2. (Optional) store the files in a new location.
3. Process the data using the spark script.

### Data splits
#### Competition data
We use the entire training data as the training dataset, and the test data as the final test dataset.  Note that within the training data, we use 20% as a holdout validation dataset.

#### 1TB data
We use days zero through eighteen as the train dataset, day nineteen for the validation dataset, and days twenty and twentyone as the test dataset.

### Misc.
 While we cannot provide the processed data, we do provide the useful statistics that can be used to reproduce our work (of course, you should also be able to reproduce our work if you follow the above steps verbatim, but we make the statstics avaialble for those who don't have access to large computing clusters).  These include the number of unique values for each categorical, the categorical-to-index mapping for each categorical, and the minimum-maximum value pairs for each dense feature.  For the *competition data* these are computed using the training file.  For the *1TB data*, these are computed over days zero through nineteen.
 

