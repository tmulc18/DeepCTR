# Criteo Data

## Download
Download the classic competition data [here](http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/).
Download the 1TB data [here](https://labs.criteo.com/2013/12/download-terabyte-click-logs-2/).

## Processing
### Competition data
1. Run [text2csv.py](./txt2csv.py) to convert the text file into a CSV file.
2. Split the file into managable chunks that will fit in memory.
3. Process the data using the spark script.

### 1TB data
1. Unzip each file.
2. (Optional) store the files in a new location.
3. Process the data using the spark script.
