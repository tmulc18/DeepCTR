import os
import pandas as pd

nrows_per_file = 1000000
files2split = ['/mnt/train.csv']
save_loc = '/mnt/train_files/'

if not os.path.exists(save_loc):
    os.makedirs(save_loc)

# get number of lines in original file
file = files2split[0]
with open(file) as f:
    for i, l in enumerate(f):
        pass
total_lines = i
print("Total lines: ",total_lines)

lines_left = total_lines
chunk_index = 0
fname = file.split('/')[-1].split('.csv')[0]
while lines_left > 0:
    df = pd.read_csv(file,nrows=nrows_per_file,skiprows=range(1,chunk_index*nrows_per_file))
    df.to_csv(save_loc+fname+'-{ci}.csv'.format(ci=chunk_index),index=False)
    chunk_index += 1
    lines_left -= nrows_per_file
