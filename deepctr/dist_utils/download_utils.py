import os
import boto3
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

def download_partition_s3(worker_id, num_workers, save_dir, bucket, prefix, nthreads):
    resource = boto3.resource('s3') #high-level object-oriented API
    my_bucket = resource.Bucket(bucket) #subsitute this for your s3 bucket name. 

    # get file names
    files = list(my_bucket.objects.filter(Prefix=prefix)) #get file names

    # extract and filter
    fkeys = [f.key for f in files]
    fkeys = [fk for fk in fkeys if fk.split('.')[-1] == 'csv']# filter non-relevant files

    # partition
    fkeys.sort() # must sort so multiple workers can take subsets
    fkeys = fkeys[worker_id::num_workers]

    # create location locally
    if not os.path.exists(save_dir): os.mkdir(save_dir) 

    # define download function
    def download_func(fkey):
        my_bucket.download_file(fkey,save_dir+'/'+fkey.split('/')[-1])

    # download
    p = ThreadPool(nthreads)
    total_files = len(fkeys)
    with tqdm(total=total_files) as pbar:
        for i,_ in tqdm(enumerate(p.imap_unordered(download_func,fkeys))):
            pbar.update()

