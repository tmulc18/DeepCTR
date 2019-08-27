import os
import random
import boto3
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

def download_partition_s3(worker_id, num_workers, save_dir, bucket, prefix, nthreads, nval=200, ntest=None):
    """Downloads a partition of data from s3.

    If the buckets has L files, there are V files for validation,
    and there are W workers, this function downloads (L-V)/W 
    files for trainng and V files for validation.
    """
    
    resource = boto3.resource('s3') #high-level object-oriented API
    my_bucket = resource.Bucket(bucket) #subsitute this for your s3 bucket name. 

    # get file names
    files = list(my_bucket.objects.filter(Prefix=prefix)) #get file names

    # extract and filter
    fkeys = [f.key for f in files]
    fkeys = [fk for fk in fkeys if fk.split('.')[-1] == 'csv']# filter non-relevant files

    # partition
    fkeys.sort() # must sort so multiple workers can take subsets

    # take out the test dataset
    if ntest is not None:
        fkeys_test = fkeys[len(fkeys)-ntest:]
        if not os.path.exists(save_dir+"_test"): os.mkdir(save_dir+"_test")
        fkeys = fkeys[:len(fkeys)-ntest]

    # randomize, but set seed so all workers randomize the same
    random.seed(1)
    random.shuffle(fkeys)

    # all workers get the same validation set
    fkeys_val = fkeys[:nval]

    # take out the validation data from the training keys
    fkeys = fkeys[nval:]

    # partition training data
    fkeys = fkeys[worker_id::num_workers]

    # create location train locally
    if not os.path.exists(save_dir): os.mkdir(save_dir) 
    if not os.path.exists(save_dir+"_val"): os.mkdir(save_dir+"_val") 

    # define download function
    def download_func(fkey):
        print("Downloading training, ",fkey)
        my_bucket.download_file(fkey,save_dir+'/'+fkey.split('/')[-1])

    # define download function for validation
    def download_func_val(fkey):
        my_bucket.download_file(fkey,save_dir+"_val"+'/'+fkey.split('/')[-1])

    # define download function for test
    if ntest is not None:
        def download_func_test(fkey):
            my_bucket.download_file(fkey,save_dir+"_test"+'/'+fkey.split('/')[-1])

    # download training data
    p = ThreadPool(nthreads)
    total_files = len(fkeys)
    with tqdm(total=total_files) as pbar:
        for i,_ in tqdm(enumerate(p.imap_unordered(download_func,fkeys))):
            pbar.update()

    # download validation data
    p = ThreadPool(nthreads)
    total_files = len(fkeys_val)
    with tqdm(total=total_files) as pbar:
        for i,_ in tqdm(enumerate(p.imap_unordered(download_func_val,fkeys_val))):
            pbar.update()

    # download test data
    if ntest is not None:
        p = ThreadPool(nthreads)
        total_files = len(fkeys_test)
        with tqdm(total=total_files) as pbar:
            for i,_ in tqdm(enumerate(p.imap_unordered(download_func_test,fkeys_test))):
                pbar.update()


