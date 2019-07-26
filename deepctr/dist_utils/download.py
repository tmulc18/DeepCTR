""" Script for downloading data."""

import argparse
from deepctr.dist_utils.download_utils import download_partition_s3


def main(worker_id, num_workers, save_dir, bucket, prefix, nthreads):
    download_partition_s3(worker_id, num_workers, save_dir, bucket, prefix, nthreads)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('save_dir')
    parser.add_argument('bucket')
    parser.add_argument('prefix')
    parser.add_argument('-worker_id',default=0,type=int)
    parser.add_argument('-num_workers',default=1,type=int)
    parser.add_argument('-nthreads',default=4,type=int)

    param_dict = vars(parser.parse_args())
    save_dir = param_dict['save_dir']
    bucket = param_dict['bucket']
    prefix = param_dict['prefix']

    worker_id = param_dict['worker_id']
    num_workers = param_dict['num_workers']
    nthreads = param_dict['nthreads']

    main(worker_id, num_workers, save_dir, bucket, prefix, nthreads)

