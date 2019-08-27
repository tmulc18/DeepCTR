from deepctr.dist_utils.download_utils import download_partition_s3


download_partition_s3(0, 10, "data", "s2019i-yunkzhou", "criteo_processed-competition", nthreads=1, nval=4)
