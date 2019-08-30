from deepctr.dist_utils.download_utils import download_partition_s3


download_partition_s3(0, 1, "data", "s2019i-yunkzhou", "criteo_processed-competition", nthreads=20, nval=200,ntest=1000)
