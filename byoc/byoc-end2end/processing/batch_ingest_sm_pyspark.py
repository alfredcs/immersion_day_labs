# Feature engineering and feature ingestion

from pyspark.sql.functions import udf, datediff, to_date, lit
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.sql import SparkSession, DataFrame
from argparse import Namespace, ArgumentParser
from pyspark.ml.linalg import Vector
from pyspark.ml import Pipeline
from datetime import datetime
import argparse
import logging
import boto3
import time
import os


logger = logging.getLogger('__name__')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def transform_row(row) -> list:
    columns = list(row.asDict())
    record = []
    for column in columns:
        feature = {'FeatureName': column, 'ValueAsString': str(row[column])}
        record.append(feature)
    return record


def ingest_to_feature_store(args: argparse.Namespace, rows) -> None:
    feature_group_name = args.feature_group_name
    session = boto3.session.Session()
    featurestore_runtime_client = session.client(service_name='sagemaker-featurestore-runtime')
    rows = list(rows)
    logger.info(f'Ingesting {len(rows)} rows into feature group: {feature_group_name}')
    for _, row in enumerate(rows):
        record = transform_row(row)
        response = featurestore_runtime_client.put_record(FeatureGroupName=feature_group_name, Record=record)
        assert response['ResponseMetadata']['HTTPStatusCode'] == 200


def parse_args() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_processes', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--feature_group_name', type=str)
    parser.add_argument("--s3_uri_prefix", type=str)
    args, _ = parser.parse_known_args()
    return args


def run_spark_job():
    args = parse_args()
    spark_session = SparkSession.builder.appName('PySparkJob').getOrCreate()
    spark_context = spark_session.sparkContext
    total_cores = int(spark_context._conf.get('spark.executor.instances')) * int(spark_context._conf.get('spark.executor.cores'))
    logger.info(f'Total available cores in the Spark cluster = {total_cores}')
    logger.info('Reading input file from S3')
    df = spark_session.read.options(Header=True).csv(args.s3_uri_prefix)
    
    # transform raw features
    # Wo do nothing in our case
    
    logger.info(f'Number of partitions = {df.rdd.getNumPartitions()}')
    # Rule of thumb heuristic - rely on the product of #executors by #executor.cores, and then multiply that by 3 or 4
    df = df.repartition(total_cores * 3)
    logger.info(f'Number of partitions after re-partitioning = {df.rdd.getNumPartitions()}')
    logger.info(f'Feature Store ingestion start: {datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}')
    df.foreachPartition(lambda rows: ingest_to_feature_store(args, rows))
    logger.info(f'Feature Store ingestion complete: {datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}')


if __name__ == '__main__':
    logger.info('BATCH INGESTION - STARTED')
    run_spark_job()
    logger.info('BATCH INGESTION - COMPLETED')