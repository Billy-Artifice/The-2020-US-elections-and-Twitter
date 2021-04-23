import pyspark
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import lower, col, udf
import pandas as pd
from pyspark.sql.types import *
import string
import re


def define_structure(string, format_type):
    try:
        incorrect_type = equivalent_type(format_type)
    except:
        incorrect_type = StringType()
    return StructField(string, incorrect_type)


def equivalent_type(format_type):
    if format_type == 'datetime64[ns]':
        return DateType()
    elif format_type == 'float64':
        return FloatType()
    elif format_type == 'int32':
        return IntegerType()
    elif format_type == 'int64':
        return LongType()
    else:
        return StringType()


# Given pandas dataframe, it will return a spark's dataframe.
def pandas_to_spark(pandas_df, spark_sql_context):
    columns = list(pandas_df.columns)
    types = list(pandas_df.dtypes)
    struct_list = []
    for column, typo in zip(columns, types):
        struct_list.append(define_structure(column, typo))
    p_schema = StructType(struct_list)
    return spark_sql_context.createDataFrame(pandas_df, p_schema)


def process_df(filename, spark_context, save_name,candiate_name):
    df = pd.read_csv(filename, lineterminator='\n', low_memory=False,nrows=600000)
    df.drop(df.columns.difference(
        ['created_at', 'tweet_id', 'tweet', 'likes', 'source','user_id', 'lat', 'long', 'country', 'state']), 1,
            inplace=True)
    df.loc[:, 'candidate'] = candiate_name
    df['country'].replace({'United States of America': 'United States'}, inplace=True)
    sparkDf = pandas_to_spark(df, spark_context)
    removePunctuation = udf(lambda x: re.sub(r'[^\w\s]', '', x))
    sparkDf = sparkDf.unpersist().withColumn('tweet', lower(col('tweet')))
    sparkDf.cache()
    sparkDf = sparkDf.unpersist().withColumn('tweet', removePunctuation('tweet'))

    sparkDf.toPandas().to_csv('./' + save_name + '.csv')


if __name__ == '__main__':
    s_config = SparkConf()
    sc = SparkContext.getOrCreate(conf=s_config)
    spark = SparkSession.builder.master('local').appName('project').getOrCreate()

    # df_biden = process_df('./hashtag_joebiden.csv', spark, 'joebiden_clean','biden')
    df_trump = process_df('./hashtag_donaldtrump.csv', spark, 'donaldtrump_clean','trump')
