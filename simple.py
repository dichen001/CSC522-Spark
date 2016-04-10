# coding=utf-8
import os
import sys
import time
import pickle
import operator
import numpy as np
from nltk.corpus import stopwords
from sklearn.externals import joblib
from nltk.stem.lancaster import LancasterStemmer
from helpers import *

SPLITTER = '>>>'
st = LancasterStemmer()
stopwords = stopwords.words('english')

baseDir = os.path.join('data')
FILE0 = os.path.join(baseDir, '10k_posts.txt')

"""
Set up the Spark and PySpark Environment for PyCharm
Notes:  Adapted from http://renien.com/blog/accessing-pyspark-pycharm/
"""
# Path for spark source folder
os.environ['SPARK_HOME'] = "/Users/Jack/Projects/spark-1.6.1"

# Append pyspark  to Python Path
sys.path.append("/Users/Jack/Projects/spark-1.6.1/python/")
sys.path.append("/Users/Jack/Projects/spark-1.6.1/python/lib/py4j-0.9-src.zip")

try:
    from pyspark import SparkContext
    from pyspark import SparkConf
    from pyspark.sql import SQLContext
    from pyspark.mllib.linalg import Vectors, SparseVector, DenseVector
    from pyspark.mllib.feature import PCA, IDF, HashingTF
    print ("Successfully imported Spark Modules")

except ImportError as e:
    print ("Can not import Spark Modules", e)
    sys.exit(1)



def split_docs(line):
    line = line.split(SPLITTER)
    return line


def preProcess(list):
    """ remove stopwords and stemming
    Args:
        string (str): list of words
    Returns:
        list: preprocessed words without stopwords
    """
    return [token for token in list if token not in stopwords]

def vec_add(v1, v2):
    """Add two sparse vectors
    >>> v1 = Vectors.sparse(3, {0: 1.0, 2: 1.0})
    >>> v2 = Vectors.sparse(3, {1: 1.0})
    >>> add(v1, v2)
    SparseVector(3, {0: 1.0, 1: 1.0, 2: 1.0})
    """
    assert isinstance(v1, SparseVector) and isinstance(v2, SparseVector)
    assert v1.size == v2.size
    # Compute union of indices
    indices = set(v1.indices).union(set(v2.indices))
    # Not particularly efficient but we are limited by SPARK-10973
    # Create index: value dicts
    v1d = dict(zip(v1.indices, v1.values))
    v2d = dict(zip(v2.indices, v2.values))
    zero = np.float64(0)
    # Create dictionary index: (v1[index] + v2[index])
    values =  {i: v1d.get(i, zero) + v2d.get(i, zero)
       for i in indices
       if v1d.get(i, zero) + v2d.get(i, zero) != zero}

    return Vectors.sparse(v1.size, values)


# Creates and returns a tfidf rdd from FILE0
def create_tfidf(sc):
    # start = time.time()
    docs = sc.textFile(FILE0, 4).map(split_docs)
    tags = docs.map(lambda doc: doc[1].split())
    tag = tags.map(lambda tags: tags[0])
    words = docs.map(lambda doc: doc[0].split())
    words = words.map(preProcess).cache()

    # id_tag = tag.zipWithIndex().map(swapOder)

    hashingTF = HashingTF()
    tf = hashingTF.transform(words)
    tf.cache()
    idf = IDF(minDocFreq=2).fit(tf)
    tfidf = idf.transform(tf)
    #tfidf = tfidf.collect()
    return tfidf


def reduce_dimention(vec, keys):
    indices = set(keys)
    v_dict = dict(zip(vec.indices, vec.values))
    reduced = [0]*len(keys)
    zero = np.float64(0)
    for id in indices:
        id_of_reduced_words = keys.index(id)
        reduced[id_of_reduced_words] = v_dict.get(id,zero)
    return reduced


def to_sparse(v):
  values = {i: e for i,e in enumerate(v) if e != 0}
  return Vectors.sparse(v.size, values)


def reduce_tfidf(tfidf, dimension = 1000):
    tfidf_array = tfidf.map(lambda x: x.toArray())
    # vector_size = len(tfidf.first().toArray())
    start = time.time()
    print "reducing to get the whole corpus"
    all = tfidf_array.reduce(lambda x,y: x+y)
    end = time.time()
    print "finished in : " + str(round(end-start,3)) + "s"
    tmp = all.tolist()
    reduc = {tmp.index(i): i for i in tmp if i!=0}
    max = len(reduc)
    print "total count for reduced corpus: " + str(max)
    if max < dimension:
        dimension = max

    # all_dict = dict(zip(all.indices, all.values))
    sorted_x = sorted(reduc.items(), key=operator.itemgetter(1))
    reduced_key = [key[0] for key in sorted_x[int(-1*dimension):]]
    tfidf_reduced = tfidf.map(lambda x: reduce_dimention(x,reduced_key))
    return tfidf_reduced


if __name__ == '__main__':
    conf = SparkConf()
    conf.set("spark.executor.memory", "16g")
    conf.set("spark.driver.memory", "16g")
    conf.set("spark.driver.maxResultSize", "16g")
    sc = SparkContext(conf=conf)

    ## ****  you can jump this part. directly load data in the next part ****** ##
    tfidf = create_tfidf(sc)
    reduced = reduce_tfidf(tfidf, 1000)
    # comment next line out, if you want to save. (note: change the path accordingly)
    # reduced.saveAsPickleFile('./data/10k_reducedRDD')

    before_pca = reduced.map(lambda x: Vectors.dense(x))
    before_pca.cache()

    start = time.time()
    print "start PCA"
    model = PCA(10).fit(before_pca)
    end = time.time()
    print 'time for PCA: ' + str(end - start)

    # haven't found a way to save PCAModel, so you need to train by yourself if you need it.
    processed = model.transform(reduced)
    # comment next line out, if you want to save. (note: change the path accordingly)
    # processed.saveAsPickleFile('./data/10k_processedRDD')
    print 'total posts: ' + str(processed.count())


    ## ****  you can directly load data for testing, just comment out the following lines, and change the path accordingly****** ##
    # reduced = sc.pickleFile('./data/10k_reduced_RDD')
    # processed = sc.pickleFile('./data/10k_processedRDD')


