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
import scipy.stats

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
    from pyspark.mllib.stat import Statistics
    from pyspark.mllib.evaluation import MulticlassMetrics
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

def return_class(posdata_pca, negdata_pca, numdim, pos_prior, test):
    prod_pos = 1
    prod_neg = 1
    p = posdata_pca.map(lambda x: x[0]).map(lambda x: x.toArray())
    n = negdata_pca.map(lambda x: x[0]).map(lambda x: x.toArray())
    pos_rdd = sc.parallelize(p.collect())
    neg_rdd = sc.parallelize(n.collect())
    summary_pos = Statistics.colStats(pos_rdd)
    summary_neg = Statistics.colStats(neg_rdd)

    for i in range(numdim):
        prod_pos *= scipy.stats.norm(summary_pos.mean()[i], math.sqrt(summary_pos.variance()[i])).pdf(test[i])
        prod_neg *= scipy.stats.norm(summary_neg.mean()[i], math.sqrt(summary_neg.variance()[i])).pdf(test[i])
    # print prod_pos
    # print prod_neg
    # code.interact(local=locals())
    pos_prob = prod_pos * pos_prior
    neg_prob = prod_neg * (1 - pos_prior)
    # print pos_prob
    # print neg_prob
    if(pos_prob > neg_prob):
        return True
    else:
        return False

def Gaussian_nb(after_pca, tags, target_tag, test, numdim):
    pos_indices = []
    zippedtags = tags.zipWithIndex()
    pos_tags = zippedtags.filter(lambda x: target_tag in x[0])
    l = pos_tags.collect()
    for i in range(len(l)):
        pos_indices.append(l[i][1])
    #Dividing data into positive and negative
    zip_pca = after_pca.zipWithIndex()

    #pos_indices is a list of indices for which the given tag is present i.e. positive
    posdata_pca = zip_pca.filter(lambda x: x[1] in pos_indices)
    negdata_pca = zip_pca.filter(lambda x: x[1] not in pos_indices)
    #Calculating prior probability
    pos_prior = float(len(pos_indices))/float(zip_pca.count())
    return return_class(posdata_pca, negdata_pca, numdim, pos_prior, test)

def return_top_tags(tags, percent):
    #Flatten the tags E.g. -> [[java, sql, android], [swift, iphone]] will be [java, sql, android, swift, iphone]
    flattags = tags.flatMap(lambda x: x)
    #Generate unique tags
    uniquetags = flattags.distinct()
    #Generate dict of (tag, count)
    tagcountdict = flattags.countByValue()
    #Store (tag, count) in RDD
    tagcountRdd = uniquetags.map(lambda x: (x, tagcountdict[x]))
    #Select top x percent of tags
    threshold = int(math.floor(0.01 * percent * tagcountRdd.count()))
    toppercenttags = tagcountRdd.takeOrdered(threshold, key = lambda x: -x[1])
    toptagsrdd = sc.parallelize(toppercenttags).map(lambda x: str(x[0]))
    return toptagsrdd

if __name__ == '__main__':
    conf = SparkConf()
    conf.set("spark.executor.memory", "2g")
    conf.set("spark.driver.memory", "2g")
    conf.set("spark.driver.maxResultSize", "2g")
    sc = SparkContext(conf=conf)

    ## ****  you can jump this part. directly load data in the next part ****** ##
    # tfidf = create_tfidf(sc)
    # reduced = reduce_tfidf(tfidf, 1000)
    # # comment next line out, if you want to save. (note: change the path accordingly)
    # # reduced.saveAsPickleFile('./data/10k_reducedRDD')
    #
    # before_pca = reduced.map(lambda x: Vectors.dense(x))
    # before_pca.cache()
    #
    # start = time.time()
    # print "start PCA"
    # model = PCA(10).fit(before_pca)
    # end = time.time()
    # print 'time for PCA: ' + str(end - start)
    #
    # # haven't found a way to save PCAModel, so you need to train by yourself if you need it.
    # processed = model.transform(reduced)
    # # comment next line out, if you want to save. (note: change the path accordingly)
    # # processed.saveAsPickleFile('./data/10k_processedRDD')
    # print 'total posts: ' + str(processed.count())


    ## ****  you can directly load data for testing, just comment out the following lines, and change the path accordingly****** ##
    reduced = sc.pickleFile('./data/10k_reducedRDD')
    processed = sc.pickleFile('./data/10k_processedRDD')
    docs = sc.textFile(FILE0, 4).map(split_docs)
    #Fetch tags for each post
    tags = docs.map(lambda doc: doc[1].split())
    #Get top 5 percent of tags
    toptagsrdd = return_top_tags(tags, 5)
    #Get indices of all posts where tag is 'True'
    test_pca = processed.zipWithIndex()
    pos_indices = []
    zippedtags = tags.zipWithIndex()
    pos_tags = zippedtags.filter(lambda x: 'c#' in x[0])
    l = pos_tags.collect()
    for i in range(len(l)):
        pos_indices.append(l[i][1])
    predAndLabels = []
    for test_index in range(test_pca.count()):
        actual = False
        if(test_index in pos_indices):
            actual = True
        test = test_pca.filter(lambda x: x[1] == test_index)
        test = test.map(lambda x: x[0]).map(lambda x: x.toArray()).collect()[0]
        predict = Gaussian_nb(processed, tags, 'c#', test, 10)
        predAndLabels.append((predict, actual))
    predAndLabelsRDD = sc.parallelize(predAndLabels)
    metrics = MulticlassMetrics(predAndLabelsRDD)
    print metrics.confusionMatrix().toArray()
    #import code; code.interact(local=locals())
