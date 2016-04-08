# coding=utf-8
import os
import sys
import time
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from helpers import *

SPLITTER = '>>>'
st = LancasterStemmer()
stopwords = stopwords.words('english')

baseDir = os.path.join('data')
FILE0 = os.path.join(baseDir, '1000test.txt')

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
    from pyspark.mllib.feature import PCA
    from pyspark.mllib.linalg import Vectors
    from pyspark.sql import SQLContext
    print ("Successfully imported Spark Modules")

except ImportError as e:
    print ("Can not import Spark Modules", e)
    sys.exit(1)


def parseData(filenema):
    """ Parse a data file.
    :param filenema:
    :return:
        a RDD of parsed lines
    """
    return (sc.textFile(FILE0,4,0 ))


def split_docs(line):
    line = line.split(SPLITTER)
    return line


def unit_test(rdd):
    print 'count = ' + str(rdd.count())
    for i in rdd.take(3):
        print i


def preProcess(list):
    """ remove stopwords and stemming
    Args:
        string (str): list of words
    Returns:
        list: preprocessed words without stopwords
    """
    return [token for token in list if token not in stopwords]


def countTokens(vendorRDD):
    """ Count and return the number of tokens
    Args:
        vendorRDD (RDD of (recordId, tokenizedValue)): Pair tuple of record ID to tokenized output
    Returns:
        count: count of all tokens
    """
    return vendorRDD.map(lambda x: len(x[1])).sum()


def tf(tokens):
    """ Compute TF
    Args:
        tokens (list of str): input list of tokens from tokenize
    Returns:
        dictionary: a dictionary of tokens to its TF values
    """
    tf = {}
    totalCount = len(tokens)
    if totalCount == 0:
        return tf
    for t in tokens:
        if t in tf:
            tf[t] += 1.0
        else:
            tf[t] = 1.0
    return {token: count / totalCount for (token, count) in tf.iteritems() }


def idfs(corpus):
    """ Compute IDF
    Args:
        corpus (RDD): input corpus
    Returns:
        RDD: a RDD of (token, IDF value)
    """
    N = float(corpus.count())
    # Note it is very important here to use set() to make every element unique, then use list() to match the format
    uniqueTokens = corpus.flatMap(lambda x: list(set(x)))
    tokenCountPairTuple = uniqueTokens.map(lambda x: (x,1))
    tokenSumPairTuple = tokenCountPairTuple.reduceByKey(lambda x,y: x+y)
    return (tokenSumPairTuple.map(lambda (x,y): (x,N/y)))


def tfidf(tokens, idfs):
    """ Compute TF-IDF
    Args:
        tokens (list of str): input list of tokens from tokenize
        idfs (dictionary): record to IDF value
    Returns:
        dictionary: a dictionary of records to TF-IDF values
    """
    tfs = tf(tokens)
    # *** rememeber to get the Value of a Key from a Dict, you should use [] NOT (), () is for set or list.*** #
    tfIdfDict = {k: tfs[k]*idfs[k] for k in tokens if k in idfs}
    return tfIdfDict


def get_common(matrix_cell):
    """
    get the common tokens between two posts
    :param matrix_cell: a pair, (ID_tokens0, ID_tokens1)  tokens_ID: (id, [tokens])
    :return:
    """
    ID_tokens0 = matrix_cell[0]
    ID_tokens1 = matrix_cell[1]
    ID0 = ID_tokens0[0]
    ID1 = ID_tokens1[0]
    # # make sure ID1 >= ID0, so that for a N*N matrix, we only need to calculate (1+N)*N/2
    # if ID0 < ID1:
    #     ID0, ID1 = ID1, ID0

    commonTokens = [token for token in ID_tokens0[1] if token in ID_tokens1[1]]
    # if commonTokens is []:
    #     return [[ID0, ID1], []]
    return [[ID0, ID1], commonTokens]

def fastCosineSimilarity(record):
    """ Compute Cosine Similarity using Broadcast variables
    Args:
        record: ((ID, URL), token)
    Returns:
        pair: ((ID, URL), cosine similarity value)
    """
    ID0 = record[0][0]
    ID1 = record[0][1]
    tokens = record[1]
    if tokens == [] or ID0 < ID1:
        value = 0
    else:
        s = sum(tfidf_RDD_broadcast.value[ID0][token] * tfidf_RDD_broadcast.value[ID1][token]
                for token in tokens
               )
        value = s/(tfidf_norm_broadcast.value[ID0] * tfidf_norm_broadcast.value[ID1])
    key = (ID0, ID1)
    return (key, value)

def uniform_tfidf_rep(words, total_copurs):
    """ transform the tfidf dict of a post to an uniform tfidf representation for all the posts.
    :param  words: dict {word: tfidf} for each word in one post
            total_copurs: the unique words for all the posts.
    :return: a uniform representation of a list of tfidf values for this post.
            (dimension is the total unique words counts for all the posts)
    """
    uniform_rep = [0]*len(total_copurs)
    for word, tfidf in words.iteritems():
        uniform_id = total_copurs.index(word)
        uniform_rep[uniform_id] = tfidf
    return uniform_rep


# Creates and returns a tfidf rdd from FILE0
def create_tfidf(sc):
    start = time.time()


    docs = sc.textFile(FILE0, 4).map(split_docs)
    tags = docs.map(lambda doc: doc[1].split())
    words = docs.map(lambda doc: doc[0].split())
    words = words.map(preProcess).cache()
    unique_words = words.flatMap(lambda x: x).distinct().collect()
    unique_words_broadcast = sc.broadcast(unique_words)
    # use zipWithIndex() to append unique ID for each document
    ID_tokens = words.zipWithIndex().map(swapOder)
    token_ID_pairs = ID_tokens.flatMap(invert).cache()

    # Note:
    # use collectAsMap() to get a dict, instead of using collect to get a list.
    # use broadcast() to distribute the idf dict to all workers to reduce transfer time.
    words_idf = idfs(words)
    idfWeight = words_idf.collectAsMap()
    idf_weight_broadcast = sc.broadcast(idfWeight)

    # get the whole tfidf RDD and norm(tfidf)
    tfidf_RDD = ID_tokens.map(lambda (ID, tokens): (ID, tfidf(tokens, idf_weight_broadcast.value))).cache()
    tfidf_RDD_broadcast =  sc.broadcast(tfidf_RDD.collectAsMap())
    tfidf_matrix = tfidf_RDD.map(lambda row: uniform_tfidf_rep(row[1], unique_words_broadcast.value))

    return tfidf_matrix

    # tfidf_norm = tfidf_RDD.map(lambda (k, v): (k, norm(v)))
    # tfidf_norm_broadcast = sc.broadcast(tfidf_norm.collectAsMap())
    #
    #
    # '''
    # The A.cartesian(B) will be an RDD of the form:
    # [(A ID1, A String1), (A ID2, A String2), ...]  and  [(B ID1, B String1), (B ID2, B String2), ...]
    # to:
    # [ ((A ID1, A String1), (B ID1, B String1)), ((A ID1, A String1), (B ID2, B String2)), ((A URL2, A String2), (B ID1, B String1)), ... ]¶
    # '''
    # cross_RDD = ID_tokens.cartesian(ID_tokens).cache()
    # # commonTokens:  [[id1, id2], [tokens]]
    # commonTokens = cross_RDD.map(get_common)
    # similarities_RDD = commonTokens.map(fastCosineSimilarity).cache()
    #
    # end = time.time()
    # print 'total prepare: '+ str(end - start)
    # print similarities_RDD.count()
    # c_time = time.time()
    # print 'count time: ' + str(c_time - end)
    # similarities_RDD.collect()
    # c2_time = time.time()
    # print 'count time: ' + str(c2_time - c_time)
    # print 'Successfully Calculated the similarities between all the posts'


if __name__ == '__main__':
    sc = SparkContext('local')
    tfidf_matrix = create_tfidf(sc)
    tfidf_dVector_matrix = tfidf_matrix.map(lambda row: Vectors.dense(row))
    reduc = PCA(3).fit(tfidf_dVector_matrix)
    after_pca = reduc.transform(tfidf_dVector_matrix)


