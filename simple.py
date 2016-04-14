# coding=utf-8
import os
import sys
import time
import pickle
import shutil
import operator
import numpy as np
from operator import itemgetter
from collections import Counter
from nltk.corpus import stopwords
from sklearn.externals import joblib
from nltk.stem.lancaster import LancasterStemmer
from helpers import *

SPLITTER = '>>>'
st = LancasterStemmer()
stopwords = stopwords.words('english')

baseDir = os.path.join('data')
FILE0 = os.path.join(baseDir, '1000posts.txt')

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
    return tfidf, tags


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


def dist(vec_id, tests):
    result = []
    vec1 = np.asarray(vec_id[0])
    id1 = vec_id[1]
    for test_id in tests:
        vec2 = np.asarray(test_id[0])
        id2 = test_id[1]
        result.append([id1, id2, np.linalg.norm(vec2-vec1)])
    result = sorted(result, key=itemgetter(2))
    ranked_tags = [x[1] for x in result]
    return result, ranked_tags

# def dist(row, test):
#     return np.linalg.norm(row-test)


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


def get_tag(rankedIDs, tags_IDs, top = 20, KeepAllTag = True):
    candidates = [tags_IDs[c][0] for c in rankedIDs[:top]]
    if KeepAllTag:
        allTags = []
        for tags in candidates:
            allTags.extend(tags)
        allTagCount = Counter(allTags)
        most_likely = allTagCount.most_common(3)
    else:
        keepOneTag = [tag[0] for tag in candidates ]
        oneTagCount = Counter(keepOneTag)
        most_likely = oneTagCount.most_common(3)

    max = 0
    predicts = []
    for tag_count in  most_likely:
        if tag_count[1] >= max:
            max = tag_count[1]
            predicts.append(tag_count[0])
    return predicts

#get_tag(dit_matrix.first()[1], tag_id_broadcast.value, 50)


def judge(true, predict):
    common = set(true).intersection(predict)
    if common:
        return 1
    return 0

def convert2tag_id_pairs(id, tags):
    pairs = []
    for tag in tags:
        pairs.append([tag, id])
    return pairs


def get_matrixs(predict, actual, all):
    ConfusionMatrix, PerfomanceMatrix = {}, {}
    count = 0
    for tag, P_count_ids in predict.iteritems():
        A_count_ids = actual.get(tag)
        if not A_count_ids:
            continue
        P_count, P_ids = P_count_ids[0], set(P_count_ids[1])
        A_count, A_ids = A_count_ids[0], set(A_count_ids[1])
        TP_set = P_ids & A_ids
        TN_set = all - (P_ids | A_ids)
        FP_set = P_ids - TP_set
        FN_set = A_ids - TP_set
        TP, TN, FP, FN = len(TP_set), len(TN_set), len(FP_set), len(FN_set)
        ConfusionMatrix[tag] = {'TP': TP,
                                'TN' : TN,
                                'FP': FP,
                                'FN': FN}
        p = float(TP)/float(TP+FP) if 0 != (TP + FP) else 0
        r = float(TP)/float(TP+FN) if 0 != (TP + FN) else 0
        a = float(TP+TN)/float(TP+TN+FP+FN) if 0 != (TP+TN+FP+FN) else 0
        f = 2.0*p*r/float(p+r) if 0 != (p+r) else 0
        PerfomanceMatrix[tag] = {'precision': p,
                                 'recall': r,
                                 'accuracy': a,
                                 'f-score': f}
        print  "tag:\t" + str(tag) + "\taccuracy:\t" + str(a) + "\tprecision:\t" + str(p) + "\trecall:\t" + str(r) + "\tf-score:\t" + str(f)
    return ConfusionMatrix, PerfomanceMatrix


def save_obj(obj, name ):
    with open('./data/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('./data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def KNN(vectors, tags, K_percent=0.025, KeepAllTag = True):
    vec_id = vectors.zipWithIndex()
    tag_id = tags.zipWithIndex()
    top_tags = return_top_tags(tags, 1)


    train_data, test_data = tag_id.randomSplit([0.6, 0.4], seed=0)
    train_id = train_data.map(lambda x: x[1]).collect()
    test_id = test_data.map(lambda x: x[1]).collect()
    train = vec_id.filter(lambda x : x[1] in train_id)
    test = vec_id.filter(lambda x : x[1] in test_id)
    train_broadcast = sc.broadcast(train.collect())
    # each row is [ [testID, trainID, distance], rankedID ]
    dit_matrix = test.map(lambda x: dist(x, train_broadcast.value))
    # ranked_matrix = dit_matrix.map(lambda x: )
    tag_id_broadcast = sc.broadcast(tag_id.collect())
    selection = int(K_percent * tag_id.count())
    predict = dit_matrix.map(lambda x: (x[0][0][0], get_tag(x[1], tag_id_broadcast.value, top=selection, KeepAllTag = KeepAllTag)))
    predict_TagIdPairs = predict.map(lambda x: convert2tag_id_pairs(x[0],x[1]))
    print "KNN Predicting tags..."
    predict_tag_IDs = predict_TagIdPairs.flatMap(lambda x: x)\
                                        .map(lambda x: (x[0], [x[1]]))\
                                        .reduceByKey(lambda x,y: x+y)
    print "getting top_ranked predict tags..."
    top_predict_tag_IDs = predict_tag_IDs.map(lambda x: (len(x[1]),(x[0], x[1])))\
                                        .sortByKey(ascending=False)\
                                        .map(lambda x: [x[1][0], (x[0],x[1][1])])\
                                        .take(10)
    top_predict_tag_IDs = {k:v for k,v in top_predict_tag_IDs}

    actual = test_data.map(lambda x: (x[1], x[0]))
    actual_TagIdPairs = actual.map(lambda x: convert2tag_id_pairs(x[0],x[1]))
    print "getting top_ranked actual tags..."
    actual_tag_IDs = actual_TagIdPairs.flatMap(lambda x: x)\
                                        .map(lambda x: (x[0], [x[1]]))\
                                        .reduceByKey(lambda x,y: x+y)
    top_actual_tag_IDs = actual_tag_IDs.map(lambda x: (len(x[1]),(x[0], x[1])))\
                                        .sortByKey(ascending=False)\
                                        .map(lambda x: [x[1][0], (x[0],x[1][1])])\
                                        .take(100)
    top_actual_tag_IDs = {k:v for k,v in top_actual_tag_IDs}
    print "evaluating results..."
    confution_matrix, perfomanceMatrix = get_matrixs(top_predict_tag_IDs, top_actual_tag_IDs, set(test_id))
    save_obj(confution_matrix, '1k_confution_matrix')
    save_obj(perfomanceMatrix, '1k_perfomanceMatrix')
    print "\ntest size:\t" + str(len(test_id)) + "\tparameter:\t" + str(K_percent) + '\t' + str(KeepAllTag)
    print "done!"
    # evaluate = actual.join(predict).map(lambda x: (x[0], judge(x[1][0], x[1][1])))
    # correct = evaluate.map(lambda x: x[1]).reduce(lambda x,y: x+y)
    # test_size = test_data.count()
    # overall_accuracy = float(correct)/test_size
    # print "parameter:\t" + str(K_percent) + '\t' + str(KeepAllTag) + ",\ttest size:\t" + str(test_size) +"\tcorrect prediction:\t" + str(correct) + "\toverall accuracy:\t" + str(overall_accuracy)



if __name__ == '__main__':
    conf = SparkConf()
    conf.set("spark.executor.memory", "16g")
    conf.set("spark.driver.memory", "16g")
    conf.set("spark.driver.maxResultSize", "16g")
    sc = SparkContext(conf=conf)

    ## ****  you can jump this part. directly load data in the next part ****** ##
    print "calculating tfidf ..."
    tfidf, tags = create_tfidf(sc)
    #reduced = reduce_tfidf(tfidf, 1000)
    save_file = './data/1k_reducedRDD'
    # if os.path.exists(save_file):
    #     shutil.rmtree(save_file, ignore_errors=True)
    # reduced.saveAsPickleFile(save_file)
    reduced = sc.pickleFile(save_file)
    #processed = sc.pickleFile('./data/10k_processedRDD')

    #tune parameters below
    K_percent=0.005
    KeepAllTag = False
    # you can run KNN to get the results by yourself, 1k posts in 60s, 10k posts in 20 mins.
    KNN(reduced,tags, K_percent, KeepAllTag)
    # or... you can load the results for KNN directly, there is 1k and 10k version in ./data
    confution_matrix = load_obj('1k_confution_matrix')
    perfomanceMatrix = load_obj('1k_perfomanceMatrix')
    # need some one use this data to make some figures for showing our results.


    # comment next line out, if you want to save. (note: change the path accordingly)
    # reduced.saveAsPickleFile('./data/1k_reducedRDD')
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
    # processed.saveAsPickleFile('./data/1k_processedRDD')
    # print 'total posts: ' + str(processed.count())


    ## ****  you can directly load data for testing, just comment out the following lines, and change the path accordingly****** ##
    # reduced = sc.pickleFile('./data/10k_reduced_RDD')
    # processed = sc.pickleFile('./data/10k_processedRDD')


