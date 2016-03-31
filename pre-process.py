import os
import sys
''' ------------ Content in the box is adapted from http://renien.com/blog/accessing-pyspark-pycharm/ -------------'''
# Path for spark source folder
os.environ['SPARK_HOME'] = "/Users/Jack/Projects/spark-1.6.1"

# Append pyspark  to Python Path
sys.path.append("/Users/Jack/Projects/spark-1.6.1/python/")
sys.path.append("/Users/Jack/Projects/spark-1.6.1/python/lib/py4j-0.9-src.zip")

try:
    from pyspark import SparkContext
    from pyspark import SparkConf
    print ("Successfully imported Spark Modules")

except ImportError as e:
    print ("Can not import Spark Modules", e)
    sys.exit(1)
''' ----------------------------------------------------------------------------------------------------------------'''

baseDir = os.path.join('data')
FILE0 = os.path.join(baseDir, 'trunkx0.txt')
SPLITTER = '>>>'

def parseData(filenema):
    """ Parse a data file.
    :param filenema:
    :return:
        a RDD of parsed lines
    """
    return (sc.textFile(FILE0,4,0 ))

def split_docs(line):
    line = line.split(SPLITTER)

if __name__ =='__main__':
    sc = SparkContext('local')
    docs = sc.textFile(FILE0, 4).map(split_docs)
    print docs.count()
    words = docs.map(lambda posts, tags: posts).cache()
    tags = docs.map(lambda posts, tags: tags).cache()
    for i in words.take(3):
        print words.count()
    for i in tags.take(3):
        print tags.count()

    # still coding.





