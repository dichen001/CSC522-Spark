# CSC522-Spark
This repo is for the Spark project from CSC522 Spring 2016 under Dr.Raju.


## Log:

### 03/30/16
 - Create this repo for fast and efficient group cooperation. 
 - Install Spark on my local machine instead of using VirtualBox.


### 03/31/16
 - Set up the Spark Enveriment in PyCharm
 - Get the tf-idf calculation done!
 
#### Note
 - Do NOT need to transform each post (i.e. a list of words in that post) into a huge vector of tf-idf. 
 - Instead, just need to calculate the idf_weight once, and broadcast it to all the nodes.
 - When calculate similarity, or distance between two post, just need to get the tf for these two nodes, multiply each tf with idf by looking up the idf dict.
 - using `sum(a[k]*b[k] for k in a.keys() if k in b.keys())` to get the similarity.

```
>>> post1 = ['foo', 'foo', 'bar', 'bar', 'bar', 'cat', 'school']
>>> post2 = ['foo', 'cat', 'cat', 'dog', 'dog', 'dog', 'dog',]
>>> tf1, tf2 = get_tf(post1), get_tf(post2)
>>> tf1
>>> {'foo': 2, 'bar': 3, 'cat': 1, 'school': 1}
>>> tf2
>>> {'foo': 1, 'cat': 2, 'dog': 4}
# assume we have already get the idf_weights_dict, then we can easily get the tfidf for each post
>>> tfidf1, tfidf1 = get_tfidf(tf1, idf_weights_dict), get_tfidf(tf2, idf_weights_dict)

def dotproduct(a, b):
    """ Compute dot product
    Args:
        a (dictionary): first dictionary of record to value
        b (dictionary): second dictionary of record to value
    Returns:
        dotProd: result of the dot product with the two input dictionaries
    """
    return sum(a[k]*b[k] for k in a.keys() if k in b.keys())

>>> similarity = dotproduct(tfidf1, tfidf2)/(dotproduct(tfidf1, tfidf1)*dotproduct(tfidf2, tfidf2))
```

### 04/09/16
 - [x] new version for calculating TFIDF
 - [x] TFIDF reduction:
   - collecting all the corpus
   - for each word, add all the tfidf among all the posts
   - sort the tfidf after sum
   - only keep top 1000 or N words with the relative higher tfidf in total
 - [x] apply PCA after TFIDF reduction, to get 10 dimension vector representation for each post
 - [x] speed now: (20mins) 100,000 posts --> tfidf --> "tfidf reduction to 1000 words"(newly added) --> PCA to 10 dimension
 - [x] add the save RDD and load RDD feature

## Next:
 - [] train and test using package directly
 - [] implement our own classifier.
 
Notes: check how many tages will be in 1 million  
