---
title: "Naive Bayes from scratch using Spark RDDs"
date: 2018-01-21
tags: [NaiveBayes, Spark, Machine Learning, Data Science, Natural language processing, Python]
#header:
#  image: "/images/perceptron/percept.jpg"
excerpt: "NaiveBayes algorithm, Machine Learning"
mathjax: "true"
---

# Naive Bayes using PySpark package

Naive bayes is a great algorithm especially for classification task in Natural language processing. This post tries to provide implementation of Naive bayes algorithm from scratch using Spark RDDs. Spark is a very good Big data processing framework and it provides PySpark package for python. This PySpark package has this special data type called RDDs(Resilient Distributed Datasets) which are immutable and partitioned collection of records. We take a NLP problem, a document classification task where given a document, we have to classify it to one of the topic labels. Topic labels can be Management, Economics, Geometry etc.

## fit method

Fit method is generic method for any Machine learning algorithm, used for training on the dataset. Generally, Fit method takes two arguments: dataset and labels. Dataset in this context is set of document's features and values. It is RDD of form ```((id, feature), value)``` where ```id``` is the unique identifier of a document, ```feature``` in our case is word in the document (It can be ngrams feature as well) and ```value``` is value of the feature. Here, value can be TFIDF value or simply word frequency. Labels are set of labels for given document and they are represented as RDD of the form ```(id, label)```. In our fit method ```x``` is dataset and ```y``` is labels.

First we enumerate all labels and extract total number of distinct labels as well as a RDD containing label and it's frequency of the form ```(label, count)```
```python
vals = y.values()
labels = vals.distinct()
counts = vals.countByValue()  # {label: count}
```

Now we extract size of vocabulary by enumerating through dataset RDD ```x```
```python
vocabulary_size = x.keys().values().distinct().count()
```

Now we calculate prior probabilities for labels and take $$log$$
```python
import numpy as np
n = vals.count()
priors = {k:v/n for k,v in counts.items()} # {label: prior}
log_priors = {k:np.log(v) for k,v in priors.items()} # {label: log(prior)}
```

I want to mention one thing here that Spark accesses RDDs by tracing it back. It means that once you perform an operation on RDD, it doesn't actually execute the operation until you try to access that RDD so you might perform multiple operations on RDD but all of them will be executed by tracing them back when you access RDD. Now, you don't want it to trace back too deep so it is a good idea to broadcast your variables periodically. Below is how you broadcast RDD (you can only broadcast RDD in form of a dictionary or map).
```python
y  = y.collectAsMap()  # {id: label}
y = self.ctx.broadcast(y)
```

Now, we want to create a RDD which represents features by label. We will replace ```id``` of the ```x``` dataset with it's feature from ```y``` labels RDD. Then we will sum up the ```value``` for all unique combination of ```(label, feature)``` which is a reduce operation. In short we will transform dataset RDD of the type ```((id, feature), value)``` to ```((label, feature), value)```.
```python
def doc_to_label(x):
  ((doc_id, feature), value) = x
  label = y.value[doc_id]
  return ((label, feature), value)
by_label = x.map(doc_to_label)  # ((label, feature), value)
by_label = by_label.reduceByKey(lambda x, y: x+y)
by_label_map = by_label.collectAsMap()
```

Now, we will perform cartesian product of all unique labels with our dataset. The cartesian product will be of the form ```(label, (id, feature), value)```. We don't need ```id``` so we will additionally remove it from the cartesian product yielding RDD of the form ```((label, feature), value)```. Please note that this RDD is same form as that of ```by_label``` RDD that we got in the last step but the data and meaning it contains is different.
```python
cartesian_label = labels.cartesian(x) # (label, (id, feature), value)
def restructure_cartesian_product(a):
  (label, ((id, feature), value)) = a
  return ((label, feature), value)
cartesian_product_rdd = cartesian_label.map(restructure_cartesian_product)
```

Now, we are coming to the core part of Naive Bayes where we will calculate likelyhood probability of a word (feature) given a label. All of the steps we did previously might have seemed random but it will all connect. Let is forget about TFIDF feature for a while. If we consider simple word as feature and it's frequency as value, likelyhood probability of a word given a label is given by dividing number of times word appears in all documents having the label by sum of vocabulary size and label frequency.

We already calculated the numerator part when we extracted ```by_label_map```. Now we calculate the denominator part:

```python
prob_denom = {k:(v+vocabulary_size) for k,v in counts.items()}
prob_denom = self.ctx.broadcast(prob_denom)
```
Now, we create log likelyhood probability RDD which will give likelyhood probability of a word or in our context feature given a label. 

```python
def calculate_likelyhood_probability(by_label):
  ((label, feature), value) = by_label
  value = (by_label_map.get((label, feature),0)+1)/prob_denom.value[label]
  return ((label, feature), np.log(value))
log_likelyhood_probability = cartesian_product_rdd.map(calculate_likelyhood_probability) # ((label, feature), log likelyhood value)
```
## predict method
Predict method is also a generic method to all Machine learning algorithms, used to predict on the test dataset. Generally, it has only one input which is a test dataset. We say test dataset because most of the time, this dataset is unseen data. This dataset is also of the same form (```((id, feature), value)```) as the dataset input of fit method. 

We first perform cartesian product of dataset and distinct labels but this time we will not get rid of ```id``` because we need it to map our classification label to it. So, our cartesian product RDD will be of the form ```(label, ((id, feature), value))```. Additionally, we reorganize this RDD in the form of ```((label, feature), (id, value))``` because we have ```(label, feature)``` as key in our log likelyhood probability RDD and we want to perform a join between two. Our resultant RDD after join will be of the form ```((label, feature), ((id, value), log_likelyhood))```
```python
# Cross and rekey by label
def key_by_label(a):
    (label, ((id, feature), value)) = a
    return ((label, feature), (id, value))
# x has initial shape ((id, feature), value)
x = self.labels.cartesian(x)  # (label, ((id, feature), value))
x = x.map(key_by_label)  # ((label, feature), (id, value))

# compute the probability for test data by joining x with log likelyhood probability
x = x.join(self.log_likelyhood_probability) # ((label, feature), ((id, value), log_likelyhood))
```

Now we will drop the ```value``` and convert the RDD in the form ```((id, label, feature), log_likelyhood))```

```python
 # dropping the value and converting x RDD to ((id, label, feature), log_likelyhood))
def id_in_key(a):
    ((label, feature), ((id, value), log_likelyhood)) = a
    return ((id, label, feature), log_likelyhood)
x = x.map(id_in_key) # ((id, label, feature), log_likelyhood)
```

Now we want to get rid of the ```feature ``` as well because we have to classify document by label but in doing so we will also add up log likelyhood of all ```feature``` per ```id``` per ```label```. Resultant RDD will be of the form ```((id, label), sum_log_likelyhood)```
```python
# adding all features log_likelyhood per id per label
def remove_feature(a):
    ((id, label, feature), log_likelyhood) = a
    return ((id, label), log_likelyhood)
x = x.map(remove_feature)
x = x.reduceByKey(lambda x, y: x+y) # ((id, label), sum_log_likelyhood)
```

Now we just add log priors of labels to the sum we calculated in last step which will give us RDD of the form ```((id, label), sum_log_likelyhood+log_prior)```
```python
# adding log prior to this
log_priors = self.log_priors
def add_log_prior(a):
    ((id, labels), sum_words) = a
    return ((id, labels), sum_words + log_priors.get(labels))
x = x.map(add_log_prior) # ((id, label), sum_words+log_prior)

```

We are on the last step now. We just have to find out which ```label``` has maximum sum for the ```id``` which can be performed by below code.
```python
def key_by_id(a):
    ((id, label), rank) = a
    return (id, (label, rank))
def max_label(a, b):
    (label_a, rank_a) = a
    (label_b, rank_b) = b
    return b if rank_a < rank_b else a
def flatten(a):
    (id, (label, rank)) = a
    return (id, label)
x = x.map(key_by_id)  # (id, (label, rank))

x = x.reduceByKey(max_label)  # (id, (label, rank))
x = x.map(flatten)
```

Here ```x``` RDD is of the form ```(id, label)``` which is a mapping of document to the classified label.

You can visit our repo for more methods on perfroming document classification without using any Machine learning package [here](https://github.com/ankit-vaghela30/Distributed-Documents-classification)
