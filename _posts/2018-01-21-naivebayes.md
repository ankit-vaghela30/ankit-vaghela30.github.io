---
title: "Naive Bayes from scratch using Spark RDDs"
date: 2018-01-21
tags: [NaiveBayes, Spark, Machine Learning, Data Science, Natural language processing]
#header:
#  image: "/images/perceptron/percept.jpg"
excerpt: "NaiveBayes algorithm, Machine Learning"
mathjax: "true"
---

# Naive Bayes from scratch in 

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




### H3 Heading

Here's some basic text.

And here's some *italics*

Here's some **bold** text.

What about a [link](https://github.com/dataoptimal)?

Here's a bulleted list:
* First item
+ Second item
- Third item

Here's a numbered list:
1. First
2. Second
3. Third

Python code block:
```python
    import numpy as np

    def test_function(x, y):
      z = np.sum(x,y)
      return z
```

R code block:
```r
library(tidyverse)
df <- read_csv("some_file.csv")
head(df)
```

Here's some inline code `x+y`.

Here's an image:
<img src="{{ site.url }}{{ site.baseurl }}/images/perceptron/linsep.jpg" alt="linearly separable data">

Here's another image using Kramdown:
![alt]({{ site.url }}{{ site.baseurl }}/images/perceptron/linsep.jpg)

Here's some math:

$$z=x+y$$

You can also put it inline $$z=x+y$$
