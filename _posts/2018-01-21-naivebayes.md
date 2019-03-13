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
