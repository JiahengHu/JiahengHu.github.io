---
layout: single
mathjax: true
title: Machine Learning Notes
---

In this post, I mainly post thoughts from CS 229 by Prof. Andrew Ng and Machine Learning by Prof. John Paisley. It is much like self-displined. Meanwhile, I am always trying to summarize and combine what it is worth in this notes from different machine learning classes that I have been involved. 

It is recommended that you have some basic knowledge about probability, linear algebra and vector calculus before reading the notes. The reading can be done simply from beginning to ending or be indexed from table of contents based on your needs. 

* This will become a table of contents (this text will be scraped).
{:toc}

# Lecture 1 Supervised Learning

A classical learning problem is called supervised learning. In this case, we usually have an input called features and output called target. The goal is that given some features we ask the trained model to predict the output. To do so, we collect a training dataset in which we have a number of pairs of training sample composing of a feature vector and its corresponding output. Since we have ground truth for every single input, we call this type of learning as supervised learning and the learned model as hypothesis. An example can be shown in the table below. 

![Supervise Learning Intuition](/images/cs229_lec1_intuit.png)

When the target output is in continuous space, we call it a regression problem. When the target output is in discrete space, we call it as a classification problem. 
