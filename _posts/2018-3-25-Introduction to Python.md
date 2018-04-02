In this post, I wll mainly post the supplementary materials for the class I am teaching, including shell script, code and etc. I will update it regularly. 

## First Class(2018-3-26)
1 By using {% highlight python %}cd{% endhighlight %} command in both Wdinows and Mac, one can switch path. 

2 "Hello World" example:

{% highlight python %}
#the two statements below are just for testing if both packages are installed
import numpy
import scipy

print 'Hellow World'
{% endhighlight %}

3 "Ask user to input" example:

{% highlight python %}
#the two statements below are just for testing if both packages are installed
age = input("What is your age?")
print "Your age is:" + age
{% endhighlight %}

4 To run the saved script (**test.py**), one should first use {% highlight python %}cd{% endhighlight %} to switch to the path where the **test.py** is saved and then type {% highlight python %}python test.py{% endhighlight %} with **Enter** to run the program. 


## Second Class(2018-4-01)
The slides presented in class can be found at [here][1]

[1]:{{ site.url }}/download/Python基础和人工智能课程（第二讲).pdf

The codes done in the class can be found below. 

{% highlight python %}
# this is comment 
# this is also comment
import numpy as np #this is comment 
# this is comment
import scipy


a = 1
print a
b = 10000
print b

print a
a = 1.0

b = 3.1415926
print b


a = 1
b = 2.0
print a/b
print b/a

a = 1.6
b = int(a)
print b

a = 1
b = float(a)
print b

a = True
b = False
print a
print b

a = 1
b = 1.0
print a == b

a = '6'
b = 6
print a == b

a = 'resent'
print a[0]
print a[0:5]
print a[-1]
print a[2:]
print a[:3]

a = 'I love'
b = 'reading'

print a + b
print a + ' ' + b


a = np.array([[1,2,3],[4,5,6]])
print a[0,:]

a = [1,2,3,4,5]
print len(a)
print a[0:2]
print max(a)
a.append(6)
print a
del a[-1]
print a

a = 3
b = 5
a += b
a = a + b
print a

a = True
b = True

print a and b

{% endhighlight %}
