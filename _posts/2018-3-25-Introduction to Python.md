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

[1]:{{ site.url }}/download/Python基础和人工智能课程(第二讲).pdf

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

## Third Class(2018-4-05)
The slides presented in class can be found at [here][1]

[1]:{{ site.url }}/download/Python和人工智能基础课程（第三讲).pdf

The codes done in the class can be found below. 

{% highlight python %}
import numpy
import scipy

rad = input('What is the radius?')
rad = float(rad)
area = 3.14*rad**2
print area

total_value = input('what is the value of the house?')
mortgage_year = input('how many years do you prefer?')

group=[]
while 1:
	val = input('what is your input value?')
	if len(group) > 10:
		group.append(val)
		group.sort()
		del group[0]
	else:
		group.append(val)
		group.sort()


a = 1
b = 0
c = 1
print (b or a) and c
print b or a and c

y = [1,2,3,4,5]
x = 1
z = 0
print x in y
print z in y
print 

a = 6
b = '6'
print a is b
print a is not b
print type(a) is type(b)
print type(a) is int

a = 100
if a:
	print 'I am running this code'
	print a


a = 0
if a:
	print 'I am still here'
print 'after if indent'

if True:
	a = 5
print a

a = 0
if a:
	print 'I am True'
	print a
else:
	print 'I am False'

a = 100
b = 1000
if a == 100:
	print 'a is true'
if b == 1000:
	print 'b is true'
else:
	print 'else case'

a = [1,2,3,4,5,6,7,8,9,10]
for idx in range(0,len(a)):
	print a[idx]

index = 0
for element in a:
	print element
	index = index + 1
print 'done'

for idx,val in enumerate(a):
	print 'index is:',idx
	print 'value is:',val

count = 0
while count <10:
	print 'the count is:',count
	# count = count + 1
print 'done'

a = 1
while a == 1:print 'a = 1'
print 'done'

for letter in 'Programming':
	if letter == 'm':
		break
	print

for letter in 'Programming':
	if letter == 'm':
		pass
	print 'current letter is:',letter

import time
localtime = time.localtime(time.time())
print localtime

a = 'abcsdadhe'
{% endhighlight %}

