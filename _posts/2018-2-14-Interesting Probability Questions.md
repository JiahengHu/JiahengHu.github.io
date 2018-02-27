---
layout: post
mathjax: true
title: Interesting Probability Questions
---

In this blog, I will keep updating interesting probability questions as time goes by. The questions listed are likely to apprear in interview. The answers written by me will be given as well. Please be skeptical about the answers. 


**1 In coin tossing, assuming it is a fair coin, what is the expected tosses of the event that a head comes before a tail?**

Solution: Let A denote the event that a head comes before a tail and $H_i$ denote the event that first toss is a head. Thus, we have: 

$$E(A) = 0.5\times E(A | H_1) + 0.5\times E(A | T_1) \label{1}\tag{1}$$

To solve this, we look into each term. 

$$E(A | H_1) = 0.5\times 2 + 0.5\times (1 + E(A | H_1) \label{2}\tag{2}$$

$$E(A | T_1) = 1 + E(A) \label{3}\tag{3}$$

From \ref{2}, we have $E(A \lvert H_1) = 3$. We plug this with \ref{3} to \ref{1} so as to solve it: $E(A) = 4$

**2 In a similar manner, we have a fair coin, what is the expected tosses of the event that two heads comes consecutively?**

Solution: Test it with yourself. My answer is 6. 

**3 Given that we have a fair coin, what is the expected tosses of the event that two heads appear?**

Solution: This question requires a bit tricks. Intuitively, suppose that we have probability p to have a head, if we toss n times, the expected times of heads is $np$. If we set it to k, the number of heads we want, then $n=\frac{k}{p}$. So the quick answer is 4. 

Now, I show the completed steps. Let $X = X_1 + X_2 + \dots + X_k$ where k is the number of heads we want (k=2 in this case) and $X_i$ denotes the event of ith head after getting (i-1) heads. 

**4 Tossing a fiar coin, what is the probability of the event that HTT comes before HHT?**

Solution: This quesiton is a bit trickier than the Q1 since it asks for a probability instead of a expected number. However, the idea is essentially the same. Let A denote the event that HTT comes before HHT. Then, it is easy to say:

$$P(A) = P(A \lvert H_1)P(H_1) + P(A \lvert T_1)P(T_1) = 0.5P(A \lvert H_1) + 0.5P(A \lvert T_1) \label{4}\tag{4}$$

$$P(A \lvert H_1) = P(A \lvert H_1 H_2)P(H_2) + P(A \lvert H_1 T_2)P(T_2) = 0 + 0.5P(A \lvert H_1 T_2) \label{5}\tag{5}$$

$$P(A \lvert H_1 T_2) =(A \lvert H_1 T_2 H_3)P(H_3) + P(A \lvert H_1 T_2 T_3)P(T_3) = 0.5P(A \lvert H_1) + 0.5 \label{6}\tag{6}$$

Combining \ref{5} and \ref{6}, we have $P(A\lvert H_1) = \frac{1}{3}$. From \ref{4}, we have $P(A) = P(A\lvert H_1) = \frac{1}{3}$.
