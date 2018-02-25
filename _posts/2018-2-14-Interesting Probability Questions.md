---
layout: post
mathjax: true
title: Interesting Probability Questions
---

In this blog, I will keep updating interesting probability questions as time goes by. The questions listed are likely to apprear in an interview. The answers written by me will be given as well. Please be skeptical about the answers. 


1.  In coin tossing, assuming it is a fair coin, what is the expected toss of the event that a head comes before a tail?

Solution: Let **A** denote the event that a head comes before a tail and **H<sub>i</sub>** denote the event that first toss is a head. Thus, we have: 

$$E(A) = 0.5\times E(A | H_1) + 0.5\times E(A | T_1) \label{1}\tag{1}$$

To solve this, we look into each term. 

$$E(A | H_1) = 0.5\times 2 + 0.5\times (1 + E(A | H_1) \label{2}\tag{2}$$

$$E(A | T_1) = 1 + E(A) \label{3}\tag{3}$$

From \ref{2}, we have $E(A \lvert H_1) = 3$. We plug this with \ref{3} to \ref{1} so as to solve it: $E(A) = 4$

2.  In a similar manner, we have an fair coin, what is the expected toss of the event that two heads comes consecutively?

Solution: Test it with yourself. My answer is 6. 
