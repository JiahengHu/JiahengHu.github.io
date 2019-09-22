---
layout: single
mathjax: true
toc: true
toc_sticky: true
author_profile: true
title: Trajectory Following and Gait Generation through Reinforcement Learning
share: false
permalink: /research/snake_traj/
---

This work is advised by Professor Tony Dear. This is an ongoing work and only part of results are shown here. 

# Overview

In this project, we designed a pipeline that allows a three link wheeled snake robot to **learn to follow a trajectory trhough reinforcement learning**.  

# Model

The model of the robot we are using can be seen in fig 1.

![fig 1](/_research/images/snake.png)

I wrote the lagrangian equations for it, with three wheels providing three nonholonomic constraints, and use it as our simulator during training. 

# Training

We train the agent through TD3 algorithm, which is a variation of DDPG. Through reward shaping and cirriculum learning, we are able to get the snake to locomote based on the guidance of a 2D trajectory. Please see the gif below for an illustration, where the red dots are the trajectory feeded to the agent, and the blue dot is the head coordinate of the snake.

 ![staright line](/_research/images/straight_line.gif)
 ![curve line](/_research/images/curve_line.gif)


**This project is expected to be updated in the near future with regard to the gait generation part**.


|:-----:|:-------:|
|Platform|Ubuntu 16.04|
|Programming Language|Python|
|Deep Learning Framework|Tensorflow|
|Research Area|motion planning, robot learning|