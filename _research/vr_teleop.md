---
layout: single
mathjax: true
toc: true
toc_sticky: true
author_profile: false
title: VR Teleoperation for Grasping and Data Collection
share: false
permalink: /research/vr_teleop/
---

This work is performed at Columbia Robotics Lab and advised by Professor Peter Allen. This is an ongoing work and only part of results are shown here. 

# Overview

This project is built on top of the [Ros Reality](https://github.com/h2r/ros_reality) project by researchers from Brown University. In this project, we designed a pipeline that allows the robot to **execute complex tasks such as grasping based on human input through VR equipment**.  

This video showcase the whole pipeline where the user controls a fetch robot in real world to:

[0:08 - 0:27] Segment out graspable objects in the scence and run shape completion to get meshes of the objects.

[0:27 - 0:47] The user selects the object to grasp, and the server plans a grasp for the chosen object.

[0:47 - 0:59] Execute the grasp.

{% include video id="LKDF6XDl3kw" provider="youtube" %}



|:-----:|:-------:|
|Platform|Ubuntu 16.04, ROS|
|Programming Language|C#,Python|
|Deep Learning Framework|Tensorflow|
|Research Area|Human-robot interaction|