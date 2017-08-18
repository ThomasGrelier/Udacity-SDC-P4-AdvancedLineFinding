# **Advanced Lane Finding Project**

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Overview
This repository contains the work I did within **Project #4 of Udacity's Self-Driving Car Nanodegree Program**. Objective of the project is to identify the lane boundaries in a video from a front-facing camera on a car.

*An example of achieved lane detection*

![](./test_images_videos/lane-drawn.jpg)


## Project objectives and steps

The goals / steps of this project are the following:

 * Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
 * Apply a distortion correction to raw images.
 * Use color transforms and gradients to create a thresholded binary image.
 *  Apply a perspective transform to rectify binary image (“birds-eye view”).
 *  Detect lane pixels and fit to find the lane boundary.
 *  Determine the curvature of the lane and vehicle position with respect to center.
 * Warp the detected lane boundaries back onto the original image.
 * Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.



## Repository content

The repository includes the following files:

* _image\_pipeline.ipynb_ : a jupyter notebook of the pipeline to process images
* _video\_pipeline.ipynb_ : a jupyter notebook of the pipeline to process videos
* _pipeline\_functions.py_ : functions called by the two previous pipelines
* _writeup\_report.md_ : work writeup

And two directories:

* _/images_: images displayed in the report
* _/videos_: videos of the detected lanes

## Dependencies

This lab requires the [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit).

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.


