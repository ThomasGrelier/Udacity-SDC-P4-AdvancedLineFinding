# -*- coding: utf-8 -*-
"""
Created on Wed Aug 09 16:01:33 2017

This file was developed for Project#4 of Self Driving Car Nanodegree
program, entitled "Advanced lane finding". 
It defines functions used by image_pipeline.ipynb and video_pipeline.ipynb.

@author: greliert
"""

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
from scipy.signal import savgol_filter

def calib_camera(calib_path, n_corners):
    '''
    Compute camera calibration matrix and distortion coefficients using
    chessboard images
    Inputs:
    - calib_path: path of directory containing the chessboard images
    - n_corners: nb of corners of the chessboard (nx,ny)
    Outputs:
    - ret: boolean, indicates if calibration is succesful
    - mtx: calibration matrix (3,3)
    - dist: distortion coefficients (5,)
    '''
    # Defining coordinates of the chess board in the real world
    # We assume for simplicity that z-coordinate is equal to 0
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    
    # Make a list of calibration images
    images = glob.glob(calib_path+'*.jpg')
    
    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # print img.shape
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, n_corners, None)
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    
            # Draw and display the corners
            cv2.drawChessboardCorners(img, n_corners, corners, ret)
            write_name = fname[:-4]+'_corners.png'
            cv2.imwrite(write_name, img)
        else:
            print('Corners not found on picture: {}'.format(fname))
    
    # Compute calibration matrix and distortion coefficients
    img_size = (img.shape[1], img.shape[0])  # caution: image dimension = (n_col, n_row)
    if np.size(objpoints)>0:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
        
        # Save the camera calibration result for later use (we won't care about rvecs / tvecs)
        dist_pickle = {}
        dist_pickle["mtx"] = mtx
        dist_pickle["dist"] = dist
        pickle.dump(dist_pickle, open( calib_path+'calib.p', "wb" ) )

        return ret, mtx, dist
    else:
        return 0, 0, 0

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0,255),color='grayscale'):
    '''
    Returns a mask indicating which pixels of the input images have a gradient
    (computed over one axis only) within the 'thresh' bounds
    Inputs:
    - img: image
    - orient: 'x' or 'y', axis along which to compute gradient
    - sobel_kernel: size of the kernel of the sobel operator
    - thresh : thresholds (min,max) of the mask (between 0 and 255)
    - color: colorspace to use for computing gradient ('grayscale' or 'S')
    Outputs:
    - binary_output: mask (0|1)
    '''
    # check inputs    
    assert (orient=='x' or orient=='y'), 'orient must be equal either to ''x'' or ''y'''
    assert (color=='grayscale' or color=='S'), 'color must be gray or S'
    assert (sobel_kernel%2)==1, 'kernel size must be odd'
    # Apply the following steps to img
    # 1) Convert to grayscale
    if color=='grayscale':
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:,:,2]
        
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient=='x':
        sobel = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=sobel_kernel)
    elif orient=='y':
        sobel = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=sobel_kernel)
    else:
        pass    
    # 3) Take the absolute value of the gradient
    sobel = np.abs(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    sobel = np.uint8(sobel/np.max(sobel)*255)
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    # Return this mask as the binary_output image
    binary_output =  np.zeros_like(sobel)
    binary_output[(sobel >= thresh[0]) & (sobel <= thresh[1])] = 1
    return binary_output

def mag_thresh(img, sobel_kernel=3, thresh=(0,255),color='grayscale'):
    '''
    Returns a mask indicating which pixels of the input images have a gradient 
    magnitude within the 'thresh' bounds
    Inputs:
    - img: image
    - sobel_kernel: size of the kernel of the sobel operator
    - mag_thresh : thresholds (min,max) of the mask (between 0 and 255)
    - color: colorspace to use for computing gradient ('grayscale' or 'S')
    Outputs:
    - binary_output: mask (0|1)
    '''
    # check inputs    
    assert (sobel_kernel%2)==1, 'kernel size must be odd'
    assert (color=='grayscale' or color=='S'), 'color must be gray or S'
    
    # 1) Convert to grayscale or Saturation
    if color=='grayscale':
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    elif color=='S':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:,:,2]
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=sobel_kernel)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=sobel_kernel)
    # 3) Calculate the magnitude 
    sobel = np.sqrt(sobelx**2+sobely**2)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    absgraddir = np.uint8(sobel/np.max(sobel)*255)
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    # Return this mask as the binary_output image
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(sobel >= thresh[0]) & (sobel <= thresh[1])] = 1
    return binary_output

def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2),color='grayscale'):
    '''
    Returns a mask indicating which pixels of the input images have a gradient 
    direction within the 'thresh' bounds
    Inputs:
    - img: image
    - sobel_kernel: size of the kernel of the sobel operator
    - mag_thresh : thresholds (min,max) of the mask, in radians
    - color: colorspace to use for computing gradient ('grayscale' or 'S')
    Outputs:
    - binary_output: mask (0|1)
    '''
    # check inputs    
    assert (sobel_kernel%2)==1, 'kernel size must be odd'
    assert (color=='grayscale' or color=='S'), 'color must be gray or S'
    # 1) Convert to grayscale or Saturation
    if color=='grayscale':
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:,:,2]
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=sobel_kernel)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=sobel_kernel)
    # 3) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    # 4) Create a mask of 1's where the gradient direction 
            # is > thresh_min and < thresh_max
    # Return this mask as the binary_output image
    binary_output =  np.zeros_like(absgraddir,dtype=np.uint8)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output

def color_thresh(img, colorspace='HLS',thresh=[[0,255],[0,255],[0,255]]):
    '''
    Convert RGB image to colorspace.
    Returns a mask indicating which pixels of the input image have a magnitude
    within 'thresh' bounds (included)
    Inputs:
    - img: RGB image
    - colorspace: ('HLS'|'HSV'|'GRAY')
    - thresh : thresholds (min,max) (between 0 and 255) for the three channels
    Outputs:
    - binary_output: mask (0|1)
    '''
    # check inputs
    ['HLS','HSV','GRAY'].index(colorspace)
    color_binary = np.zeros_like(img[:,:,0])
    # Convert to other color space and extract channel
#    exec('img_conv = cv2.cvtColor(img, cv2.COLOR_RGB2'+colorspace+').astype(np.float)')
    if colorspace == 'HLS':    
        img_conv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    elif colorspace == 'GRAY':
        img_conv = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float)
    elif colorspace == 'HSV':    
        img_conv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float)

    # Thresholding
    if colorspace == 'GRAY':
        color_binary[(img_conv >= thresh[0][0]) & (img_conv <= thresh[0][1])] = 1
    else:
        color_binary[(img_conv[:,:,0] >= thresh[0][0]) & (img_conv[:,:,0] <= thresh[0][1]) & (img_conv[:,:,1] >= thresh[1][0]) & (img_conv[:,:,1] <= thresh[1][1]) & (img_conv[:,:,2] >= thresh[2][0]) & (img_conv[:,:,2] <= thresh[2][1])] = 1
    return color_binary

def rectify_image(img, src, dst):
    '''
    Applies a perspective transformation to an image
    Inputs:
    - img: image
    - src: Coordinates of quadrangle vertices in the source image
    - dst: Coordinates of the corresponding quadrangle vertices in the destination image
    Outputs:
    - warp_img: warped image
    - M: transformation matrix
    '''
    # Compute the transformation matrix
    M = cv2.getPerspectiveTransform(src,dst)
    # Warp image to a top-down view
    warp_img = cv2.warpPerspective(img, M, img.shape[1::-1], flags=cv2.INTER_LINEAR) 
    return warp_img, M

def draw_lines(img, lines, color=[0, 255, 0], thickness=3):
    """
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    Inputs:
    - img: image on which to draw lines
    - lines: list of (int) coordinates of the lines [[x1,y1,x2,y2],[...],[...]]
    - color: color of the line
    - thickness: thickness of the line
    """
    for line in lines:
        cv2.line(img, (line[0], line[1]), (line[2], line[3]), color, thickness)

def fit_poly(nonzero,lane_inds):
    '''
    Fit a second order polynomial on non-zero values of a mask
    Inputs:
    - nonzero: coordinates ([xi],[yi]) of non-zero values of the mask
    - lane_inds: indices of 'nonzero' to use for fitting
    - color = colormap 
    - title = title
    '''    
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Extract left and right line pixel positions
    x = nonzerox[lane_inds]
    y = nonzeroy[lane_inds] 
    # Fit a second order polynomial to each
    return np.polyfit(y, x, 2)
            
def subplot_save_img(img1,img2,savepath,cmap=['jet','jet'],title=['Original image','Processed image']):
    '''
    Plot two images on a subplot and save the image
    Inputs:
    - img1 = first image, img2 = second image
    - savepath = path to save figure
    - cmap = colormap for images
    - title = title for images
    '''    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,7))
    ax1.imshow(img1, cmap=cmap[0])
    ax1.set_title(title[0], fontsize=10)
    ax1.axis('on')
    ax2.imshow(img2, cmap=cmap[1])
    ax2.set_title(title[1], fontsize=10)
    ax2.axis('on')
    plt.savefig(savepath)

def plot_save_img(img,savepath,cmap='jet',title='Original image'):
    '''
    Plot and save an image
    Inputs:
    - img1 = image
    - savepath = path to save figure
    - cmap = colormap 
    - title = title
    '''    
    plt.figure(figsize=(10,5))
    plt.title(title, fontsize=10)
    plt.axis('on')
    plt.imshow(img, cmap=cmap)
    plt.savefig(savepath)

def plot_save(x,savepath,color='blue',title='Original image',xlabel='',ylabel=''):
    '''
    Plot and save an image
    Inputs:
    - x = series to plot
    - savepath = path to save figure
    - color = colormap 
    - title = title
    '''    
    plt.figure(figsize=(10,5))
    plt.title(title, fontsize=10), plt.grid('on')
    plt.axis('on'), plt.xlabel(xlabel), plt.ylabel(ylabel)
    plt.plot(x, color=color), plt.savefig(savepath)


def find_lines(mask_warp, left_fit, right_fit, left_flag, right_flag, savepath=0):
    '''
    Detect and fit a second order polynomial on the right and left lines of a
    warped mask (bird's eye view)
    Two methods are implemented. First one has no a priori values. Second one
    takes a priori values for polynomial coefficients as input.
    Inputs:
    - mask_warp = binary mask
    - left_fit: previous values for coefficients of the polynomial (A*x2+B*x+*C) for left line
    - right_fit: a priori values for coefficients of the polynomial (A*x2+B*x+*C) for right line
    - left_flag: indicates if 'left_fit' is valid
    - right_flag: indicates if 'right_fit' is valid
    - savepath: path to save figure. If not provided, figure is not plotted
    Outputs:
    - left_fit_out: coefficients of the polynomial (A*x2+B*x+*C) for left line
    - right_fit_out: coefficients of the polynomial (A*x2+B*x+*C) for right line
    '''    
    
    # Internal parameters
    margin = 100        # Half-width of the window
    minpix = 100        # Minimum number of pixels found to recenter window
    max_step_x = 100    # Maximum shift of x between two consecutive windows
    leftx_def = 300     # default x-position of left window
    rightx_def = 980    # default x-position of right window
    nwindows = 9        # Set height of windows
    
    # - Detection and fitting of 2nd order polynomials: 2 distinct cases
    # FIRST CASE: no a priori values are available in input
    if (left_flag == 0) & (right_flag==0):
        # compute histogram along y axis
        histogram = np.sum(mask_warp[mask_warp.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((mask_warp, mask_warp, mask_warp))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        # Choose the number of sliding windows
        window_height = np.int(mask_warp.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = mask_warp.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = mask_warp.shape[0] - (window+1)*window_height
            win_y_high = mask_warp.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            # Following code was modified to make detection more robust
            # If enough pixels, take the mean over all pixels
            # Else, fit a 2nd order polynomial on stored non zero pixels to predict
            # x central position of next window  (if enough data available!)
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            else:
                left_lane_inds_ = np.concatenate(left_lane_inds)
                if np.size(left_lane_inds_)>minpix:
                    left_fit = fit_poly(nonzero,left_lane_inds_)    
                    leftx_next = np.int(left_fit[0]*win_y_high**2 + left_fit[1]*win_y_high + left_fit[2])
                    # check that window shift is not too big!
                    if np.abs(leftx_next-leftx_current)<max_step_x:
                        leftx_current = leftx_next
                else:
                    leftx_current = leftx_def
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            else:
                right_lane_inds_ = np.concatenate(right_lane_inds)
                if np.size(right_lane_inds_)>minpix:
                    right_fit = fit_poly(nonzero,right_lane_inds_)    
                    rightx_next = np.int(right_fit[0]*win_y_high**2 + right_fit[1]*win_y_high + right_fit[2])
                    # check that window shift is not too big!
                    if np.abs(rightx_next-rightx_current)<max_step_x:
                        rightx_current = rightx_next
                else:
                    rightx_current = rightx_def
            if 0:
                print('Nb of good pixel %d / %d' %(len(good_left_inds),len(good_right_inds)))
                print('x window: %d / %d' %(leftx_current,rightx_current))
                
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        # Fit 2nd order polynomial
        if np.size(left_lane_inds)>0:
            left_fit_out = fit_poly(nonzero,left_lane_inds)
        else:
            left_fit_out = np.array([0,0,0])
        if np.size(right_lane_inds)>0:
            right_fit_out = fit_poly(nonzero,right_lane_inds)    
        else:
            right_fit_out = np.array([0,0,0])
            
        if savepath!=0:
            # PLOT
            plot_save(histogram,savepath[:-4]+'_histo.jpg',title='Mask histogram along x',xlabel='Pixel position',ylabel='Counts')
            # Generate x and y values for plotting the fitted polynomial
            ploty = np.linspace(0, mask_warp.shape[0]-1, mask_warp.shape[0] )
            left_fitx = left_fit_out[0]*ploty**2 + left_fit_out[1]*ploty + left_fit_out[2]
            right_fitx = right_fit_out[0]*ploty**2 + right_fit_out[1]*ploty + right_fit_out[2]
            # Set left line pixels in red, and right line pixels in blue
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
            plt.figure(figsize=(10,5))
            plt.imshow(out_img)
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
            plt.xlim(0, mask_warp.shape[1])
            plt.ylim(mask_warp.shape[0], 0)
            plt.savefig(savepath)

    # SECOND CASE: a priori valid values are available in input 
    else:
        # if one of the left/right previous fit is invalid, take the valid one
        # to initialize the other
        lane_width = 685
        #y_max = mask_warp.shape[0]-1
        if left_flag == 0:
            left_fit = right_fit-np.array([0,0,lane_width])
            margin = 120         # increase half-width of the window
        elif right_flag == 0:
            right_fit = left_fit+np.array([0,0,lane_width])
            margin = 120         # increase half-width of the window
            
        nonzero = mask_warp.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin))
                & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin))
                & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  
        
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        if np.size(leftx)>0:
            left_fit_out = np.polyfit(lefty, leftx, 2)
        else:
            left_fit_out = np.array([0,0,0])
        if np.size(rightx)>0:
            right_fit_out = np.polyfit(righty, rightx, 2)    
        else:
            right_fit_out = np.array([0,0,0])
        
        if savepath!=0:
            # PLOT
            # Generate x and y values for plotting
            ploty = np.linspace(0, mask_warp.shape[0]-1, mask_warp.shape[0] )
            left_fit_outx = left_fit_out[0]*ploty**2 + left_fit_out[1]*ploty + left_fit_out[2]
            right_fit_outx = right_fit_out[0]*ploty**2 + right_fit_out[1]*ploty + right_fit_out[2]
            # Create an image to draw on and an image to show the selection window
            out_img = np.dstack((mask_warp, mask_warp, mask_warp))*255
            window_img = np.zeros_like(out_img)
            # Color in left and right line pixels
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
            
            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))
            
            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
            result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
            
            plt.figure(figsize=(10,5))
            plt.imshow(result)
            plt.plot(left_fit_outx, ploty, color='yellow')
            plt.plot(right_fit_outx, ploty, color='yellow')
            plt.xlim(0, mask_warp.shape[1])
            plt.ylim(mask_warp.shape[0], 0)
            plt.savefig(savepath)
    
    return left_fit_out, right_fit_out

def sanity_check(left_fit, right_fit, left_fit_old, right_fit_old, left_flag_old, right_flag_old):
    '''
    Check if left and line right estimates are valid.
    # Check if values are not all equal to zero
    # Check that absolute values are within thresholds
    # Check the difference with previous value (if the latter is valid) is 
      within thresholds
    
    Inputs:
    - left_fit: coefficients of the polynomial (A*x2+B*x+*C) for left line
    - right_fit: coefficients of the polynomial (A*x2+B*x+*C) for right line
    - left_fit_old: previous values of coefficients for left line 
    - right_fit_old: previous values of coefficients for right line
    - left_flag_old: previous values of flag for left line 
    - right_flag_old: previous values of flag for right line
    Outputs:
    - left_flag: indicates if values for left line are valid (=1)
    - right_flag: indicates if values for right line are valid (=1)
    '''    
    # Internal Parameters
    # thresholds on absolute values of coefficients
    thres_abs = [0.001, 1.6, 550]
    # thresholds on coefficient variations
    thres_diff_old = [0.0005, 0.4, 150]
    
    left_flag = 1
    right_flag = 1
    left_fit_ = left_fit - np.array([0,0,310])
    right_fit_ = right_fit - np.array([0,0,1005])
    
    # Checks for left values
    if (not np.any(left_fit)):
        left_flag = 0
    elif (not all(np.abs(left_fit_)<thres_abs)):
        left_flag = 0
    elif left_flag_old:
        if (not all(np.abs(left_fit-left_fit_old)<thres_diff_old)):
            left_flag = 0
    # Checks for right values   
    if (not np.any(right_fit)):
        right_flag = 0
    elif (not all(np.abs(right_fit_)<thres_abs)):
        right_flag = 0
    elif right_flag_old:
        if (not all(np.abs(right_fit-right_fit_old)<thres_diff_old)):
            right_flag = 0
        
    return left_flag, right_flag

def smooth(data,flag,n_win,n_min,n_last,poly_order):
    '''
    Computes a smoothed value from an input time series 'data' using a polynomial fitting
    Using only valid data, as indicated by 'flag' input.
    Returns only the last smoothed value.
    Smoothing is not performed if the number of valid samples is less than n_min
    or if the 'n_last' trailing values are invalid 
    
    Inputs:
    - data: time series (shape=(n,m))
    - flag: data validity (shape=(n,))
    - n_win: size of the smoothing window
    - n_min: min number of samples to apply smoothing
    - n_last: number of trailing values that must be different from zero
    - poly_order: polynomial order
    Outputs:
    - data_smooth: smoothed value (shape=(1,m))
    - process_flag: indicates if smoothing was done
    '''

    assert n_min<=n_win
    
    # for initial points, reduce n_win & n_min & n_last
    n_win = np.min([n_win,len(data)])
    n_last = np.min([n_last,len(data)])
    
    if n_min > n_win:
        n_min = n_win
    ind = np.arange(len(data))
    ind_ok = flag[-n_win:]==1
    n_ok = np.sum(ind_ok)
    
    if (n_ok<n_min) | (not np.any(flag[-n_last:])):
#        print('No smoothing')
        data_smooth = data[-1,:]
        process_flag = 0
    else:
        x = ind[-n_win:]
        y = data[-n_win:,:]
        # fit polynomial
        p_coeff = np.polyfit(x[ind_ok],y[ind_ok,:],poly_order)
        # compute smoothed last value
        if poly_order==2:
            x_mat = [x[-1]**2,x[-1],1]
        elif poly_order==3:
            x_mat = [x[-1]**3,x[-1]**2,x[-1],1]
        elif poly_order==4:
            x_mat = [x[-1]**4,x[-1]**3,x[-1]**2,x[-1],1]
            
        data_smooth = np.dot(x_mat,p_coeff)
        process_flag = 1
        
    # return last value
    return data_smooth, process_flag
