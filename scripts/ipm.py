
# coding: utf-8

# In[1]:

import numpy as np
import cv2
# focal length
fu=0.0 ;
fv=0.0 ;

# optical center
center_u=0.0 ;
center_v=0.0 ;

# extrinsic parameters
pitch=0.0 ;
yaw=0.0 ;
# height of the camera in mm
h=0.0; 

# ROI (region of interest)
ROILeft=0 ;	
ROIRight=0 ;					
ROITop=0 ;
ROIBottom=0 ;

# ipm size 
ipm_width=0 ;
ipm_height=0 ;

# intermediate variables
# sin and cos use radians, not degrees
c1=0.0 ;
c2=0.0 ;
s1=0.0 ;
s2=0.0 ;

# distances (in the world frame) - to - pixels ratio
ratio_x=0, 
ratio_y=0;

# transformation of a point from image frame [u v] to world frame [x y]
def image2ground(uv):
    dummy_data=np.array([
        -c2/fu,     s1*s2/fv,   center_u*c2/fu-center_v*s1*s2/fv-c1*s2,
        s2/fu,      s1*c2/fv,   -center_u*s2/fu-center_v*s1*c2/fv-c1*c2,
        0,          c1/fv,      -center_v*c1/fv+s1,
        0,          -c1/h/fv,   center_v*c1/h/fv-s1/h
                         ])
    # static cv::Mat transformation_image2ground = cv::Mat(4, 3, CV_32F, dummy_data);
    # Mat object was needed because C/C++ lacked a standard/native implementation of matrices.
    # However, numpy's array is a perfect replacement for that functionality. Hence, the cv2 module accepts numpy.arrays wherever a matrix is indicated in the docs.
    transformation_image2ground=dummy_data.reshape((4,3))
    
    # Construct the image frame coordinates
    dummy_data2=[uv.x, uv.y, 1]
    image_coordinate=dummy_data2.reshape((3,1))
    
    # Find the world frame coordinates
    world_coordinate=transformation_image2ground*image_coordinate
    # Normalize the vector
    # the indexing of matrix elements starts from 0
    # world_coordinate.at<float>(3, 0);?
    world_coordinate=world_coordinate/(world_coordinate)
    return (world_coordinate[0.0],world_coordinate[1,0]);

# transformation of a point from world frame [x y] to image frame [u v]
def ground2image(xy):
    dummy_data=np.array([
        c2*fu+center_u*c1*s2,   center_u*c1*c2-s2*fu,   -center_u*s1,
        s2*(center_v*c1-fv*s1),  c2*(center_v*c1-fv*s1), -fv*c1-center_v*s1,
        c1*s2,                  c1*c2,                  -s1,
        c1*s2,                  c1*c2,                  -s1                         
                         ])

    # static cv::Mat transformation_ground2image = cv::Mat(4, 3, CV_32F, dummy_data);
    transformation_ground2image=dummy_data.reshape(4,3)
    
    # Construct the image frame coordinates
    dummy_data2=[xy.x, xy.y, -h]
    world_coordinate=dummy_data2.reshape((3,1))
    # Find the world frame coordinates
    image_coordinate=np.multiply(transformation_ground2image,world_coordinate)
    # Normalize the vector
    # the indexing of matrix elements starts from 0
    # image_coordinate = image_coordinate / image_coordinate.at<float>(3, 0);
#??
    image_coordinate=image_coordinate/image_coordinate[3,0]
    return (image_coordinate[0,0],image_coordinate[1,0])

# transformation of a point from ipm image frame [x' y'] to perspective image frame [u v]
# x_world = offset_x + u * ratio_x
# y_world = offset_y + (ipm_height - v) * ratio_y
def ipm2image(uv):
    x_world=offset_x+u*ratio_x
    y_world=offset_y+(ipm_height-v)*ratio_y
    return ground2image((x_world,y_world))


def getIPM(input,ipm_width,ipm_height):
    # Input Quadilateral or Image plane coordinates
    imageQuad=np.empty([4,2])
    # World plane coordinates
    groundQuad=np.empty([4,2])
    
    # Output Quadilateral
    ipmQuad=np.empty([4,2])
    
    # Lambda Matrix
    # cv::Mat lambda( 3, 3, CV_32FC1 );
    lambda_mat=np.empty([3,3])
    # The 4 points that select quadilateral on the input , from top-left in clockwise order
    # These four pts are the sides of the rect box used as input
    imageQuad[0]=(ROILeft,ROITop)
    imageQuad[1]=(ROIRight,ROITop)
    imageQuad[2]=(ROIRight,ROIBottom)
    imageQuad[3]=(ROILeft,ROIBottom)
    # The world coordinates of the 4 points
    for i in range(0,4):
        groundQuad[i]=image2ground(imageQuad[i])

    offset_x=groundQuad[0][0]
    offset_y=groundQuad[3][1]
    # float ground_width = (groundQuad[1][0]-groundQuad[0][0])   //top-right.x - top-left.x
    # float ground_length = (groundQuad[0][1]-groundQuad[4][1])  //top-left.y - bottom-left.y
    ratio_x=(groundQuad[1][0]-groundQuad[0][0])/ipm_width
    ratio_y=(groundQuad[0][1]-groundQuad[4][1])/ipm_width
    # Compute coordinates of the bottom two points in the ipm image frame
    x_bottom_left=(groundQuad[3][0]-groundQuad[0][0])/ratio_x
    x_bottom_right=(groundQuad[2][0]-groundQuad[0][0])/ratio_x
    
    # The 4 points where the mapping is to be done , from top-left in clockwise order
    ipmQuad[0]=(0,0)
    ipmQuad[1]=(ipm_width-1,0)
    ipmQuad[2]=(x_bottom_right,ipm_height-1)
    ipmQuad[3]=(x_bottom_left,ipm_height-1)

    # Get the Perspective Transform Matrix i.e. lambda
    lambda_mat=cv2.getPerspectiveTransform(imageQuad,ipmQuad)
    
    # Apply the Perspective Transform just found to the src image
    ipm=cv2.warpPerspective(input,lambda_mat,(ipm_width,ipm_height))
    return ipm




