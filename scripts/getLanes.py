import cv2
import cv
import numpy as np
import misc
import ipm

# // input image size 
image_width=0
image_height=0


# Hough transform
thres_num_points=0

# clustering of lines
thres_cluster_delta_angle=0
thres_cluster_delta_rho=0

# if two lanes are parallel and of certain distance, then left and right lanes are both detected. Pick the left one
thres_parallel_delta_angle=0
thres_parallel_delta_rho=0

# if two lanes are converging. Pick the right one
thres_converge_delta_angle=0
thres_converge_delta_rho=0

# method for edge detection
detect_method=0
dilation_white_size=0

# method for white pixel extraction
extract_method=0
dilation_element=0
dilation_edge_size=0


# /*  
#     This is the main function for lane detection. It takes an image as input and returns a vector of lines.
#         Each element in the returned vector contains rho and theta of the detected lane in the ground plane.
#         rho - the angle between the detected lane and the heading of the robot (i.e., the camera).
#         theta - the distance from the origin (bottom left of the ground plane) to the detected lane
#  */

def getLanes(input,isDebug):
    if input.size==0:
        print "Error: Input image is empty.Function getLanes(input) aborts."
        return clusters
    # Verify size of input images.
    rows=input.rows
    cols=input.cols
    if rows is not image_height and cols is not image_width:
        print "Warning: forced resizing of input images"
        size=(image_height,image_width)
        np.resize(input,size)
    
    # Get inverse projection mapping
    ipm_rgb=getIPM(input,ipm_width,ipm_height)
    
    # Edge detection
    detection_edge=edgeDetection(ipm_rgb,detect_method,False)
    dilated_edges=Dilation(detection_edge,dilation_element,dilation_edge_size,isDebug,"Dilated Edges")
    
    # Get white pixels
    white_pixels=extractWhitePixel(ipm_rgb,extract_method,False)
    
    # Dilation of the white pixels
    dilated_white_pixels=Dilation(white_pixels,dilation_element,dilation_white_size,isDebug,"Dilated White Pixels")
    
    # combine edge detection and white pixel extraction
    lane_boundaries=cv2.bitwise_and(dilated_white_pixels,dilated_edges)
    if isDebug:
        cv2.imshow("Bitwise and",lane_boundaries)
        
    # HoughLines: First parameter, Input image should be a binary image, so apply threshold or use canny edge detection before finding applying hough transform. Second and third parameters are \rho and \theta accuracies respectively. Fourth argument is the threshold, which means minimum vote it should get for it to be considered as a line.
    lines=cv2.HoughLines(lane_boundaries,1,np.pi/180,thres_num_points)
    
    # Result cleanning: make sure the distance rho is always positive.    
    # rho_theta_pairs are list of [rho,theta] generated from the picture
    rho_theta_pairs=lines[0]
    for i in range(0,len(rho_theta_pairs)):
    # if rho in the ith [rho,theta] pairs is smaller than 0
        if rho_theta_pairs[i][0]<0:
            lines[i][0]=-lines[i][0]
            lines[i][1]=np.pi+lines[i][1]
        # ?? what does wrapTheta means
        # lines[i][1] = wrapTheta(lines[i][1]);
    
    # Show results before clustering
    if False:
        #?? What does clone mean?
        # cv::Mat ipm_duplicate = ipm_rgb.clone();
        ipm_duplicate=ipm_rgb
        for i in range(0,len(rho_theta_pairs)):
            rho=rho_theta_pairs[i][0]
            theta=rho_theta_pairs[i][1]
            a=np.cos(theta)
            b=np.sin(theta)
            x0=a*rho
            y0=b*rho
            
            pt1=(cv.Round(x0+1000*(-b)),cv.Round(y0+1000*(a)))
            pt2=(cv.Round(x0-1000*(-b)),cv.Round(y0-1000*(a)))
            ipm_duplicate=cv2.line(pt1,pt2,(0,255,0),3)
            print len(lines[0])
    #     // cluster lines into groups and take averages, in order to remove duplicate segments of the same line
    #     // TODO: need a robust way of distinguishing the left and right lanes
    num_of_lines=0
    for i in range(0,len(rho_theta_pairs)):
        rho=rho_theta_pairs[i][0]
        theta=rho_theta_pairs[i][1]
        if isDebug:
            print "Now it's debugging"
        a=np.cos(theta)
        b=np.sin(theta)
        custer_found=False
        
        # Match this line with existing clusters
        for j in range(0,len(clusters)):
            avg_line=clusters[j]/num_of_lines[j]
            avg_rho=avg_line[0]
            avg_theta=avg_line[1]
            # clustered in the same cluster if it is close to the cluster average
            if abs(rho-avg_rho)<thres_cluster_delta_rho and abs(theta-avg_theta)/np.pi*180<thres_cluster_delta_angle:
                clusters[j]+=lines[i]
                num_of_lines[j]+=1
                clusters_found=True
                break
        # Create a new cluster if it doesn't match with any existing clusters
        if cluster_found:
            pass
        else:
            #?? not sure how does clusters look like and how push_back applied to clusters
            # clusters.push_back(lines[i])
            # num_of_lines.push_back(1);
            clusters=lines[i]
            num_of_lines=1
    # Take averages of each cluster and show results after clustering
    for i in range(0,len(clusters)):
        clusters[i]=clusters[i]/num_of_lines[i]
        rho=clusters[i][0]
        theta=clusters[i][1]
        a=np.cos(theta)
        b=np.sin(theta)
        x0=a*rho
        y0=b*rho
        pt1=(cv.Round(x0+1000*(-b)),cv.Round(y0+1000*(a)))
        pt2=(cv.Round(x0-1000*(-b)),cv.Round(y0-1000*(a)))
        ipm_rgb=cv2.line(pt1,pt2(0,255,0),3)
        
    if isDebug:
        cv2.imshow("Hough Line Transform After Clustering",ipm_rgb)
        print len(clusters),"clusters found."
    # // TODO: verify clusters by taking samples on the line and check color
    return clusters

