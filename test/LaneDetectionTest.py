# ipm.py
import numpy as np
import cv2
import cv
# # focal length
# fu = 0.0
# fv = 0.0
#
# # optical center
# center_u = 0.0
# center_v = 0.0
#
# # extrinsic parameters
# pitch = 0.0
# yaw = 0.0
# # height of the camera in mm
# h = 0.0
#
# # ROI (region of interest)
# ROILeft = 0
# ROIRight = 0
# ROITop = 0
# ROIBottom = 0
#
# # ipm size
# ipm_width = 0
# ipm_height = 0
#
# # intermediate variables
# # sin and cos use radians, not degrees
# c1 = 0.0
# c2 = 0.0
# s1 = 0.0
# s2 = 0.0
#
# # distances (in the world frame) - to - pixels ratio
# ratio_x = 0
# ratio_y = 0



# focal length
fu = 750.0
fv = 720.0

# optical center
center_u = 658
center_v = 372

# extrinsic parameters
pitch = 9.0
yaw = 0
# height of the camera in mm
h = 790

# ROI (region of interest)
ROILeft = 0
ROIRight = 1100
ROITop = 500
ROIBottom = 719

# ipm size
ipm_width = 600
ipm_height = 800

# intermediate variables
# sin and cos use radians, not degrees
c1 = 1.0
c2 = 1.0
s1 = 1.0
s2 = 1.0

# distances (in the world frame) - to - pixels ratio
ratio_x = 10
ratio_y = 10
# transformation of a point from image frame [u v] to world frame [x y]
offset_x=0.0
offset_y=0.0

def image2ground(uv):
    dummy_data = np.array([
        -c2 / fu,     s1 * s2 / fv,   center_u * c2 /
        fu - center_v * s1 * s2 / fv - c1 * s2,
        s2 / fu,      s1 * c2 / fv,   -center_u *
        s2 / fu - center_v * s1 * c2 / fv - c1 * c2,
        0,          c1 / fv,      -center_v * c1 / fv + s1,
        0,          -c1 / h / fv,   center_v * c1 / h / fv - s1 / h
    ])
    # static cv::Mat transformation_image2ground = cv::Mat(4, 3, CV_32F, dummy_data);
    # Mat object was needed because C/C++ lacked a standard/native implementation of matrices.
    # However, numpy's array is a perfect replacement for that functionality.
    # Hence, the cv2 module accepts numpy.arrays wherever a matrix is
    # indicated in the docs.
    transformation_image2ground = dummy_data.reshape((4, 3))
    transformation_image2ground=np.asmatrix(transformation_image2ground)

    # Construct the image frame coordinates
    # dummy_data2 = [uv.x, uv.y, 1]
    image_coordinate=np.matrix([[uv[0]],[uv[1]],[1]])

    # Find the world frame coordinates
    world_coordinate = transformation_image2ground * image_coordinate
    # Normalize the vector
    # the indexing of matrix elements starts from 0
    #?? world_coordinate.at<float>(3, 0);
    # print(world_coordinate)
    world_coordinate = world_coordinate / (world_coordinate[3][0])
    return (world_coordinate[0][0], world_coordinate[1][0])


# transformation of a point from world frame [x y] to image frame [u v]
def ground2image(xy):
    dummy_data = np.array([
        c2 * fu + center_u * c1 * s2,   center_u * c1 * c2 - s2 * fu,   -center_u * s1,
        s2 * (center_v * c1 - fv * s1),  c2 *
        (center_v * c1 - fv * s1), -fv * c1 - center_v * s1,
        c1 * s2,                  c1 * c2,                  -s1,
        c1 * s2,                  c1 * c2,                  -s1
    ])

    transformation_ground2image = dummy_data.reshape(4, 3)

    # Construct the image frame coordinates
    dummy_data2 = [xy.x, xy.y, -h]
    world_coordinate = dummy_data2.reshape((3, 1))
    # Find the world frame coordinates
    image_coordinate = np.multiply(
        transformation_ground2image, world_coordinate)
    # Normalize the vector
    # the indexing of matrix elements starts from 0
    image_coordinate = image_coordinate / image_coordinate[3, 0]
    return (image_coordinate[0, 0], image_coordinate[0, 1])


def ipm2image(uv):
    x_world = offset_x + u * ratio_x
    y_world = offset_y + (ipm_height - v) * ratio_y
    return ground2image((x_world, y_world))


def getIPM(input, ipm_width, ipm_height):
    # Input Quadilateral or Image plane coordinates
    imageQuad = np.empty([4, 2])
    # World plane coordinates
    groundQuad = np.empty([4, 2])

    # Output Quadilateral
    ipmQuad = np.empty([4, 2])

    # Lambda Matrix
    lambda_mat = np.empty([3, 3])
    # The 4 points that select quadilateral on the input , from top-left in clockwise order
    # These four pts are the sides of the rect box used as input
    imageQuad=np.array([[ROILeft,ROITop],[ROIRight,ROITop],[ROIRight,ROIBottom],[ROILeft,ROIBottom]],np.float32)

    # The world coordinates of the 4 points
    for i in range(0, 4):
        groundQuad[i] = image2ground(imageQuad[i])

    offset_x = groundQuad[0][0]
    offset_y = groundQuad[3][1]

    # float ground_width = (groundQuad[1][0]-groundQuad[0][0])   //top-right.x - top-left.x
    # float ground_length = (groundQuad[0][1]-groundQuad[4][1])  //top-left.y - bottom-left.y
    ratio_x = (groundQuad[1][0] - groundQuad[0][0]) / ipm_width
    ratio_y = (groundQuad[0][1] - groundQuad[3][1]) / ipm_height

    # Compute coordinates of the bottom two points in the ipm image frame
    x_bottom_left = (groundQuad[3][0] - groundQuad[0][0]) / ratio_x
    x_bottom_right = (groundQuad[2][0] - groundQuad[0][0]) / ratio_y

    # The 4 points where the mapping is to be done , from top-left in
    # clockwise order
    ipmQuad=np.array([[0,0],[ipm_width-1,0],[x_bottom_right,ipm_height-1],[x_bottom_left,ipm_height-1]],np.float32)

    # Get the Perspective Transform Matrix i.e. lambda
    lambda_mat = cv2.getPerspectiveTransform(imageQuad, ipmQuad)

    # Apply the Perspective Transform just found to the src image
    ipm = cv2.warpPerspective(input, lambda_mat, (ipm_width, ipm_height))
    return ipm


# misc.py
# // parameters for white pixel extraction
hueMinValue = 0
hueMaxValue = 255
satMinValue = 0
satMaxValue = 15
volMinValue = 240
volMaxValue = 255
lightMinValue = 190
lightMaxValue = 255

# // extraction of white pixels
thres_white_init = 0.5
thres_exposure_max = 1500
thres_exposure_min = 1200


# // This function takes an angle in the range [-3*pi, 3*pi] and
# // wraps it to the range [-pi, pi].
def wrapTheta(theta):
    if theta > np.pi:
        return theta - 2 * np.pi
    elif theta < -np.pi:
        return theta + 2 * np.pi
    return theta

# // Construct a new image using only one single channel of the input image
# // if color_image is set to 1, create a color image; otherwise a single-channel image is returned.
# // 0 - B; 1 - G; 2 - R
def getSingleChannel(input, channel, color_image):
    spl = cv2.split(input)
    if color_image == 0:
        return spl[channel]
#     emptyMat = thresholded
    channels = np.empty(input.shape)
# Only show color blue channel
    if channel == 0:
        input[:, :, 1] = 0
        input[:, :, 2] = 0
        channels = input
    elif channel == 1:
        # Only show color green channel
        input[:, :, 0] = 0
        input[:, :, 2] = 0
        channels = input
    else:
        # Only show colorred channel
        input[:, :, 0] = 0
        input[:, :, 1] = 0
        channels = input
    print "channels:", channels.shape
    output = channels
    return output


# // Show different channels of an image
def showChannels(input):
    #     cv2.imshow("B",getSingleChannel(input,0,True)) #b
    #     cv2.imshow("G",getSingleChannel(input,1,True)) #g
    #     cv2.imshow("R",getSingleChannel(input,2,True)) #r
    #     greyMat = thresholded
    #     greyMat=cv2.cvtColor(input,cv2.COLOR_BGR2GRAY)
    #     cv2.imshow("GREY",greyMat) #grey-scale
    print "Hello"


def edgeDetection(rgb_frame, detect_method, debug_mode):
    singleChannel = getSingleChannel(rgb_frame, 0, False)
    #   // First apply Gaussian filtering
    blurred_ipm = cv2.GaussianBlur(singleChannel, (5, 5), 0)
    cv2.imshow("blurred_ipm:",blurred_ipm)

    if debug_mode:
        cv2.imshow('Blurred ipm:', blurred_ipm)
    #   // Edge detection
    #   // adaptive thresholding outperforms canny and other filtering methods
    max_value = 255
    adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    thresholdType = cv2.THRESH_BINARY_INV
    blockSize = 11
    C = 5
    detection_edge=cv2.adaptiveThreshold(blurred_ipm,max_value,adaptiveMethod,thresholdType,blockSize,C)
    # detection_edge = np.zeros(blurred_ipm.shape, np.uint8)
    # detection_edge = cv2.adaptiveThreshold(
    # blurred_ipm, max_value, adaptiveMethod, thresholdType, blockSize, C)
    if debug_mode:
        cv2.imshow('Detection edge:', detection_edge)
    return detection_edge


def testShowChannels():
    img = cv2.imread(
        '/Users/g41903/Desktop/MIT/Media Lab/LaneDetection/LaneView.jpg', cv2.CV_8UC1)
    edgeDetection(img, 0, 0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # showChannels(img)
    # cv2.waitKey(0)
    # cv2.imshow("image",img)
# testShowChannels()


# // Extract white pixels from a RGB image#
def extractWhitePixel(rgb_frame,extract_method,debug_mode):
    if extract_method=='HSV':
        # Convert BGR to HSV
        hsv_frame=cv2.cvtColor(rgb_frame,cv2.COLOR_BGR2HSV)
        # define range of color in HSV
        min_color=np.array([hueMinValue,satMinValue,volMinValue])
        max_color=np.array([hueMaxValue,satMaxValue,volMaxValue])
        # Threshold the HSV image to get only specific color
        threshold_frame = cv2.inRange(hsv_frame, min_color, max_color)
        return  threshold_frame
        # cv2.imshow('frame',rgb_frame)
        # cv2.imshow('mask',mask)
        # cv2.imshow('res',res)
    elif extract_method=='HLS':
        # Convert BGR to HLS
        hls_frame=cv2.cvtColor(rgb_frame,cv2.COLOR_BGR2HLS)
        # define range of color in HSV
        min_color=np.array([hueMinValue,satMinValue,volMinValue])
        max_color=np.array([hueMaxValue,satMaxValue,volMaxValue])
        # Threshold the HSV image to get only specific color
        mask = cv2.inRange(hls_frame, min_color, max_color)
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(rgb_frame,rgb_frame, mask= mask)
        cv2.imshow('frame',rgb_frame)
        cv2.imshow('mask',mask)
        cv2.imshow('res',res)
        return res
    elif extract_method=='ADAPTIVE':
        # Get a single channel
        singleChannel=getSingleChannel(rgb_frame,0,False)
        # Extraction of white pixels
        minVal,maxVal,minLoc,maxLoc=cv2.minMaxLoc(singleChannel)
        # min,max=minMaxLoc(singleChannel)
        thresholded=np.zeros(singleChannel.shape, np.uint8)
        #adaptive thresholding
        maxValue=255
        thres_count=0
        thres_adaptive=thres_white_init
        thres_upper_bound=1
        thres_lower_bound=0
        type=cv2.THRESH_BINARY
        while thres_count<10:
            thres_count+=1
            thresh=min+(max-min)*thres_adaptive
            thresholded=cv2.threshold(singleChannel,thresh,maxValue,type)
            s=np.sum(singleChannel,axis=0)/255
            if s>thres_exposure_max:
                thres_lower_bound=thres_adaptive
                thres_adaptive=(thres_upper_bound+thres_lower_bound)/2
            elif s<thres_exposure_min:
                thres_upper_bound=thres_adaptive
                thres_adaptive=(thres_upper_bound+thres_lower_bound)/2
            else:
                break
        return thresholded

def testExtractWhitePixel():
#     hueMinValue=110
#     satMinValue=50
#     volMinValue=50
#     hueMaxValue=130
#     satMaxValue=255
#     volMaxValue=255
    cap = cv2.VideoCapture(0)
    while(1):
        # Take each frame
        _, frame = cap.read()

        extractWhitePixel(frame,'HSV',True)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
# testExtractWhitePixel()


# /** @function Dilation */
def Dilation(src,dilation_elem,dilation_size,debug_mode,title):
    dilation_type=0
    if dilation_elem==0:
        dilation_type=cv2.MORPH_RECT
    elif dilation_elem==1:
        dilation_type=cv2.MORPH_CROSS
    elif dilation_elem==2:
        dilation_type=cv2.MORPH_ELLIPSE
    element=cv2.getStructuringElement(dilation_type,(int(2*dilation_size+1),int(2*dilation_size+1)))
    dilation_dst=0
    dilation_dst=cv2.dilate(src,element,iterations=1)
    if debug_mode:
        cv2.imshow(title,dilation_dst)
    return dilation_dst

# // TODO: curve fitting
# void fitCurve(cv::Mat lane_boundaries, cv::Mat ipm_rgb)
# {
#   std::vector<Point> filtered_points;
#   int count = 0;
#   for (int y = 0; y < lane_boundaries.rows; y++)
#   {
#       for (int x = 0; x < lane_boundaries.cols; x++)
#       {
#           // cout << lane_boundaries.at<uchar>(y, x) << " ";
#           // In OpenCV, x and y are inverted when trying to access an element
#           if (lane_boundaries.at<uchar>(y, x) == 255)
#           {
#               Point pt(x, y);
#               filtered_points.push_back(pt);
#           }
#       }
#   }


# def fitCurve(lane_boundaries,ipm_rgb):
#     for y in range(0,lane_boundaries.rows):
#         for x in range(0,lane_boundaries.cols):
#             if point


#   std::vector<double> coefficients = polyFit(filtered_points);
#   double c1 = coefficients[0];
#   double c2 = coefficients[1];
#   double c3 = coefficients[2];
#   cout << "c1 = " << c1 << "\tc2 = " << c2 << "\tc3 = " << c3 << endl;

#   // cout << "filtered_points.size() = " << filtered_points.size() << "\tapproxCurve.size() = " << approxCurve.size() << endl;

#   std::vector<Point> testCurve;
#   for (int x=0; x<lane_boundaries.cols; x++)
#   {
#       int appro_y = c1 + c2 * x + c3 * pow(x, 2);
#       Point pt(x, appro_y);
#       // cout << "appro_y = " << appro_y << endl;
#       testCurve.push_back(pt);
#   }

#   Scalar color = Scalar( 255, 0, 0 );
#   polylines(ipm_rgb, testCurve, false, color);
#   imshow("Curve detection", ipm_rgb);

#     // polylines(ipm_rgb, filtered_points, false, color);
#   // imshow("check input to curve detection", ipm_rgb);
#   imshow("lane_boundaries", lane_boundaries);

# }






# LaneDetectTest.py

# // input image size
image_width = 1280
image_height = 720

# // Hough transform
thres_num_points = 200

# // clustering of lines
thres_cluster_delta_angle = 10
thres_cluster_delta_rho = 20

# // if two lanes are parallel and of certain distance, then left and right lanes are both detected. Pick the left one
thres_parallel_delta_angle = 3
thres_parallel_delta_rho =150

# // if two lanes are converging. Pick the right one
thres_converge_delta_angle = 10
thres_converge_delta_rho = 60

# // method for edge detection
detect_method = 10
dilation_white_size = 3

# // method for white pixel extraction
extract_method = 'HSV'
dilation_element = 0.5
dilation_edge_size = 0.5


# /*
#     This is the main function for lane detection. It takes an image as input and returns a vector of lines.
#         Each element in the returned vector contains rho and theta of the detected lane in the ground plane.
#         rho - the angle between the detected lane and the heading of the robot (i.e., the camera).
#         theta - the distance from the origin (bottom left of the ground plane) to the detected lane
#  */

def getLanes(input, isDebug):
    clusters=np.empty(shape=(2,2),dtype=float)
    if input.size == 0:
        print "Error: Input image is empty.Function getLanes(input) aborts."
        return clusters
    # Verify size of input images.
    rows = input.shape[0]
    cols = input.shape[1]
    if rows is not image_height and cols is not image_width:
        print "Warning: forced resizing of input images"
        size = (image_height, image_width)
        np.resize(input, size)

    # Get inverse projection mapping
    ipm_rgb = getIPM(input, ipm_width, ipm_height)

    # Edge detection
    detection_edge = edgeDetection(ipm_rgb, detect_method, False)
    cv2.imshow('Final Detection Edge: ',detection_edge)

    dilated_edges=Dilation(detection_edge,dilation_element,dilation_edge_size,isDebug,"Dilated Edges")
    cv2.imshow('Final dilated edges: ', dilated_edges)


    # Get white pixels
    white_pixels = extractWhitePixel(ipm_rgb, extract_method, False)
    cv2.imshow('Final white pixels: ',white_pixels)


    # Dilation of the white pixels
    dilated_white_pixels = Dilation(
        white_pixels, dilation_element, dilation_white_size, isDebug, "Dilated White Pixels")
    cv2.imshow('Final dilated white pixels: ', dilated_white_pixels)
    # dilated_white_pixels=0
    # cv2.dilate(white_pixels, dilated_white_pixels, dilation_white_size, isDebug, "Dilated White Pixels")

    # combine edge detection and white pixel extraction
    lane_boundaries = cv2.bitwise_and(dilated_white_pixels, dilated_edges)
    cv2.imshow('Final lane_boundaries: ', lane_boundaries)
    if isDebug:
        cv2.imshow("Bitwise and", lane_boundaries)

    # HoughLines: First parameter, Input image should be a binary image, so
    # apply threshold or use canny edge detection before finding applying
    # hough transform. Second and third parameters are \rho and \theta
    # accuracies respectively. Fourth argument is the threshold, which means
    # minimum vote it should get for it to be considered as a line.
    lines = cv2.HoughLines(lane_boundaries, 1, np.pi / 180, thres_num_points)

    # Result cleanning: make sure the distance rho is always positive.
    # rho_theta_pairs are list of [rho,theta] generated from the picture
    rho_theta_pairs = lines[0]
    for i in range(0, len(rho_theta_pairs)):
        # if rho in the ith [rho,theta] pairs is smaller than 0
        if rho_theta_pairs[i][0] < 0:
            rho_theta_pairs[i][0] = -rho_theta_pairs[i][0]
            rho_theta_pairs[i][1] = np.pi + rho_theta_pairs[i][1]
        # ?? what does wrapTheta means
        # In case theta is over pi or less then -pi: If the theta is over pi, then it will be deducted by 2pi, if it's less then -pi, it will add up 2pi
        rho_theta_pairs[i][1] = wrapTheta(rho_theta_pairs[i][1]);

    # Show results before clustering
    if True:
        ipm_duplicate = ipm_rgb
        for i in range(0, len(rho_theta_pairs)):
            rho = rho_theta_pairs[i][0]
            theta = rho_theta_pairs[i][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (cv.Round(x0 + 1000 * (-b)), cv.Round(y0 + 1000 * (a)))
            pt2 = (cv.Round(x0 - 1000 * (-b)), cv.Round(y0 - 1000 * (a)))
            img = np.zeros((1280,760,3), np.uint8)
            cv2.line(img,pt1, pt2, (0, 255, 0), 3)
            cv2.imshow('Show Line:',img)
            # print len(lines[0])

    #     // cluster lines into groups and take averages, in order to remove duplicate segments of the same line
    #     // TODO: need a robust way of distinguishing the left and right lanes
    num_of_lines = 0
    # for i in range(0, len(rho_theta_pairs)):
    #     rho = rho_theta_pairs[i][0]
    #     theta = rho_theta_pairs[i][1]
    #     if isDebug:
    #         print "Now it's debugging"
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     custer_found = False
    #
    #     # Match this line with existing clusters
    #     for j in range(0, len(clusters)):
    #         avg_line = clusters[j] / num_of_lines[j]
    #         avg_rho = avg_line[0]
    #         avg_theta = avg_line[1]
    #
    #         if abs(rho - avg_rho) < thres_cluster_delta_rho and abs(theta - avg_theta) / np.pi * 180 < thres_cluster_delta_angle:
    #             clusters[j] += lines[i]
    #             num_of_lines[j] += 1
    #             clusters_found = True
    #             break
    #     if cluster_found:
    #         pass
    #     else:
    #         #?? not sure how does clusters look like and how push_back applied to clusters
    #         # clusters.push_back(lines[i])
    #         # num_of_lines.push_back(1);
    #         clusters = lines[i]
    #         num_of_lines = 1
    # for i in range(0, len(clusters)):
    #     clusters[i] = clusters[i] / num_of_lines[i]
    #     rho = clusters[i][0]
    #     theta = clusters[i][1]
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a * rho
    #     y0 = b * rho
    #     pt1 = (cv.Round(x0 + 1000 * (-b)), cv.Round(y0 + 1000 * (a)))
    #     pt2 = (cv.Round(x0 - 1000 * (-b)), cv.Round(y0 - 1000 * (a)))
    #     ipm_rgb = cv2.line(pt1, pt2(0, 255, 0), 3)
    #
    # if isDebug:
    #     cv2.imshow("Hough Line Transform After Clustering", ipm_rgb)
    #     print len(clusters), "clusters found."
    # return clusters



def testShowChannels():
    img = cv2.imread(
        './LaneView3.jpg', 1)
    edgeDetection(img, 0, 0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    input = cv2.imread('../data/LaneView3.jpg', 1)
    cv2.imshow('Origin input:', input)
    getLanes(input, True)
    print("Finished")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # testShowChannels()
