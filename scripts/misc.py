
# coding: utf-8

# In[5]:

import cv2
import numpy as np

# // parameters for white pixel extraction
hueMinValue=0 
hueMaxValue=0
satMinValue=0
satMaxValue=0
volMinValue=0
volMaxValue=0
lightMinValue=0 
lightMaxValue=0

# // extraction of white pixels
thres_white_init=0.0
thres_exposure_max=0
thres_exposure_min=0


# // This function takes an angle in the range [-3*pi, 3*pi] and
# // wraps it to the range [-pi, pi].
def wrapTheta(theta):
    if theta > CV_PI:
        return theta - 2 * CV_PI
    elif theta < -CV_PI:
        return theta + 2 * CV_PI
    return theta

# // Construct a new image using only one single channel of the input image
# // if color_image is set to 1, create a color image; otherwise a single-channel image is returned.
# // 0 - B; 1 - G; 2 - R

def getSingleChannel(input,channel,color_image):
    spl=cv2.split(input)
    if color_image==0:
        return spl[channel]
    emptyMat = thresholded
    channels=np.empty(input.shape)
# Only show color blue channel
    if channel==0:
        input[:,:,1]=0
        input[:,:,2]=0
        channels=input
    elif channel==1:
# Only show color green channel
        input[:,:,0]=0
        input[:,:,2]=0
        channels=input
    else:
# Only show colorred channel
        input[:,:,0]=0
        input[:,:,1]=0
        channels=input
    print "channels:",channels.shape
    output=channels
    return output


# // Show different channels of an image
def showChannels(input):
#     cv2.imshow("B",getSingleChannel(input,0,True)) #b
#     cv2.imshow("G",getSingleChannel(input,1,True)) #g
#     cv2.imshow("R",getSingleChannel(input,2,True)) #r
    greyMat = thresholded
    greyMat=cv2.cvtColor(input,cv2.COLOR_BGR2GRAY)
#     cv2.imshow("GREY",greyMat) #grey-scale
    
    
def testShowChannels():
    img = cv2.imread('/Users/g41903/Desktop/MIT/Media Lab/LaneDetection/LaneView.jpg',1)
    edgeDetection(img,0,0)
#     showChannels(img)
#     cv2.waitKey(0)
#     cv2.imshow("image",img)


def edgeDetection(rgb_frame,detect_method,debug_mode):
    # 	// Get a single channel
    singleChannel=getSingleChannel(rgb_frame,0,False)
    # 	// First apply Gaussian filtering
    blurred_ipm=cv2.GaussianBlur(singleChannel,(5,5),0)
    if debug_mode:
        cv2.imshow('Blurred ipm:',blurred_ipm)
    # 	// Edge detection
    #   // adaptive thresholding outperforms canny and other filtering methods
    max_value=255
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    thresholdType=cv2.THRESH_BINARY_INV
    blockSize=11
    C=5
    detection_edge = np.zeros(blurred_ipm.shape, np.uint8)
    
    detection_edge=cv2.adaptiveThreshold(blurred_ipm,max_value,adaptiveMethod,thresholdType,blockSize,C)
    if debug_mode:
        cv2.imshow('Detection edge:',detection_edge)        
    return detection_edge

testShowChannels()



# In[4]:

import cv2
import numpy as np
hueMinValue=110
satMinValue=50
volMinValue=50
hueMaxValue=130
satMaxValue=255
volMaxValue=255
# // Extract white pixels from a RGB image
# cv::Mat extractWhitePixel(cv::Mat rgb_frame, int extract_method, bool debug_mode)
# {
# 	// convert from RGB to HSV representation
# 	if (extract_method == HSV)
# 	{
# 		cv::Mat hsv_frame;
# 		cv::cvtColor(rgb_frame, hsv_frame, CV_RGB2HSV);

# 		cv::Scalar min(hueMinValue, satMinValue, volMinValue);
# 		cv::Scalar max(hueMaxValue, satMaxValue, volMaxValue);
# 		cv::Mat threshold_frame;
# 		cv::inRange( hsv_frame, min, max, threshold_frame);
# 		return threshold_frame;
# 	}

def extractWhitePixel(rgb_frame,extract_method,debug_mode):
    if extract_method=='HSV':
        # Convert BGR to HSV
        hsv_frame=cv2.cvtColor(rgb_frame,cv2.COLOR_BGR2HSV)
        # define range of color in HSV
        min_color=np.array([hueMinValue,satMinValue,volMinValue])
        max_color=np.array([hueMaxValue,satMaxValue,volMaxValue])
        # Threshold the HSV image to get only specific color
        mask = cv2.inRange(hsv_frame, min_color, max_color)
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(rgb_frame,rgb_frame, mask= mask)
        cv2.imshow('frame',rgb_frame)
        cv2.imshow('mask',mask)
        cv2.imshow('res',res)
# 	else if (extract_method == HLS)
# 	{	
# 		cv::Mat hls_frame;
# 		cv::cvtColor(rgb_frame, hls_frame, CV_RGB2HLS);

# 		cv::Scalar min(hueMinValue, lightMinValue, satMinValue);
# 		cv::Scalar max(hueMaxValue, lightMaxValue, satMaxValue);
# 		cv::Mat threshold_frame;
# 		cv::inRange( hls_frame, min, max, threshold_frame);
# 		return threshold_frame;
# 	}
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
    elif extract_method=='ADAPTIVE':
        # Get a single channel
        singleChannel=getSingleChannel(rgb_frame,0,False)
        # Extraction of white pixels
        min,max=minMaxLoc(singleChannel)
        thresholded=np.zeros(singleChannel.shape, np.uint8)
        #adaptive thresholding
        maxValue=255
        thres_count=0
        thres_adaptive=thres_white_init
        thres_upper_bound=1
        thres_lower_bound=0
        type=THRESH_BINARY
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
            
            
            


# 	else if (extract_method == ADAPTIVE)
# 	{
# 		// Get a single channel
# 		static cv::Mat singleChannel;
# 		singleChannel = getSingleChannel(rgb_frame, 0, false);

# 		// Extraction of white pixels
# 		double min, max;
# 		static cv::Mat thresholded;
# 		cv::minMaxLoc(singleChannel, &min, &max);

# 		// adaptive thresholding
# 		int maxValue = 255;
# 		int thres_count = 0;
# 		float thres_adaptive = thres_white_init;
# 		float thres_upper_bound = 1;
# 		float thres_lower_bound = 0;
# 		//int type = cv::THRESH_TOZERO;
# 		int type = cv::THRESH_BINARY;

# 		// Binary search for the proper exposure threshold
# 		while (thres_count < 10)
# 		{
# 			thres_count ++;	

# 			int thresh = min + (max - min) * thres_adaptive;
# 			threshold( singleChannel, thresholded, thresh, maxValue, type);


# cv::sum returns a cv::Scalar element. If you have a 3-channel image for example, the return value has 3 elements, one for each channel. So each channel is summed up independently. [0] would access the first value of that Scalar. For RGB images with BGR ordering (like mostly used in OpenCV), [0] of the Scalar would access the summed up "blue channel", '[1]' would be the sum of the "green channel" and '[2]' is the sum of the "red channel" in that example. docs.opencv.org/modules/core/doc/operations_on_arrays.html#sum – Micka Feb 19 '14 at 9:04 
# 			// Deal with over-exposure
# 			double s = cv::sum( thresholded )[0] / 255;
# 			// cout << "Exposure: " << s << endl;
		
# 			if (s > thres_exposure_max)
# 			{
# 				// cout << "Over-exposed. s = " << s << "\tthres_adaptive = " << thres_adaptive << endl;
# 				thres_lower_bound = thres_adaptive;
# 				thres_adaptive = (thres_upper_bound + thres_lower_bound) / 2;
# 			}
# 			else if (s < thres_exposure_min)
# 			{
# 				// cout << "Under-exposed. s = " << s << "\tthres_adaptive = " << thres_adaptive << endl;
# 				thres_upper_bound = thres_adaptive;
# 				thres_adaptive = (thres_upper_bound + thres_lower_bound) / 2;
# 			}
# 			else
# 			{
# 				// cout << "Proper-exposed. s = " << s << "\tthres_adaptive = " << thres_adaptive << endl;
# 				break;
# 			}
# 		}
# 		return thresholded;
# 	}
# }



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
testExtractWhitePixel()



# In[ ]:



