import cv2 # computer vision library
import helpers # helper functions

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # for loading in images

%matplotlib inline

# Image data directories
IMAGE_DIR_TRAINING = "traffic_light_images/training/"
IMAGE_DIR_TEST = "traffic_light_images/test/"


# Using the load_dataset function in helpers.py
# Load training data
IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TRAINING)

## TODO: Write code to display an image in IMAGE_LIST (try finding a yellow traffic light!)
## TODO: Print out 1. The shape of the image and 2. The image's label

# The first image in IMAGE_LIST is displayed below (without information about shape or label)
selected_image = IMAGE_LIST[250][0]
plt.imshow(selected_image)

# This function should take in an RGB image and return a new, standardized version
def standardize_input(image):
    
    ## TODO: Resize image and pre-process so that all "standard" images are the same size --------------------- 
    standard_im = np.copy(image)
    standard_im = cv2.resize(standard_im,(32,32))
    
    
    return standard_im
    
selected_image = standardize_input(IMAGE_LIST[34][0])
plt.imshow(selected_image)

## TODO: One hot encode an image label
## Given a label - "red", "green", or "yellow" - return a one-hot encoded label

# Examples: 
# one_hot_encode("red") should return: [1, 0, 0]
# one_hot_encode("yellow") should return: [0, 1, 0]
# one_hot_encode("green") should return: [0, 0, 1]

def one_hot_encode(label):
    
    ## TODO: Create a one-hot encoded label that works for all classes of traffic lights
    if label == "red":
        one_hot_encoded = [1,0,0]
    elif label == "yellow":
        one_hot_encoded = [0,1,0]
    elif label == "green":
        one_hot_encoded = [0,0,1]
    else:
        raise TypeError("The label has to be either string named 'red', 'yellow' or 'green'.")
    
    return one_hot_encoded

# Importing the tests
import test_functions
tests = test_functions.Tests()

# Test for one_hot_encode function
tests.test_one_hot(one_hot_encode)

def standardize(image_list):
    
    # Empty image data array
    standard_list = []

    # Iterate through all the image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]

        # Standardize the image
        standardized_im = standardize_input(image)

        # One-hot encode the label
        one_hot_label = one_hot_encode(label)    

        # Append the image, and it's one hot encoded label to the full, processed list of image data 
        standard_list.append((standardized_im, one_hot_label))
        
    return standard_list


# Standardize all training images
STANDARDIZED_LIST = standardize(IMAGE_LIST)
print(STANDARDIZED_LIST)

def show_image(number=0):
    
    plt.imshow(STANDARDIZED_LIST[number][0])
    print('Standardized image number : {}'.format(number))
    print('Label :{}'.format(STANDARDIZED_LIST[number][1]))

show_image(980)

# Convert and image to HSV colorspace
# Visualize the individual color channels

image_num = 34
test_im = STANDARDIZED_LIST[image_num][0]
test_label = STANDARDIZED_LIST[image_num][1]

# Convert to HSV
hsv = cv2.cvtColor(test_im, cv2.COLOR_RGB2HSV)

# Print image label
print('Label [red, yellow, green]: ' + str(test_label))

# HSV channels
h = hsv[:,:,0]
s = hsv[:,:,1]
v = hsv[:,:,2]

# Plot the original image and the three channels
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,10))
ax1.set_title('Standardized image')
ax1.imshow(test_im)
ax2.set_title('H channel')
ax2.imshow(h, cmap='gray')
ax3.set_title('S channel')
ax3.imshow(s, cmap='gray')
ax4.set_title('V channel')
ax4.imshow(v, cmap='gray')

## TODO: Create a brightness feature that takes in an RGB image and outputs a feature vector and/or value
## This feature should use HSV colorspace values
def create_feature(rgb_image,slicing = 2):
    
    """
    goal : this function creates masked image out of available RGB_image
    
    How does it work? : 
    
    1) Histogram :: first it makes histogram of V and S values, here we have bin of 128 
    different bins with intensity varying from 0 to 256
    
    2) Slicing :: Ignore lower intensity in both S and V spaces, sliced histograms focusing higher intensity.
    
    3) Maximum repeated Intensity :: filter the maximum repeated frequency for intensity and use that as
    threshold for mask with upper bound
    
    4) HSV Mask :: creating s and v mask using lower and upper bound. Firstly, s mask is used to create masked image,
    than that image is used as input to create v masked image.
    
    input : rgb_image
    
    output : masked rgb image processed using v and s filter mask
    
    """
    #print(slicing)
    ## TODO: Convert image to HSV color space
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    # HSV channels
    h = hsv_image[:,:,0]
    s = hsv_image[:,:,1]
    v = hsv_image[:,:,2]
    
    np. set_printoptions(threshold=np. inf) #setting print function to visulaize the full array
    #print('s :{}'.format(s))
    #print('v :{}'.format(v))
    
    #--------------------------------------#  Step 1 : histogram  #------------------------------------------#
    #histogram to see the repeating intenstiy with its frequency
    hist_v = np.histogram(v, bins = 128, range = (0,256))
    #print('hist_v : {}'.format(hist_v))
    
    #histogram to see the repeating intenstiy with its frequency
    hist_s = np.histogram(s, bins = 128, range = (0,256))
    #print('hist_s : {}'.format(hist_s))

    #histogram has range which is a bin, each bin ranges from x to y, some value, here is the center point of each bin
    bin_edges = hist_v[1]
    #print('bin_edges : {}'.format(bin_edges))
    bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges)-1])/2
    #print('bin_centers : {}'.format(bin_centers))
    
    #--------------------------------------#  Step 2 : Slicing  #------------------------------------------#
    bin_centers_sliced = bin_centers[slicing:]
    #print('bin_centers_sliced : {}'.format(bin_centers_sliced))
    #slicing hist_s[0] to access only the higher intensity frequencies >136,
    hist_s_sliced = hist_s[0][slicing:]
    #slicing hist_v[0] to access only the higher intensity frequencies >152,
    hist_v_sliced = hist_v[0][slicing:]
    #print('hist_v_sliced : {}'.format(hist_v_sliced))
    #print('hist_s_sliced : {}'.format(hist_s_sliced))
    
    #--------------------------------------#  Step 3 : Maximum repeated Intensity  #------------------------------------------#
    #finding a max repeating frequency and using it to set a threshold for s values
    max_num_of_freq_s = np.where(hist_s_sliced == hist_s_sliced.max()) #index of maximum number in frequency, returns array
    index_of_max_freq_s = max_num_of_freq_s[0][0] #accesing index out of that array
    max_freq_int_s =  int(bin_centers_sliced[index_of_max_freq_s]) #intensity value at that index
    
    
    #finding a max repeating frequency and using it to set a threshold for v values
    max_num_of_freq_v = np.where(hist_v_sliced == hist_v_sliced.max()) #index of maximum number in frequency, returns array
    index_of_max_freq_v = max_num_of_freq_v[0][0] #accesing index out of that array
    max_freq_int_v =  int(bin_centers_sliced[index_of_max_freq_v]) #intensity value at that index
    
    #hsv values to creat mask for v values
    lower_bound_v = np.array([0,0,0])
    #print('max_freq_int_v : {}'.format(max_freq_int_v))
    upper_bound_v = np.array([179,255,max_freq_int_v])
    
    #hsv values to creat mask for s values
    lower_bound_s = np.array([0,0,0])
    #print('max_freq_int_s : {}'.format(max_freq_int_s))
    upper_bound_s = np.array([179,max_freq_int_s,255])
    """
    #--------------------------------------#  Step 4 : creating S then V masked image  #-------------------------------#
    #creating hsv mask for s values
    mask_s = cv2.inRange(hsv_image,lower_bound_s,upper_bound_s)
    #print('mask_s : {}'.format(mask_s))
    s_masked_im = np.copy(rgb_image)
    s_masked_im[mask_s != 0] = [0,0,0]
    
    
    #creating hsv mask for v values
    mask_v = cv2.inRange(hsv_image,lower_bound_v,upper_bound_v)
    #print('mask_v : {}'.format(mask_v))
    v_masked_im = s_masked_im
    v_masked_im[mask_v != 0] = [0,0,0]
    #print('v_masked_im : {}'.format(v_masked_im))
    #print('len : {}'.format(v_masked_im.shape))
    
    return v_masked_im
    """ 
    #--------------------------------------#  Step 4A : creating V then S masked image  #-------------------------------#
    
    #creating hsv mask for v values
    mask_v = cv2.inRange(hsv_image,lower_bound_v,upper_bound_v)
    #print('mask_v : {}'.format(mask_v))
    v_masked_im = np.copy(rgb_image)
    v_masked_im[mask_v != 0] = [0,0,0]
    #print('v_masked_im : {}'.format(v_masked_im))
    #print('len : {}'.format(v_masked_im.shape))


    #creating hsv mask for s values
    mask_s = cv2.inRange(hsv_image,lower_bound_s,upper_bound_s)
    #print('mask_s : {}'.format(mask_s))
    s_masked_im = v_masked_im
    s_masked_im[mask_s != 0] = [0,0,0]
    #plt.imshow(s_masked_im)
    
    #f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
    #ax1.set_title('Standardized image')
    #ax1.imshow(rgb_image)
    #ax2.set_title('Masked image')
    #ax2.imshow(s_masked_im, cmap='gray')
    
    return s_masked_im

def proportion_zeros(rgb_image):
    
    """
    goal : returns proportional ratio of number of zeros vs. total count of members in array,
    to determine percentage of useful information available.
    
    how does it work? : 
    1) it counts number of zeros in the masked image and number of total avilable resolution, 
    here it would be 32X32X3 3 is for RGB having 3 layer.
    2) that ratio of  = zeros in the image / total spaces in the image is the proportion!
    
    input : masked image
    
    output : propotion varying from 0 to 1
    
    """
    
    occurrences = np.count_nonzero(rgb_image == [0,0,0]) #counting the number of zeros within masked image
    total_count = 32*32*3 #total places for stored intensity value within RGB values
    proportion = occurrences/total_count #proportion of zeros/total_count
    
    return proportion

def opt_masked_image(rgb_image, upper_limit = 0.92, lower_limit = 0.87):
    
    """
    goal : this function adjust the slicing values to optimize masked image to make sure thare is
    enough details available in masked image
    
    How does it work?:
    1) the lower and upper limit of proportion is set. proportion of cells with zero w.r.t total cells
    
    2) slicing parameter is modified, to vary the slicing, which will take into consideration higher or lower
    intensity based on amount of black available in the image.
    
    - if black portion is higher than upper_limit it will reduce the slicing and reduced the upper bound of v and s mask
    - if black portion is lower than lower_limit it will increase the slicing to increase the upper bound of v and s mask
    
    3) plotting the image for visulization
    
    input : output from function 'create_feature'
    
    output : masked rgb image to be analyzed for final layer of color analysis
    
    """
    #--------------------------------------#  Step 1 : proportion limits  #------------------------------------------#
    masked = create_feature(rgb_image)
    proportion = proportion_zeros(masked)
    
    #--------------------------------------#  Step 2 : Using slicing to modify proportions  #-----------------------------#
    slicing = 2
    for i in range(128):
        #if the black area is much more, we reduce the slicing and try to get atleast 100 - upper_limit % useful info
        if proportion >= upper_limit:
            if slicing != 128 and slicing != 0:
                slicing-=1
                masked = create_feature(rgb_image,slicing)
                proportion = proportion_zeros(masked)
                #print(proportion)
        #if the black area is much less, we increase the slicing and try to get atleast 100 - lower limit % useful info
        elif proportion <= lower_limit:
            if slicing != 128 and slicing != 0:
                slicing+=1
                masked = create_feature(rgb_image,slicing)
                proportion = proportion_zeros(masked)
                #print(proportion)
    
    #--------------------------------------#  Step 3 : plotting the image for verification  #-----------------------------#

    #f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
    #ax1.set_title('Standardized image')
    #ax1.imshow(rgb_image)
    #ax2.set_title('Masked image')
    #ax2.imshow(masked, cmap='gray')

    return masked
    
one = opt_masked_image(STANDARDIZED_LIST[780][0])

# This function should take in RGB image input
# Analyze that image using your feature creation code and output a one-hot encoded label
def estimate_label(rgb_image):
    
    ## TODO: Extract feature(s) from the RGB image and use those features to
    ## classify the image and output a one-hot encoded label
    
    #Red = Hue (0 to 10) + (160 to 180)
    #Yellow = Hue(16..45)
    #Green = Hue(46..100)
    
    """ 
    goal : predicts the lable "Green, yellow or red" based on hue value avilable in the masked image
    
    how does it work? :
    
    Step 1 : isolating hue channel of the image.
    
    Step 2 : calculating histograms to calculate hue range
    
    Step 3 : calculating total intensity of red, yellow and green color within the image
    
    Step 4 : Checking if the red, yellow and Green is on the right location. Red would be on top 1/3rd, green would be
    normally on bottom 1/3rd and yellow would be in between
    
    step 5 : generating list of ratios between total color/ color spot, if minimum 40% of color is in right spot,
    if true, it will return TRUE
    
    step 6 : checking the difference between first and second highest intensity, here if the difference is minor,
    it will select the color which has higher proportion on the right spot!
    
    step 7 : if differnece is not much, but none of the lights in usual spot
    
    step 8 : if intensity is not first or second, but all the intensity is on right spot, good chance thats our color
    
    step 9 : if masked image is completely black, return [0,0,0]
    
    step 10 : if 90% of green color on its supposed place, thats our color
    
    step 11 : if most of the value of color_freq and color itself is zero, based on few avalilable values, 
    if they are on right spot, thats our color 
    
    step 12 : if predicted label is shorter than 3 elements - raise error!
    
    input : standardized and masked 32x32 rgb image as an input from function "opt_masked_image" 
    
    output : array for predicted label '[1,0,0]' for red, '[0,1,0]' for yellow and '[0,0,1]' for red
    
    """
    
    #-------------------# Step 1 : isolating hue channels of the image #------------------------------#
    masked_image = opt_masked_image(rgb_image) #providing with masked image
    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2HSV) #converting rgb masked image to hsv space
    h = masked_image[:,:,0]
    #print(h)
    #plt.imshow(h)
    
    #-------------------# Step 2 : creating histograms to calculate hue range #---------------------------#
    hist = np.histogram(h, bins = 36, range = (1,181))
    #print(hist)
    #finding center of the range and assigning the frequency to it
    bin_edges = hist[1]
    bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges)-1])/2
    #print(bin_centers)
    #print(hist)
    
    #-------------------# Step 3 : calculating total intensity of red, yellow and green color #------------------------#
    #Red = Hue (0 to 10) + (160 to 180) # here it is 1 to 11 and 161 to 181
    #Yellow = Hue(16..45) # here it is 16 to 46
    #Green = Hue(46..100) # here it is 47 to 101
    
    red = [3.5,8.5,163.5,168.5,173.5,178.5]
    red_freq = []
    yellow = list(np.arange(18.5, 43.5, 5.0)) #for floats range does not work, so have to use numpy
    yellow_freq = []
    green = list(np.arange(43.5, 98.5, 5.0))
    green_freq = []
    
    for center in red:
        frequency = np.where(bin_centers == center) #index of frequency 
        index = frequency[0][0]
        red_freq.append(hist[0][index]) #accesing index out of that array
    
    for center in yellow:
        frequency = np.where(bin_centers == center) #index of frequency 
        index = frequency[0][0]
        yellow_freq.append(hist[0][index]) #accesing index out of that array
        
    for center in green:
        frequency = np.where(bin_centers == center) #index of frequency 
        index = frequency[0][0]
        green_freq.append(hist[0][index]) #accesing index out of that array
        
    #print("red_freq : {}".format(red_freq))
    #print("red : {}".format(sum(red_freq)))
    #print("yellow_freq : {}".format(yellow_freq))
    #print("yellow : {}".format(sum(yellow_freq)))
    #print("green_freq : {}".format(green_freq))
    #print("green : {}".format(sum(green_freq)))
    
    #the best intensity in the list
    list_of_color = [sum(red_freq),sum(yellow_freq), sum(green_freq)]
    max_color = max(list_of_color)
    index_max = list_of_color.index(max_color) #index of max_color in the list
    
    #second best intensity in the list
    list_of_color_2nd_best = [sum(red_freq),sum(yellow_freq), sum(green_freq)]
    list_of_color_2nd_best.sort()
    second_best = list_of_color_2nd_best[1]
    
    #------------------# step 4 : calculating physical location of the color #---------------------------------------#
    
    #possible locations of colors in images for red, yellow and green hue
    h_red = h[0:10,0:31]
    h_yellow = h[9:22,0:31]
    h_green = h[18:32,0:31]
    
    hist_red_1 = np.histogram(h_red, bins = 1, range = (20,30))
    hist_red_2 = np.histogram(h_red, bins = 1, range = (150,170))
    hist_yellow = np.histogram(h_yellow, bins = 1, range = (15,45))
    hist_green = np.histogram(h_yellow, bins = 1, range = (46,100))
    
    red_top = hist_red_1[0][0] + hist_red_2[0][0]
    yellow_middle = hist_yellow[0][0]
    green_bottom = hist_green[0][0]
    
    #------------------# step 5 : checking if minimum 40% of color is in right spot #------------------------------------#
    
    #creating conditions for the physical location of color
    is_red_on_top = ((red_top/sum(red_freq)) > 0.4) if sum(red_freq)!=0 else False
    is_yellow_on_middle = ((yellow_middle/sum(yellow_freq)) > 0.4) if sum(yellow_freq)!=0 else False
    is_green_on_bottom = ((green_bottom/sum(green_freq)) > 0.4) if sum(green_freq)!=0 else False
    is_on_spot = [is_red_on_top, is_yellow_on_middle, is_green_on_bottom]          

    #print('red_top : {}'.format(red_top))
    #print('yellow_middle : {}'.format(yellow_middle))
    #print('green_bottom : {}'.format(green_bottom))
    #print('is_red_on_top : {}'.format(is_red_on_top))
    #print('is_yellow_on_middle : {}'.format(is_yellow_on_middle))
    #print('is_green_on_bottom : {}'.format(is_green_on_bottom))
    #print('max_color : {}'.format(max_color))
    #print(list_of_color)
    
    #------------------# step 6 : checking the difference between first and second highest intensity #-------------------#
    #if the difference between second best and best is less than imp_pix pixels,clarification is needed based on 
    #the place of the light, top middle or the bottom
    
    imp_pix = 5
    first_vs_second = abs(max_color-second_best)
    
    predicted_label = []
    if first_vs_second > imp_pix:
        for i in range(len(list_of_color)):
            if max_color == list_of_color[i]:
                predicted_label.append(1)
            else:
                predicted_label.append(0)
    elif first_vs_second <= imp_pix:
        for i in range(len(list_of_color)):    
            if max_color == list_of_color[i] and is_on_spot[i]:
                predicted_label.append(1)
            else:
                predicted_label.append(0)
    
    #------------------# step 7 : if differnece is not much, but none of the lights in usual spot #-------------------#
    
    #if none of the color of light in its usual spot, it will choose, color_max label and ignore the second best
    if True not in is_on_spot:
        predicted_label[index_max] = 1
        
    #------------------# step 8 : if intensity is not first or second, but all the intensity
                       # is on right spot, good chance thats our color #-----------------------------------------------#
        
    #if any proportion is exceptionally well above 0.9, there is a good chance thats the light we want,
    final_red = red_top/sum(red_freq) if sum(red_freq)!=0 else 0
    final_yellow = yellow_middle/sum(yellow_freq) if sum(yellow_freq)!=0 else 0
    final_green = green_bottom/sum(green_freq) if sum(green_freq)!=0 else 0
    is_on_spot_value = [final_red,final_yellow,final_green]
    spot_value = [sum(red_freq),sum(yellow_freq),sum(green_freq)]
            
    #proportion on spot is more than 90% and that proportion is max in is_on_spot_value list 
    #and that color on spot has more than 20 freq
    for i in range(len(is_on_spot_value)):
        if is_on_spot_value[i]>0.9 and is_on_spot_value[i] == max(is_on_spot_value) and spot_value[i]>= 10:
            predicted_label = [0,0,0]
            predicted_label[i] = 1         
    
    #------------------# step 9 : if masked image is completely black, return [0,0,0] #-------------------#
    
    #if mask is black completely, there is no red, yellow or green light - adjust the masking proportion to 
    #uncover the lights behind mask for blurr images.
    if sum(list_of_color) == 0:
        predicted_label = [0,0,0]
     
    #------------------# step 10 : if 90% of green color on its supposed place, thats our color #-------------------#
    
    #priority of green image over everything
    if spot_value[2] == max(spot_value) and spot_value[2] != 0:
        predicted_label = [0,0,1]
    
    #------------------# step 11 : if most of the value of color_freq and color itself is zero, 
                        #based on few avalilable values, if they are on right spot, thats our color #-------------------#
    
    # if all the values are zero for all colors, except green and red, green is more than half of red, classify as green    
    if red_top == 0 and yellow_middle == 0 and sum(yellow_freq) == 0 and (sum(green_freq) != 0 or green_bottom != 0):
        predicted_label = [0,0,1]
    
    # if all the values are zero for all colors, except yellow and red, yellow is more than 0.15 of red, classify as yellow 
    if red_top == 0 and green_bottom == 0 and (sum(yellow_freq) != 0 or yellow_middle != 0) and sum(green_freq) == 0:
        predicted_label = [0,1,0]
    
    #-----------------# step 12 : if predicted label is shorter than 3 elements - raise error! #-----------------------------#
    if len(predicted_label)<3:
        raise ValueError('predicted label is only 2')
        
    return predicted_label

# Using the load_dataset function in helpers.py
# Load test data
TEST_IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TEST)

# Standardize the test data
STANDARDIZED_TEST_LIST = standardize(TEST_IMAGE_LIST)

# Shuffle the standardized test data
random.shuffle(STANDARDIZED_TEST_LIST)

# Constructs a list of misclassified images given a list of test images and their labels
# This will throw an AssertionError if labels are not standardized (one-hot encoded)

def get_misclassified_images(test_images):
    # Track misclassified images by placing them into a list
    misclassified_images_labels = []

    # Iterate through all the test images
    # Classify each image and compare to the true label
    for image in test_images:

        # Get true data
        im = image[0]
        true_label = image[1]
        assert(len(true_label) == 3), "The true_label is not the expected length (3)."

        # Get predicted label from your classifier
        predicted_label = estimate_label(im)
        assert(len(predicted_label) == 3), "The predicted_label is not the expected length (3)."

        # Compare true and predicted labels 
        if(predicted_label != true_label):
            # If these labels are not equal, the image has been misclassified
            misclassified_images_labels.append((im, predicted_label, true_label))
            
    # Return the list of misclassified [image, predicted_label, true_label] values
    return misclassified_images_labels


# Find all misclassified images in a given test set
MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST)

# Accuracy calculations
total = len(STANDARDIZED_TEST_LIST)
num_correct = total - len(MISCLASSIFIED)
accuracy = num_correct/total

print('Accuracy: ' + str(accuracy))
print("Number of misclassified images = " + str(len(MISCLASSIFIED)) +' out of '+ str(total))

# Visualize misclassified example(s)
## TODO: Display an image in the `MISCLASSIFIED` list 
## TODO: Print out its predicted label - to see what the image *was* incorrectly classified as

def show_mc_image(number=0):
    
    plt.imshow(MISCLASSIFIED[number][0])
    print('Misclassified image number : {}'.format(number))
    print('Label :{}'.format(MISCLASSIFIED[number][1]))
    
    estimate_label(MISCLASSIFIED[number][0])
    plt.imshow(MISCLASSIFIED[number][0])

    #for i in range(40,len(MISCLASSIFIED)):
        #estimate_label(MISCLASSIFIED[i][0])

show_mc_image(7)
    
# Importing the tests
import test_functions
tests = test_functions.Tests()

if(len(MISCLASSIFIED) > 0):
    # Test code for one_hot_encode function
    tests.test_red_as_green(MISCLASSIFIED)
else:
    print("MISCLASSIFIED may not have been populated with images.")

