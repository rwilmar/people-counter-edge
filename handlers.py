import cv2
import numpy as np
import pandas as pd

# takes frame picture from a video
def getVideoFrame(videoFile, frameNumber, enableErrImg=True):
    capture = cv2.VideoCapture(videoFile, cv2.CAP_FFMPEG)
    if (capture.isOpened() == False): 
        raise("Unable to open video file")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frameNumber)
    ret, frame = capture.read()
    capture.release()
    if not ret:
        if enableErrImg:
            frame=cv2.imread("./images/error.png")
        else:
            raise("Can't receive frame (video end?)")
    return frame

#Preprocess the input image.
def preprocessing(input_image, height, width):
    '''
    Given an input image, height and width:
    - Resize width and height
    - Transpose BGR to RGB
    - Reshape the image to add a "batch" of 1 at the start 
    '''
    image = np.copy(input_image)
    image = cv2.resize(image, (width, height))
    image = image.transpose((2,0,1))
    image = image.reshape(1, 3, height, width)

    return image


def get_mask(processed_output):
    '''
    Given an input image size and processed output for a semantic mask,
    returns a masks able to be combined with the original image.
    '''
    # Create an empty array for other color channels of mask
    empty = np.zeros(processed_output.shape)
    # Stack to make a Green mask where text detected
    mask = np.dstack((empty, processed_output, empty))

    return mask

def draw_boxes(frame, result, ct, width, height):
    '''
    Draw bounding boxes onto the frame.
    with a minimun confidence thereshold ct
    '''
    for box in result: # Output shape is nx7
        conf = box[2]
        if conf >= ct:
            #print(box)
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            #print(xmin, ymin, xmax, ymax)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 153, 255), 2)
    return frame

def create_output_image(model_type, image, output):
    '''
    Using the model type, input image, and processed output,
    creates an output image showing the result of inference.
    '''
    [h, w, c] = image.shape


    if model_type == "person-detection-retail-0002":
        print("returned")
        # Get only text detections above 0.5 confidence, set to 255
        #output = np.where(output[1][2]>0.5, output, 0)
        nPers, perVector = output.shape
        print("numero de personas: ", nPers)
        
        return draw_boxes(image, output, 0.4, w, h)

    elif model_type == "ssd_mobilenet_v2_FP16":
        
        return draw_boxes(image, output, 0.4, w, h)

    else:
        print("Unknown model type, unable to create output image.")
        return image

def process_freq_heatmap(fqImage_raw, peak_ceil):
    fqImage_raw=np.dot(fqImage_raw, (254/peak_ceil))
    fqImage_out = cv2.resize(fqImage_raw.astype(np.uint8), (90,90)) # resize to trunk resol.
    fqImage_out =np.dot(fqImage_out.astype(np.double), (peak_ceil/254)) #recover values
    return fqImage_out

def handle_personDet(output,  min_conf=0.1):
    '''
    Handles the output of the Person Detection model.
    Returns two integers: the argmax of each softmax output.
    The first is for color, and the second for type.
    '''
    detPeople=output['detection_out'] #[1x1xNx7], where N is the number of detected pedestrians.   
    return detPeople[0][0]

def handle_personDet2(output, min_conf=0.1):
    '''
    Handles the output of the Object Detection model.
    Returns list of detections above 0.1 confidence
    for only the people class defined in the COCO trained dataset
    '''
    det_list=output['DetectionOutput'][0][0]
    df1=pd.DataFrame(det_list, columns=['image_id', 'label', 'conf', 'x_min', 'y_min', 'x_max', 'y_max'])
    people=df1[(df1['conf']>min_conf) & (df1['label']==1)] 
    return people.to_numpy()

def handle_output(model_type):
    '''
    Returns the related function to handle an output,
        based on the model_type being used.
    '''
    if model_type == "person-detection-retail-0002":
        return handle_personDet
    elif model_type == "ssd_mobilenet_v2_FP16":
        return handle_personDet2
    else:
        return None

def pathToName(path = "out.mp4"):
    x = path.split("/")
    return x[-1].split(".")[0]

        