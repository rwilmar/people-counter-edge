"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
#import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network
from handlers import (preprocessing, create_output_image, path_get_extension,
                    path_get_name, handle_output)

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60
CPU_EXT=None

#MODEL_SRC="./ssd_mobilenet_v2_FP16.xml"



def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-m", "--model", required=False, type=str,
                        default="./models/ssd_mobilenet_v2_FP16.xml",
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = None

    return client

def get_input_type(filename):
    if len(filename)==1: filename=filename+".0"
    ext=path_get_extension(filename)
    switcher = { 
        "bmp": "img", 
        "jpg": "img", 
        "png": "img", 
        "m4v": "video", 
        "mp4": "video", 
        "0": "video",
    } 
    return switcher.get(ext, "invalid") 

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    model_name = path_get_name(args.model)
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    n,c,h,w = infer_network.get_input_shape()

    ### TODO: Handle the input stream ###
    input_type=get_input_type(args.input)
    frame = None
    count_history=[]
    people_frame = 0
    people_past = 0
    total_people = 0
    
    print("network loaded", h,w)

    print ("tipo: ", input_type)
    if input_type=="img":
        frame = cv2.imread(args.input)
        preprocessed_image = preprocessing(frame, h, w)
        output = infer_network.exec_sync(preprocessed_image)
        processed_output = handle_output(model_name)(output, prob_threshold)
        people_frame, perVector = processed_output.shape
        print("numero de personas: ", people_frame)

        ### TODO: Write an output image if `single_image_mode` ###
        output_frame=create_output_image(model_name, frame, processed_output)
        cv2.imwrite("./output_net.png", output_frame)

    elif input_type=="video":
        capture = cv2.VideoCapture(args.input, cv2.CAP_FFMPEG)
        fps = capture.get(cv2.CAP_PROP_FPS)
        print("fps:",fps)
        if (capture.isOpened() == False): 
            raise Exception("ERROR: Unable to open video file")
        n=0
        ### TODO: Loop until stream is over ###
        while(capture.isOpened()):
            n+=1
            ### TODO: Read from the video capture ###
            frame = cv2.imread(args.input)
            ret, frame = capture.read()
            if not ret:
                print("Can't receive frame (video end?). Exiting ...")
                break
            if cv2.waitKey(1)==27: # exit on keyboard esc
                break

            ### TODO: Pre-process the image as needed ###
            preprocessed_image = preprocessing(frame, h, w)
            ### TODO: Start asynchronous inference for specified request ###
            request_handler = infer_network.exec_net(preprocessed_image)
            ### TODO: Wait for the result ###
            while True:
                status = infer_network.wait(request_handler)
                if status == 0:
                    break
                else:
                    print("waiting for inference . . .")
                    time.sleep(1)
            ### TODO: Get the results of the inference request ###
            output = infer_network.get_output(request_handler)
            processed_output = handle_output(model_name)(output, prob_threshold)
            ### TODO: Extract any desired stats from the results ###
            people_frame, perVector = processed_output.shape
            
            count_history.append(people_frame)
            if len(count_history)>30: count_history.pop(0)
            people_frame=sum(count_history)//len(count_history)
            if (people_frame-people_past)>0:
                total_people=total_people+(people_frame-people_past)
            people_past=people_frame
            
                ### TODO: Calculate and send relevant information on ###
                ### current_count, total_count and duration to the MQTT server ###
                ### Topic "person": keys of "count" and "total" ###
                ### Topic "person/duration": key of "duration" ###

            ### TODO: Send the frame to the FFMPEG server ###
            output_frame=create_output_image(model_name, frame, processed_output)
            cv2.imwrite("./output_net.png", output_frame)
            if n%30==0:
                print(n, people_frame, total_people)
        print("numero de personas frame:", people_frame)
        print("numero personas total:", total_people)
        capture.release()
        cv2.destroyAllWindows()
    else:
        raise Exception("ERROR: invalid input filetype (only images .jpg, .png, and videos .mp4, .m4v are supported)")


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    if args.i == "CAM":
        args.i=0
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
