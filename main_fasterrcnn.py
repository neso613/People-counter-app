import time
import socket
import json
import cv2
import os
import sys
import numpy as np
import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference_fasterRCNN import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

INPUT_STREAM = "resources/Pedestrian_Detect_2_1_1.mp4"
CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=False, type=str,
                        help="Path to image or video file",default=INPUT_STREAM)
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default = CPU_EXTENSION,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.6,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser

def draw_bounding_boxes(frame, result,prob_threshold,width,height):
    '''
    Draw bounding boxes onto the frame.
    '''
    no_of_person_in_frame = 0
    probs = result[0, 0, :, 2]
    for i, p in enumerate(probs):
        if p > prob_threshold:
            no_of_person_in_frame += 1
            box = result[0, 0, i, 3:]
            p1 = (int(box[0] * width), int(box[1] * height))
            p2 = (int(box[2] * width), int(box[3] * height))
            frame = cv2.rectangle(frame, p1, p2, (255, 255, 255), 2)
    return frame, no_of_person_in_frame

def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client

def infer_on_stream(args,client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    frame_count = 0
    frame_time = 0
    
    duration_prev = 0
    total_count = 0
    time_thresh = 0    
    person_count_in_each_frame = 0
    last_count = 0
    previous_last_count = 0
    
    font_scale = 0.5
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Flag for the input image
    single_image_mode = False
    
    # Initialise the class
    infer_network = Network()
    
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    model = args.model
    DEVICE = args.device
    CPU_EXTENSION = args.cpu_extension
    
    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(model, CPU_EXTENSION, DEVICE)
    infer_network_input_shape = infer_network.get_input_shape()
    in_shape = infer_network_input_shape['image_tensor']

    ### TODO: Handle the input stream ###
    # Checks for live feed
    if args.input == 'CAM':
        input_Type = args.input = 0

    # Checks for input image
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        single_image_mode = True
        input_Type = args.input

    # Checks for video file
    else:
        input_Type = args.input

    ### TODO: Handle the input stream ###
    input_stream = cv2.VideoCapture(input_Type)
    if input_Type:
        input_stream.open(args.input)
    if not input_stream.isOpened():
        log.error("ERROR! Unable to open video source")

    width = int(input_stream.get(3))
    height = int(input_stream.get(4))
    
    if not single_image_mode:
        # The second argument should be `cv2.VideoWriter_fourcc('M','J','P','G')`
        # on Mac, and `0x00000021` on Linux
        # 100x100 to match desired resizing
        out = cv2.VideoWriter('output_video.mp4', 0x00000021, 30, (width,height))
    else:
        out = None
    
    ### TODO: Loop until stream is over ###
    while input_stream.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = input_stream.read()
        if not flag:
            break
        frame_count += 1
        t = time.time()
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the image as needed ###
        image = cv2.resize(frame, (in_shape[3], in_shape[2]))
        image_p = image.transpose((2, 0, 1))
        image_p = image_p.reshape(1, *image_p.shape)
  

        ### TODO: Start asynchronous inference for specified request ###
        net_input = {'image_tensor': image_p,'image_info': image_p.shape[1:]}
        total_time_spent = None
        inferencing_start = time.time()
        infer_network.exec_net(net_input, request_id=0)

        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:

            ### TODO: Get the results of the inference request ###
            detection_time = time.time() - inferencing_start
            result = infer_network.get_output()
            
            frame, current_count = draw_bounding_boxes(frame,result,prob_threshold,width,height)
            
            inference_time_message = "Inference time: {:.3f}ms".format(detection_time * 1000)
            cv2.putText(frame, inference_time_message, (25, 25),cv2.FONT_HERSHEY_COMPLEX, font_scale, (0, 10, 250),1)

            ### TODO: Extract any desired stats from the results ###
            if current_count == last_count:
                time_thresh += 1
                if time_thresh >= 5:
                    person_count_in_each_frame = last_count
                    if time_thresh == 5 and last_count > previous_last_count:
                        total_count += last_count - previous_last_count
                    elif time_thresh == 5 and last_count < previous_last_count:
                        total_time_spent = int((duration_prev / 10.0) * 1000)
            else:
                previous_last_count = last_count
                last_count = current_count
                if time_thresh >= 5:
                    duration_prev = time_thresh
                    time_thresh = 0
                else:
                    time_thresh = duration_prev + time_thresh
            current_count_label = "No of Detected Persons in current frame : {:.2f}".format(current_count)
            cv2.putText(frame, current_count_label,(25,50),font, font_scale,(255, 0, 0), 1) 
            
            total_count_label = "Total Detected Person : {:.2f}".format(total_count)
            cv2.putText(frame, total_count_label,(25,75),font, font_scale,(255, 0, 0), 1)

            alert_flag = False
            alert_msg = None
            if current_count > 5:
                alert_msg = "ALERT!!!! \n" + str(current_count) + "persons are at same place"
                alert_flag = True
            if total_time_spent is not None and total_time_spent > 3000000: # 5 min
                alert_msg = "ALERT!!!! \n" + str(current_count) + "person are in store from long time."
                alert_flag = True
            if alert_flag:
                # set the rectangle background to white
                rectangle_bgr = (0, 0, 255)
                # get the width and height of the text box
                (text_width, text_height) = cv2.getTextSize(alert_msg, font, fontScale=font_scale, thickness=1)[0]
                # set the text start position
                text_offset_x = 0
                text_offset_y = frame.shape[0] - 15
                # make the coords of the box with a small padding of two pixels
                box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 5, text_offset_y - text_height - 5))
                cv2.rectangle(frame, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
                cv2.putText(frame, alert_msg, (text_offset_x, text_offset_y), font, font_scale, color=(255, 255, 255), thickness=2)
            
            frame_time += time.time() - t
            fps = frame_count / float(frame_time)
            fps_label = "FPS : {:.2f}".format(fps)
            cv2.putText(frame, fps_label,(25,100),font, 0.9,(255, 0, 0), 1)
        
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            client.publish("person", json.dumps({"count": current_count,"total": total_count}))
            if total_time_spent is not None:
                client.publish("person/duration",json.dumps({"duration": total_time_spent}))
 

        ### TODO: Send the frame to the FFMPEG server ###
        #  Resize the frame
        frame = cv2.resize(frame, (768, 432))
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
        
        # Break if escape key pressed
        if key_pressed == 27:
            break

        ### TODO: Write an output image if `single_image_mode` ###
        if single_image_mode:
            cv2.imwrite('output_image.jpg', frame)
        else:
            out.write(frame)
        
    # Release the capture and destroy any OpenCV windows
    if not single_image_mode:
        out.release()
    input_stream.release()
    cv2.destroyAllWindows()
    
    ### TODO: Disconnect from MQTT
    client.disconnect()


def main():
    """
    Load the network and parse the output.
    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args,client)


if __name__ == '__main__':
    main()