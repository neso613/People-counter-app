# People-counter-app
People counter app demonstrates a AI based IoT solution using Intel® hardware and software tools. This app detects person/s in camera view, providing the number of persons in the frame, average duration of person in frame, and total no of persons detected by app. This project is a part of Intel Edge AI for IOT Developers Nanodegree
![people-counter-python](./images/people-counter-image.png)

### How it works?
The counter will use the Inference Engine included in the Intel® Distribution of OpenVINO™ Toolkit. The model used should be able to identify people in a video frame. The application should count the number of people in the current frame, the duration that a person is in the frame (time elapsed between entering and exiting a frame) and the total count of people. It then sends the data to a local web server using the Paho MQTT Python package.
![architectural diagram](./images/arch_diagram.png)

## Requirements
#### Hardware
* 6th to 10th generation Intel® Core™ processor with Iris® Pro graphics or Intel® HD Graphics.
* OR use of Intel® Neural Compute Stick 2 (NCS2)

#### Software
*   Intel® Distribution of OpenVINO™ toolkit 2019 R3 release
*   Node v6.17.1
*   Npm v3.10.10
*   CMake
*   MQTT Mosca server
*   Python 3.5 or 3.6

## Explaining Custom Layers

I have tried four public model downloaded from Tensorflow model zoo to optimize. Named [ssd_mobilenet_v1_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz), [ssd_mobilenet_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz), [ssdlite_mobilenet_v2_coco](http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz) and [faster_rcnn_inception_v2_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz). 

All models have different accuracy and speed([see](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md))

To convert the models into Intermediate Representation. I have used following command:
<strong>For SSD models</strong>
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model path_to_folder/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config path_to_folder/pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --model_name any_new_name_for_model --output_dir models/IR

<strong>For Faster RCNN model</strong>
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model path_to_folder/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config path_to_folder/pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support --model_name any_new_name_for_model --output_dir models/IR

To run faster-rcnn model, main_fasterrcnn.py and main.py for SSD models are used respectively.
App also notify user when person more than 5 or person/s in store more than 5 mins.

## Comparing Model Performance

SSD based models are small in size and fast but not good at accuracy point of view. Faster RCNN model is good at accuracy but lack at speed. To fulfill the exact necessity, I found Intels pretrainded and optimised model i.e person-detection-retail_0013 is good for use to main good model's prediction propbability greater than 50%.

Original model consist of following contents: checkpoint frozen_inference_graph.mapping model.ckpt.index pipeline.config frozen_inference_graph.pb model.ckpt.data-00000-of-00001 model.ckpt.meta saved_model. I have only consider size of frozen model file(contains weight and bias) and pipeline.config for comparision because these are used for optimization i.e to generate IR files. IR files used for inferencing is of type FP32.

Model Name | Frozen Model Size | Intermediate Representation Size
------------ | -------------
ssd_mobilenet_v1_coco | 27.7 MB | 26 MB
ssd_mobilenet_v2_coco | 66.5 MB | 64.3 MB
ssdlite_mobilenet_v2_coco | 18.9 MB | 17.2 MB
faster_rcnn_inception_v2_coco | 54.5 MB | 50.9 MB
person-detection-retail-0013 | - | 2.90 MB

To compare Inferencing speed and Frame per second after optimization, I have saved output video to disk also.
Model Name | Inferencing speed | Frame per second
------------ | -------------
ssd_mobilenet_v1_coco | 50ms | 8
ssd_mobilenet_v2_coco | 69ms | 7
ssdlite_mobilenet_v2_coco | 33ms | 10
faster_rcnn_inception_v2_coco | 875ms | 1
person-detection-retail-0013 | 45ms | 9

## Setup

### Install Intel® Distribution of OpenVINO™ toolkit

Refer to https://software.intel.com/en-us/articles/OpenVINO-Install-Linux for more information about how to install and setup the Intel® Distribution of OpenVINO™ toolkit.

You will need the OpenCL™ Runtime Package if you plan to run inference on the GPU. It is not mandatory for CPU inference. 

### Install Nodejs and its depedencies

- This step is only required if the user previously used Chris Lea's Node.js PPA.

	```
	sudo add-apt-repository -y -r ppa:chris-lea/node.js
	sudo rm -f /etc/apt/sources.list.d/chris-lea-node_js-*.list
	sudo rm -f /etc/apt/sources.list.d/chris-lea-node_js-*.list.save
	```
- To install Nodejs and Npm, run the below commands:
	```
	curl -sSL https://deb.nodesource.com/gpgkey/nodesource.gpg.key | sudo apt-key add -
	VERSION=node_6.x
	DISTRO="$(lsb_release -s -c)"
	echo "deb https://deb.nodesource.com/$VERSION $DISTRO main" | sudo tee /etc/apt/sources.list.d/nodesource.list
	echo "deb-src https://deb.nodesource.com/$VERSION $DISTRO main" | sudo tee -a /etc/apt/sources.list.d/nodesource.list
	sudo apt-get update
	sudo apt-get install nodejs
	```

### Install the following dependencies

```
sudo apt update
sudo apt-get install python3-pip
pip3 install numpy
pip3 install paho-mqtt
sudo apt install libzmq3-dev libkrb5-dev
sudo apt install ffmpeg
```
### Install npm

There are three components that need to be running in separate terminals for this application to work:

-   MQTT Mosca server 
-   Node.js* Web server
-   FFmpeg server
     
Go to people-counter-python directory
```
cd <path_to_people-counter-python_directory>
```
* For mosca server:
   ```
   cd webservice/server
   npm install
   ```

* For Web server:
  ```
  cd ../ui
  npm install
  ```
  **Note:** If any configuration errors occur in mosca server or Web server while using **npm install**, use the below commands:
   ```
   sudo npm install npm -g 
   rm -rf node_modules
   npm cache clean
   npm config set registry "http://registry.npmjs.org"
   npm install
   ```
## Configure the application

### Step 1 - Start the Mosca server

```
cd webservice/server/node-server
node ./server.js
```

You should see the following message, if successful:
```
connected to ./db/data.db
Mosca server started.
```

### Step 2 - Start the GUI

Open new terminal and run below commands.
```
cd ../../ui
npm run dev
```

You should see the following message in the terminal.
```
webpack: Compiled successfully
```

### Step 3 - FFmpeg Server

Open new terminal and run the below commands.
```
cd ../..
sudo ffserver -f ./ffmpeg/server.conf
```

### Step 4 - Run the code

Open a new terminal to run the code.

#### Setup the environment

You must configure the environment to use the Intel® Distribution of OpenVINO™ toolkit one time per session by running the following command:
```
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
```
#### Running on the CPU

When running Intel® Distribution of OpenVINO™ toolkit Python applications on the CPU, the CPU extension library is required. This can be found at /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/

Though by default application runs on CPU, this can also be explicitly specified by ```-d CPU``` command-line argument:

```
python3.5 main.py -i resources/Pedestrain_Detect_2_1_1.mp4 -m /opt/intel/openvino/deployment_tools/tools/model_downloader/Retail/object_detection/pedestrian/rmnet_ssd/0013/dldt/person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://localhost:8090/fac.ffm
```
To see the output on a web based interface, open the link [http://localhost:8080](http://localhost:8080/) in a browser.


#### Running on the GPU

* To use GPU in 16 bit mode, use the following command

  ```
  python3.5 main.py -i resources/Pedestrain_Detect_2_1_1.mp4 -m /opt/intel/openvino/deployment_tools/tools/model_downloader/Retail/object_detection/pedestrian/rmnet_ssd/0013/dldt/person-detection-retail-0013-fp16.xml -d GPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://localhost:8090/fac.ffm
  ```
  To see the output on a web based interface, open the link [http://localhost:8080](http://localhost:8080/) in a browser.<br><br>

   **Note:** The Intel® Neural Compute Stick can only run FP16 models. The model that is passed to the application, through the `-m <path_to_model>` command-line argument, must be of data type FP16.<br><br>

* To use GPU in 32 bit mode, use the following command

  ```
  python3.5 main.py -i resources/Pedestrain_Detect_2_1_1.mp4 -m /opt/intel/openvino/deployment_tools/tools/model_downloader/Retail/object_detection/pedestrian/rmnet_ssd/0013/dldt/person-detection-retail-0013.xml -d GPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://localhost:8090/fac.ffm
  ```
  To see the output on a web based interface, open the link [http://localhost:8080](http://localhost:8080/) in a browser.


#### Running on the Intel® Neural Compute Stick
To run on the Intel® Neural Compute Stick, use the ```-d MYRIAD``` command-line argument:
```
python3.5 main.py -d MYRIAD -i resources/Pedestrain_Detect_2_1_1.mp4 -m /opt/intel/openvino/deployment_tools/tools/model_downloader/Retail/object_detection/pedestrian/rmnet_ssd/0013/dldt/person-detection-retail-0013-fp16.xml -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://localhost:8090/fac.ffm
```
To see the output on a web based interface, open the link [http://localhost:8080](http://localhost:8080/) in a browser.<br>
**Note:** The Intel® Neural Compute Stick can only run FP16 models. The model that is passed to the application, through the `-m <path_to_model>` command-line argument, must be of data type FP16.

#### Running on the FPGA

Before running the application on the FPGA, program the AOCX (bitstream) file.
Use the setup_env.sh script from [fpga_support_files.tgz](http://registrationcenter-download.intel.com/akdlm/irc_nas/12954/fpga_support_files.tgz) to set the environment variables.

```
source /home/<user>/Downloads/fpga_support_files/setup_env.sh
```
The bitstreams for HDDL-F can be found under the `/opt/intel/openvino/bitstreams/a10_vision_design_bitstreams` folder.
To program the bitstreams use the below command.

```
aocl program acl0 /opt/intel/openvino/bitstreams/a10_vision_design_bitstreams/2019R1_PL1_FP11_RMNet.aocx
```
For more information on programming the bitstreams, please refer to https://software.intel.com/en-us/articles/OpenVINO-Install-Linux-FPGA#inpage-nav-11

To run the application on the FPGA with floating point precision 16 (FP16), use the `-d HETERO:FPGA,CPU` command-line argument:
```
python3.5 main.py -d HETERO:FPGA,CPU -i resources/Pedestrain_Detect_2_1_1.mp4 -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -m /opt/intel/openvino/deployment_tools/tools/model_downloader/Retail/object_detection/pedestrian/rmnet_ssd/0013/dldt/person-detection-retail-0013-fp16.xml -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://localhost:8090/fac.ffm
```

#### Using Camera stream instead of video file

To get the input video from the camera, use ```-i CAM``` command-line argument. Specify the resolution of the camera using 
```-video_size``` command line argument.

For example:
```
python3.5 main.py -i CAM -m /opt/intel/openvino/deployment_tools/tools/model_downloader/Retail/object_detection/pedestrian/rmnet_ssd/0013/dldt/person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://localhost:8090/fac.ffm
```
To see the output on a web based interface, open the link [http://localhost:8080](http://localhost:8080/) in a browser.

**Note:**
User has to give ```-video_size``` command line argument according to the input as it is used to specify the resolution of the video or image file.

