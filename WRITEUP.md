# Project Write-Up

Detailed explanation of model used for People Counter App including model optimization and detection handling.

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

## Assess Model Use Cases

Some of the potential use cases of the people counter app are...

1. During this COVID-19 pandemic,this app is useful to find locations where people violating lockdown rules.
2. Small business persons can afford this video survillence system.
3. Easy to know the pattern of customer for shopping.
4. Easy to find traffic area and time splot for traffic.
5. Administration/Leaders could know better how much time employee give time to their work.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

I have tested this app on some other videos too. I found lighting is not major factor, but yes, in night mode or very dim light, this app suffers.

Faster RCNN is provides good accuracy and SSD models are good at speed. Intels model is good at both.

Image size makes big difference. Input image is resized as per input models. But high resolution video takes more time to render at client UI. Image size should be neither more nor less.



