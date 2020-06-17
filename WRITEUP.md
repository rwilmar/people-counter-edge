# Project Write-Up

The project has been developed according to the points explained in the course and the project requirements. Three detection models has been converted and tested from which the most promising model was selected for the final configuration.

The chosen model was ssd_mobilenet_v2_coco, source: https://github.com/opencv/open_model_zoo/blob/master/models/public/ssd_mobilenet_v2_coco/ssd_mobilenet_v2_coco.md


## Explaining Custom Layers

The model selected had the following unsupported layers:
- 'PriorBoxClustered_5'
- 'PriorBoxClustered_4'
- 'PriorBoxClustered_3'
- 'PriorBoxClustered_2'
- 'PriorBoxClustered_1'
- 'PriorBoxClustered_0'
- 'DetectionOutput'

All of the unsupported layers listed are compatible with the OpenVINO toolkit via CPU extensions this method was chosen due to the speed shown and the easy of programming.

Having layers non supported would have required to handle them outside OpenVino, and while the option is valuable for cutting edge architectures where it's absolutely necessary to use the original inference, the performance could be potentially slower than using the CPU plugins.


## Comparing Model Performance

My method to compare models before and after conversion to Intermediate Representations was using the Intel Deep Learning Workbench

The difference between model accuracy pre- and post-conversion was not tested due to difficulties on the Intel DL workbench

The size of the model post-conversion was considerably smaller, using FP16, from 69.7Mb of the frozen model to 33.6Mb of the final model 

The inference time of the model post-conversion was considerably smaller using the laptop CPU (55% faster on single image tests)

The advantages of deploying the model on the edge are numerous and important for any project: compared to the implementation on cloud, network costs are cut completely or sharply because there's no explicit need to send frames to the internet (except when monitoring is required). Another advantage is the lack of inference costs on the cloud that can be much higher on the long run, compared to the cost of the devices required for the edge inference.


## Assess Model Use Cases

Some of the potential use cases of the people counter app are:
- Ticketing system in buses: either for maintenance of the cars or for ticketing, in a scenario where people don't pay directly like airport bus shuttles, or inter-campus buses
- People attending events such as trade fairs or conferences where the input/output lines are controlled: to statistically study the rush hours or the most visited areas.
- Studies of buyer behaviours on retail/stores in either entering the store or visiting sections: showing most busy entrances or exit, the most visited areas or queue control in payment lines, all of them with great impact to the buyers experience.
- Traffic control on public infrastructures such as gates, limited stairways, etc: as a safety measure or by statistical control, is important for the authorities to measure the amount of people anonymously (if the proposed model or a similar detection model is used).
- As a safety indicator for limiting people on closed buildings, on a COVID-19 era.


## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on the model accuracy, for example the model tested has an square input limited in pixels, so is not ideal to read images with a high aspect ratio, or the ones created by fisheye lens. in those cases a correction must be applied and in extreme cases more than one inference will be needed to obtain an accurate detection.
The same problem happens when the processing is needed to images in very high resolution as the ones covering great areas, due to the very small size of the people in pixels for the input model.

No tests have been done on night conditions but due to the nature of training the accuracy is expected to suffer, for not having the color information on images, and the presence of certain light artefacts, well known in night images.


## Running the App

- For stats testing (show stats and hide ffmpeg output)
python main.py -i ./resources/Pedestrian_Detect_2_1_1.mp4 -l /opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension_sse4.so 

- Using default values
python main.py -i ./resources/Pedestrian_Detect_2_1_1.mp4 -l /opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension_sse4.so | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

- Improving model performance by adjusting thereshold to 0.3
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.3 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

## Model Research

In investigating potential people counter models, I tried each of the following models, comparing them with the Intel Deep Learning Workbench (DL Workbench) and testing them on a laptop computer:

- Model 1: ssd_mobilenet_v2_coco (option selected, FP 16)
  - https://github.com/opencv/open_model_zoo/blob/master/models/public/ssd_mobilenet_v2_coco/ssd_mobilenet_v2_coco.md
  - I converted the model to an Intermediate Representation with the following arguments
  python /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model frozen_inference_graph.pb --transformations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --data_type FP16
  - The model was ideal for the app due to the fast results compared with the other models, plus it have the smallest size of the others compared
  - Latency 66.01, 13.69 FPS in the VOCtrainval_14 dataset
  - Latency 67.46, 13.53 FPS in autogenerated dataset (1280x720)

  
- Model 2: faster_rcnn_resnet50_coco
  -  https://github.com/opencv/open_model_zoo/blob/master/models/public/faster_rcnn_resnet50_coco/faster_rcnn_resnet50_coco.md
  - The model was usable in the app, but showed a very slow throughput. 
  - Latency 3466.19, 0.27 FPS in the VOCtrainval_14 dataset
  - Latency 3573.08, 0.27 FPS in autogenerated dataset (1280x720)

- Model 3: yolo-v2-tf
  - https://github.com/opencv/open_model_zoo/blob/master/models/public/yolo-v2-tf/yolo-v2-tf.md
  - The model was usable in the app, and showed the best accuracy of the tested models, but showed a very slow throughput unusable in limited processing or high speed conditions.
  - Latency 310.7ms, 2.98 FPS in the VOCtrainval_14 dataset
  - Latency 303.33, 3.22 FPS in autogenerated dataset (1280x720)
  
- Model 4: Objects as Points - CenterNet 
 - https://github.com/xingyizhou/CenterNet/
 - The model was powerful but has the requirement of processing custom layers. For that reason it was not further tested
