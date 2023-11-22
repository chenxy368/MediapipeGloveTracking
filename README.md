# MediapipeGloveTracking
This project proposes a vision-based glove tracking and gesture recognition system. Our approach generalizes hand-tracking solutions to glove tracking using a deep learning-based preprocessing method and establishes a highly robust gesture recognition module based on the skeleton output. The system comprises three main components. The first component is a preprocessing method that utilizes the YOLOv7 instance segmentation model to obtain a mask of the glove. Once the mask is acquired, we transform the glove's color to a skin-like color in the CIELAB color space. This preprocessing step enables our system to adapt to various glove colors, patterns, materials, and styles while maintaining compatibility with existing hand-tracking solutions. With the preprocessed image, the second component employs Google's open-source MediaPipe hand-tracking solution to extract the glove's skeleton representation. This step allows our system to capture the underlying structure of the glove and its movements, which is crucial for the subsequent gesture recognition stage. The third component focuses on gesture recognition. Leveraging the extracted skeleton representation, we employ a heuristic approach and deep learning models to recognize static gestures for dynamic gesture recognition. To enhance the generalization capabilities of our preprocessing method, we develop a module that automatically generates segmentation annotations and training samples for fine-tuning the YOLOv7 model's weights. This module further improves the system's ability to handle the vast array of gloves.

## Poster Overview
<p align="center">
   <img src="readme/xinyang_chen.jpg">
</p>

## How to Use
```bash
// pass arguments of yolov7's weight, input source, name of the project, and target gesture of dynamic recognition
python main.py --weight $weight_path --source $source_path --name $name --target_gesture $gesture
// using color shift method to process gloves
python main.py --weight $weight_path --source $source_path --name $name --target_gesture $gesture --color-shift
// using dataset generation module to generate data for Yolo's training
python main.py --weight $weight_path --source $source_path --name $name --target_gesture $gesture --color-shift --dataset_generation
```

## Output
If you input an image, the static recognizer will check static gestures.

![1700524060231](https://github.com/chenxy368/MediapipeGloveTracking/assets/98029669/1726e4a7-7324-456d-a388-dc64eae84969)

If you input a video, the static and dynamic recognizer will check static and dynamic gestures separately. "Finger Gesture: XXX" is the result given by a machine learning model. Refer to more information at (https://github.com/Flora9978/CerLab_RealTime_HandGesture_Recognition). "Gesture: XXX" is the result given by a heuristic classifier. 

![1700525041866](https://github.com/chenxy368/MediapipeGloveTracking/assets/98029669/beb3d06f-e6c2-4ca2-8502-be6ffb2b7da3)

## Arguments
Checking main.py, there are many arguments.
```python
def parse_opt():
   solution_parser = argparse.ArgumentParser()
   # Using baseline preprocessing instead of Yolov7-based preprocessing
   solution_parser.add_argument('--baseline', action='store_true', help='run baseline or yolo_glove_tracker')
   # Do not use recognition module
   solution_parser.add_argument('--norecognition', default=False, action='store_true', help='run recognition module')
   # Using a simple Kalman filter to reduce glittering 
   solution_parser.add_argument('--kalmanfilter', default=False, action='store_true', help='run kalman filter module')
   # Set the target gesture of dynamic recognition. If the 'five' is set, the dynamic recognition will start once the static gesture is 'five'
   solution_parser.add_argument('--target_gesture', type=str, default='five', help='target gesture of dynamic recognition')
   solution_type, _ = solution_parser.parse_known_args()
    
    
   if solution_type.baseline:
      # baseline arg
      # Baseline preprocessing is a straightforward method that changes the glove's color in a rough way. More information can be referred at
      # branch glove of (https://github.com/Flora9978/CerLab_RealTime_HandGesture_Recognition)
      baseline_parser = argparse.ArgumentParser()
      baseline_parser.add_argument('--low_bound', nargs='+', type=int, help='low boundary for color mask in CIELAB colorspace')
      baseline_parser.add_argument('--high_bound', nargs='+', type=int, help='high boundary for color mask in CIELAB colorspace')
      baseline_parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
      baseline_parser.add_argument('--source', type=str, default='0', help='file, num for webcam')
      baseline_parser.add_argument('--nosave', default = False, action='store_true', help='do not save results')
      baseline_parser.add_argument('--save_dir', default = ROOT / 'runs/baseline', help='save results to dir')
      baseline_parser.add_argument('--view-img', default = False, action='store_true', help='show results')
      opt, _ = baseline_parser.parse_known_args()
      opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
      print_args(vars(opt))
      return solution_type, opt

   # yolo arg
`  # Most arguments below are exactly the same as the original Yolov7 segmentation model's arguments. More information can be referred at (https://github.com/WongKinYiu/yolov7/tree/u7)
   yolo_parser = argparse.ArgumentParser()
   yolo_parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s-seg.pt', help='model path(s)')
   yolo_parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
   yolo_parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
   yolo_parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
   yolo_parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
   yolo_parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
   yolo_parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
   yolo_parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
   yolo_parser.add_argument('--view-img', action='store_true', help='show results')
   yolo_parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
   yolo_parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
   yolo_parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
   yolo_parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
   yolo_parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
   yolo_parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
   yolo_parser.add_argument('--augment', action='store_true', help='augmented inference')
   yolo_parser.add_argument('--visualize', action='store_true', help='visualize features')
   yolo_parser.add_argument('--update', action='store_true', help='update all models')
   yolo_parser.add_argument('--project', default=ROOT / 'runs/predict-seg', help='save results to project/name')
   yolo_parser.add_argument('--name', default='exp', help='save results to project/name')
   yolo_parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
   yolo_parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
   yolo_parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
   yolo_parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
   yolo_parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
   yolo_parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
   # Using dataset generation module to generate a dataset for Yolo's training
   yolo_parser.add_argument('--dataset_generation', default=False, action='store_true', help='use OpenCV for dataset generation')
   # Using color shift module to change glove's color to a skin-like color
   yolo_parser.add_argument('--color-shift', default=False, action='store_true', help='Using color shift for glove preprocessing')
   opt, _ = yolo_parser.parse_known_args()
   opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
   print_args(vars(opt))
   return solution_type, opt
```

## Dataset Generation
The Automatic Glove Dataset (AGD) is generated using an automatic labeling module, which leverages YOLOv7's segmentation results and MediaPipe's hand-tracking outcomes to obtain a refined mask of gloves. The primary objective of this dataset is to enable the model to focus on specific types of gloves used in target scenarios. 

![1700613587898](https://github.com/chenxy368/MediapipeGloveTracking/assets/98029669/73eb9b6c-90f8-4b6a-a2af-62749935faa7)

```python
if dataset_generation and save_meanshift:
   folder_path, file_name = os.path.split(save_path)
                
   new_folder_path = os.path.join(folder_path, 'mask_res')
   if not os.path.exists(new_folder_path):
      os.makedirs(new_folder_path)
   save_path_mask = os.path.join(new_folder_path, file_name)
   save_path_mask = save_path_mask.replace('.mp4', '_' + str(meanshift_count) + '.jpg')

   cv2.imwrite(save_path_mask, cv2.bitwise_and(save_mask_img, save_mask_img, mask=label_mask))

   new_folder_path = os.path.join(folder_path, 'images')
   if not os.path.exists(new_folder_path):
      os.makedirs(new_folder_path)
                        
   save_path_img = os.path.join(new_folder_path, file_name)
   save_path_img = save_path_img.replace('.mp4', '_' + str(meanshift_count) + '.jpg')
   cv2.imwrite(save_path_img, save_mask_img)
                         
   new_folder_path = os.path.join(folder_path, 'labels')
   if not os.path.exists(new_folder_path):
      os.makedirs(new_folder_path)
   save_path_label = os.path.join(new_folder_path, file_name)
   save_path_label = save_path_label.replace('.mp4', '_' + str(meanshift_count) + '.txt')

   # We use No.80 to mark the gloves. YOLOv7's given weight trained by COCO dataset already has 80 classes and we added a new class.
   segment = [80]
    
   for j in range(approx.shape[0]-1, 0, -1):
      segment.append(approx[j][0][0] / label_mask.shape[1])
      segment.append(approx[j][0][1] / label_mask.shape[0])
            
   with open(save_path_label, "w") as file:
      file.write(" ".join(str(round(j, 6)) for j in segment) + "\n") 
```
To enable the module, use a video source and add argument --dataset_generation. The module can find successfully detected frames and refine the mask. In the project, there are three folders: mask_res, images, and labels. The mask_res have the masked images which are used to clean low-quality samples. You can directly use the images and labels folders to finetune YOLOv7 models. However, the training configuration file cannot be generated automatically here so you may need to write your yaml file manually. More information on training can be referred to at (https://github.com/WongKinYiu/yolov7/tree/main).

