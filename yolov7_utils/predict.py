# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 segmentation inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python segment/predict.py --weights yolov5s-seg.pt --source 0                               # webcam
                                                                  img.jpg                         # image
                                                                  vid.mp4                         # video
                                                                  path/                           # directory
                                                                  'path/*.jpg'                    # glob
                                                                  'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                                  'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python segment/predict.py --weights yolov5s-seg.pt                 # PyTorch
                                          yolov5s-seg.torchscript        # TorchScript
                                          yolov5s-seg.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                          yolov5s-seg.xml                # OpenVINO
                                          yolov5s-seg.engine             # TensorRT
                                          yolov5s-seg.mlmodel            # CoreML (macOS-only)
                                          yolov5s-seg_saved_model        # TensorFlow SavedModel
                                          yolov5s-seg.pb                 # TensorFlow GraphDef
                                          yolov5s-seg.tflite             # TensorFlow Lite
                                          yolov5s-seg_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import platform
from pathlib import Path
import os
import sys
import copy

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import torch
import torch.backends.cudnn as cudnn


from yolov7_utils.models.common import DetectMultiBackend
from yolov7_utils.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from yolov7_utils.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from yolov7_utils.utils.plots import Annotator, colors, save_one_box
from yolov7_utils.utils.segment.general import process_mask, scale_masks
from yolov7_utils.utils.segment.plots import plot_masks, pure_masks
from yolov7_utils.utils.torch_utils import select_device, smart_inference_mode


import numpy as np
from glove_utils import HandDetector
from glove_utils.GlobalQueue import put_value, put_EOF, get_arg, set_arg
from glove_utils.YoloMaskProcess import process_img, meanshift_postprocess, generate_mask

@smart_inference_mode()
def run(
        weights= ROOT / 'yolov5s-seg.pt',  # model.pt path(s)
        source= ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data= ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=80,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project= ROOT / 'runs/predict-seg',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        dataset_generation=False, # enable opencv dataset generator
        color_shift=False # enable color shift opencv image processing
):  
    # Dataset generation counter
    meanshift_count = 0

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer, vid_writer_track = [None] * bs, [None] * bs, [None] * bs

    # Runtime flag
    run_recognition = get_arg("recognition")
    init_all = False

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred, out = model(im, augment=augment, visualize=visualize)
            proto = out[1]

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, im0_mask_change, frame = path[i], im0s[i].copy(), im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, im0_mask_change, frame = path, im0s.copy(), im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg

            # init
            if not init_all:
                if run_recognition:
                    put_value([nosave, view_img, dataset.mode == 'image'])

                # initialize detector
                if dataset.mode == 'image':
                    set_arg("kalmanfilter", False)
                detector = HandDetector(imgsz[0], imgsz[1], dataset.mode == 'image')
                init_all = True

            # Send SavePath
            if run_recognition and save_img and dataset.mode != 'image':
                if vid_path[i] != save_path:  # new video
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    put_value([str(Path(save_path).with_suffix('')), (w, h)])
            if run_recognition and save_img and dataset.mode == 'image':
                put_value([str(Path(save_path).with_suffix(''))])

            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Keep the largest confidence------------
                MaxConf = []
                for index in range(len(det)):       
                    MaxConf.append(det[index][4])
                Max = max(MaxConf)  
                MaxI = MaxConf.index(Max)
                det = det[[MaxI]]
                ### --------------------------------------

                masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC

                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Process mask and run detection -----------------------------------------------------------------
                pure_mcolors = [(255, 255, 255)]
                im_pure_masks = pure_masks(im[i], masks, pure_mcolors)  # image with masks shape(imh,imw,3)       
                im_pure_masks = scale_masks(im.shape[2:], im_pure_masks, im0.shape)
                im0_auto_label = copy.deepcopy(im0_mask_change)
                im0_mask_change = cv2.resize(im0_mask_change, (im_pure_masks.shape[1], im_pure_masks.shape[0]))

                if color_shift:
                    im0_mask_change = process_img(im0_mask_change, im_pure_masks, imgsz)

                if run_recognition:
                    success = detector.findHands(im0_mask_change, im0_mask_change, True)
                    put_value([im0_mask_change, copy.deepcopy(detector.results)])
                else:
                    success = detector.findHands(im0_mask_change, im0_mask_change, True)
                # ---------------------------------------------------------------------------------------------------

                # Opencv mask generator-------------------------------------------------------------------------------
                if dataset_generation:
                    label_mask = np.zeros_like(im0_mask_change)
                    save_meanshift = False
                    if success:
                        res = meanshift_postprocess(im0_auto_label, im_pure_masks, detector.boundingBox(im0_auto_label.shape[1], im0_auto_label.shape[0]))
                        if res is not None:
                            save_meanshift = True
                            save_mask_img = copy.deepcopy(im0)
                            label_mask, approx = generate_mask(res, im_pure_masks, \
                                    detector.getBaseMask(res.shape[1], res.shape[0]), im0_auto_label, \
                                    detector.getSampleMask(res.shape[1], res.shape[0]))
                # -----------------------------------------------------------------------------------------------------------

                im0_mask_change = cv2.resize(im0_mask_change, (im0.shape[1], im0.shape[0]))

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Mask plotting
                mcolors = [colors(int(cls), True) for cls in det[:, 5]]
                im_masks = plot_masks(im[i], masks, mcolors)  # image with masks shape(imh,imw,3)
                annotator.im = scale_masks(im.shape[2:], im_masks, im0.shape)  # scale to original h, w

                # Write results
                for *xyxy, conf, cls in reversed(det[:, :6]):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
            elif run_recognition:
                put_value([im0, None])

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    save_path_track = save_path.replace('.jpg', '_trace.jpg')  # force *.mp4 suffix on results videos
                    cv2.imwrite(save_path_track, im0_mask_change)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if isinstance(vid_writer_track[i], cv2.VideoWriter):
                            vid_writer_track[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        save_path_track = save_path.replace('.mp4', '_trace.mp4')  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer_track[i] = cv2.VideoWriter(save_path_track, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)
                    vid_writer_track[i].write(im0_mask_change)

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

                segment = [80]
    
                for j in range(approx.shape[0]-1, 0, -1):
                    segment.append(approx[j][0][0] / label_mask.shape[1])
                    segment.append(approx[j][0][1] / label_mask.shape[0])
            
                with open(save_path_label, "w") as file:
                    file.write(" ".join(str(round(j, 6)) for j in segment) + "\n") 

                meanshift_count += 1

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

    if run_recognition:
        put_EOF()
