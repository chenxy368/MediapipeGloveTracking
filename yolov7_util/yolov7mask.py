import torch
import os
from pathlib import Path
import sys
import numpy as np
import torch.backends.cudnn as cudnn
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolov7_util.utils.general import (Profile, cv2, non_max_suppression, scale_coords, check_img_size, check_requirements)
from yolov7_util.utils.segment.general import process_mask
from yolov7_util.utils.segment.plots import pure_masks
from yolov7_util.utils.torch_utils import smart_inference_mode, select_device
from yolov7_util.models.common import DetectMultiBackend
from yolov7_util.utils.augmentations import letterbox

cudnn.benchmark = True 

class Yolov7Generator:
    def __init__(self, imgsz):
        check_requirements(exclude=('tensorboard', 'thop'))
        weights = ROOT / 'yolov7-seg.pt'  # model.pt path(s)
        self.imgsz = imgsz
        self.device = select_device('')
        self.model = DetectMultiBackend(weights, device=self.device, dnn=False, fp16=False)
        self.stride, self.pt = self.model.stride, self.model.pt
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size
        self.model.warmup(imgsz=(1 if self.pt else 1, 3, *imgsz))  # warmup

    @smart_inference_mode()
    def run(self,
            im0 = None,
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            classes=0,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            ):
        
        im0_shape = im0.shape
        im = cv2.flip(im0, 1)
        im = letterbox(im, self.imgsz, stride=self.stride, auto=self.pt)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        dt = (Profile(), Profile(), Profile())
    
        with dt[0]:
            im = torch.from_numpy(im).to(self.device)
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            pred, out = self.model(im, augment=augment)
            proto = out[1]

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)

        # Process predictions
        im_masks = np.zeros([self.imgsz[1], self.imgsz[0], 3])
        for i, det in enumerate(pred):  # per image
            if len(det):
                masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC

                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0_shape).round()

                # Mask plotting ----------------------------------------------------------------------------------------
                mcolors = [(255, 255, 255)]
                im_masks = pure_masks(im[i], masks, mcolors)  # image with masks shape(imh,imw,3)       
                im_masks = cv2.resize(im_masks, self.imgsz)
                im_masks = cv2.flip(im_masks, 1)
                # Mask plotting ----------------------------------------------------------------------------------------
        return im_masks