# -*- coding: utf-8 -*-

import argparse
import threading
import time


from TargetDetect.models import *  # set ONNX_EXPORT in models.py
from TargetDetect.utils.datasets import *
from TargetDetect.utils.utils import *


#Start the Thread-------------------------------------------------------------------------------------------------------
class myThread(threading.Thread):
    def __init__(self,rtsp):
        threading.Thread.__init__(self)
        self.rtsp=rtsp
        self.videoCapture = cv2.VideoCapture(self.rtsp)
        sucess, self.frame = self.videoCapture.read()

    def run(self):
        sucess, frame = self.videoCapture.read()

        while (sucess):
            sucess, frame = self.videoCapture.read()

            if not (frame is None):
                # threadLock.acquire()
                self.frame=frame
                # threadLock.release()
            # cv2.imshow('frame', frame)
            # cv2.waitKey(1)

    def get_img(self):
        threadLock.acquire()
        img=self.frame
        threadLock.release()
        return img

# Plotting functions ---------------------------------------------------------------------------------------------------
def plot_singal_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    x0=int(x[0])
    y0=int(x[1])
    x1=int(x[2])
    y1=int(x[3])
    c1, c2 = (x0, y0), (x1, y1)
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], max(0,(c1[1] - 2))), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

# Detect the Person from a frame----------------------------------------------------------------------------------------
class PersonDetect():
    def __init__(self):
        self.save_img=False
        self.view_img=True
        self.opt = self.arg_parse()
        img_size = (320, 192) if ONNX_EXPORT else self.opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
        out, source = self.opt.output, self.opt.source,
        weights, half, view_img = self.opt.weights, self.opt.half, self.opt.view_img

        # Initialize
        self.device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else self.opt.device)
        if not (os.path.exists(out)):
            # shutil.rmtree(out)  # delete output folder
            os.makedirs(out)  # make new output folder

        # Initialize model
        self.model = Darknet(self.opt.cfg, img_size)

        # Load weights
        _ = load_darknet_weights(self.model, weights)

        # Second-stage classifier
        classify = False
        if classify:
            self.modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
            self.modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model'])  # load weights
            self.modelc.to(self.device).eval()
        # Eval mode
        self.model.to(self.device).eval()

        # Export mode
        if ONNX_EXPORT:
            img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
            torch.onnx.export(self.model, img, 'weights/export.onnx', verbose=True)
            return

        # Half precision
        half = half and self.device.type != 'cpu'  # half precision only supported on CUDA
        if half:
            self.model.half()

    def arg_parse(self):
        # setup the envoriment
        parser = argparse.ArgumentParser(description='API for face-recognition')
        #general
        parser.add_argument('--cfg', type=str, default='./cfg/yolov3-spp.cfg', help='cfg file path')
        parser.add_argument('--data', type=str, default='./data/coco.data', help='coco.data file path')
        parser.add_argument('--weights', type=str, default='./weights/yolov3-spp.weights', help='path to weights file')
        parser.add_argument('--source', type=str, default='./data/samples',help='source')  # input file/folder, 0 for webcam
        # parser.add_argument('--source', type=str, default='rtsp://admin:qwer1234@192.168.20.14:554/h264/ch1/sub/av_stream',help='source')  # input file/folder, 0 for webcam
        parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
        parser.add_argument('--img-size', type=int, default=320, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
        parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
        parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
        parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
        parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        # print(parser.parse_args())
        return parser.parse_args()

    def detect(self,frame,viewImg=False):
        t0=time.time()
        img_size = (320, 192) if ONNX_EXPORT else self.opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
        out, source = self.opt.output, self.opt.source,
        weights, half, view_img = self.opt.weights, self.opt.half, self.opt.view_img
        # webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
        webcam = False
        classify = False
        flag=False

        # Set Dataloader
        im0s = frame

        # Padded resize
        img = letterbox(im0s, new_shape=img_size)[0]

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float16 if half else np.float32)  # uint8 to fp16/fp32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        # Get classes and colors
        #classes：80种分类
        #colors： 从0-255，随机选取RGB的值，共80种
        classes = load_classes(parse_data_cfg(self.opt.data)['names'])
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

        # Run inference
        #path:地址
        #img: im0s resize img
        #im0s：cv2.imread(path)
        # vid_cap=cv2.VideoCaputer(rtsp).read()
        # Get detections
        img = torch.from_numpy(img).to(self.device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = self.model(img)[0]

        if half:
            pred = pred.float()

        # Apply NMS
        #运用NMS算法，选出最优目标框
        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.nms_thres)

        # Apply
        # 运用二级分类
        #classify默认False
        if classify:
            pred = apply_classifier(pred, self.modelc, img, im0s)

        # Process detections
        personDetectResult=[]
        for i, det in enumerate(pred):  # detections per image
            # print("len(pred)：{}".format(len(pred)))
            # print("i:{}".format(i))
            # print("det:{}".format(det))
            if webcam:  # batch_size >= 1
                s, im0 = '%g: ' % i, im0s[i]
            else:
                s, im0 ='', im0s
            #svae_path保存目标路径
            #s: img_h *img_w
            # s += '%gx%g ' % img.shape[2:]  # print string

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                # print("det:{}".format(det))
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Write results
                for *xyxy, conf, _, cls in det:
                    # x0, y0, x1, y1 = *xyxy[0],
                    xTop,yTop,xBot,yBot= int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])
                    tar_class='%s' % (classes[int(cls)])
                    tar_color=colors[int(cls)]
                    targetInfo=[xTop,yTop,xBot,yBot,tar_class]
                    # print("xTop,yTop,xBot,yBot",(xTop,yTop,xBot,yBot))
                    personDetectResult.append(targetInfo)

        #             label = '%s %.2f' % (tar_class, conf)
        #             plot_singal_box(xyxy, im0, label=label, color=tar_color)
        # cv2.imshow('Img', frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return personDetectResult

if __name__ == "__main__":

    # Function 1: Face detect 1vsN
    personDetectModel = PersonDetect()
    frame = cv2.imread('./data/samples/zidane.jpg')
    match_info = personDetectModel.detect(frame,viewImg=True)

    print(match_info)

    # Function 2: Real time PersonDetect
    # rtsp='rtsp://admin:qwer1234@192.168.20.14:554/h264/ch1/sub/av_stream'
    # # rtsp = 'rtsp://admin:bj123456@192.168.10.200:554/h264/ch43/sub/av_stream'
    # threadLock = threading.Lock()
    # t1 = myThread(rtsp)
    # t1.start()
    # detect_model = PersonDetect()
    # with torch.no_grad():
    #
    #         # image=detect_model.detect()
    #         # detect_model.drawDetectImg(image)
    #     while (1):
    #         frame = t1.get_img()
    #         if frame is None:
    #             continue
    #         detect_model.detect(frame)
