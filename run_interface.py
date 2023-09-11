import sys
import cv2
import argparse
import random
import torch
import numpy as np
import os
import torch.backends.cudnn as cudnn
import test_on_image_or_firstframe
from PyQt5.QtWidgets import *
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.datasets import letterbox
from utils.plots import plot_one_box

from PyQt5 import QtCore, QtGui, QtWidgets
from UI import background_image, background_large_window

# first page
class interface(QtWidgets.QMainWindow):
    def __init__(self,parent = None):
        super(interface, self).__init__(parent)
        self.setupUi(self)
        self.retranslateUi(self)
        self.help_page= help()
        # self.dio_window_list=[]
        # self.large_interface=large_page_with_result(None,None,None)
        self.init_slots()
        self.init_logo()
        self.timer_video = QtCore.QTimer()
        # self.init_logo()
        self.cap = cv2.VideoCapture()
        self.cap1 = cv2.VideoCapture(0)
        self.out = None
        self.window1 = No_face_detected()
        self.box_empty_status = 1
        self.window_status = 0
        self.create_output_folders()
        # self.init_result = cv2.imread('11_University-of-Glasgow-1.jpeg')
        # self.init_results = "Statistical information:"
        # self.dio = Ui_Dialog(self.init_result, self.init_results, None)
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str,
                            default='yolo_weights/1_2022_08_12best.pt', help='model.pt path(s)')
        # file/folder, 0 for webcam
        parser.add_argument('--source', type=str,
                            default='data/images', help='source')
        parser.add_argument('--img-size', type=int,
                            default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float,
                            default=0.6, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float,
                            default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='0',
                            help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--out_img_dir', type=str, default='output_img/output_image',
                            help='directory of output images')
        parser.add_argument('--out_img_bbox_dir', type=str, default='output_img/output_img_bbox',
                            help='directory of txt files contain info of boundingbox with confidence for output images')
        parser.add_argument('--out_frame_bbox_dir', type=str, default='predict_video/frame_bbox',
                            help='directory of txt files contain info of boundingbox with confidence for per frame')
        parser.add_argument(
            '--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true',
                            help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true',
                            help='save confidences in --save-txt labels')
        parser.add_argument('--nosave', action='store_true',
                            help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int,
                            help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument(
            '--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true',
                            help='augmented inference')
        parser.add_argument('--update', action='store_true',
                            help='update all models')
        parser.add_argument('--project', default='runs/detect',
                            help='save results to project/name')
        parser.add_argument('--name', default='exp',
                            help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true',
                            help='existing project/name ok, do not increment')
        self.opt = parser.parse_args()
        source, weights, view_img, save_txt, imgsz = self.opt.source, self.opt.weights, self.opt.view_img, self.opt.save_txt, self.opt.img_size
        self.opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = select_device(self.opt.device)
        self.device = self.opt.device
        self.half = self.device.type != 'cpu'
        cudnn.benchmark = True
        # Load model
        self.model = attempt_load(
            weights, device=self.device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16
        self.names = self.model.module.names if hasattr(
            self.model, 'module') else self.model.names
        # set the boundingbox colors
        self.colors = [[random.randint(0, 255)
                        for _ in range(3)] for _ in self.names]
    def create_output_folders(self):
        if not os.path.exists('output_img/output_image'):
            os.makedirs('output_img/output_image')
        if not os.path.exists('output_img/output_img_bbox'):
            os.makedirs('output_img/output_img_bbox')
        if not os.path.exists('output_img/output_img_bbox_pose'):
            os.makedirs('output_img/output_img_bbox_pose')
    def open_help(self):
        self.help_page.show()
    def open_large_interface(self,dio_window_list):
        # self.large_interface=large_page_with_result(image, info, boundingbox)
        window=dio_window_list[-1]
        window.show()
        print("windows shown")
    def no_face_detected(self, checked):
        dlg = QMessageBox(self)
        dlg.setFixedSize(100, 40)
        dlg.setWindowTitle("warning")
        dlg.setText("No detected face!")
        button = dlg.exec()
        if button == QMessageBox.Ok:
            print("OK!")
    def init_logo(self):
        self.logo.setMaximumSize(300, 40)
        pix=QtGui.QPixmap("UI/BHAI_logo")
        self.logo.setScaledContents(True)
        pix = pix.scaled(280, 40, QtCore.Qt.KeepAspectRatio)
        self.logo.setPixmap(pix)
    def button_image_open(self):
        print('button_image_open')
        name_list = []

        img_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "upload image", "", "*.JPG,*.JPEG,*.png,*.jpg,ALL Files(*)")
        if not img_name:
            return
        else:
            img = cv2.imread(img_name)
            # print(img_name)
            showimg = img
            filename = self.opt.out_img_bbox_dir + '/' + img_name.split('/')[-1].split(".")[0] + '_bbox.txt'
            with open(filename, "w") as file:
                file.truncate(0)
            with torch.no_grad():
                img = letterbox(img, new_shape=self.opt.img_size)[0]
                # Convert
                # BGR to RGB, to 3x416x416
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                # Inference
                pred = self.model(img, augment=self.opt.augment)[0]
                # print(pred)
                # Apply NMS
                pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                           agnostic=self.opt.agnostic_nms)
                # print(pred)
                # Process detections
                j=0
                for i, det in enumerate(pred):
                    if det.shape[0] != 0:
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(
                            img.shape[2:], det[:, :4], showimg.shape).round()

                        for *xyxy, conf, cls in reversed(det):
                            label = '%s %.2f' % (self.names[int(cls)], conf)
                            name_list.append(self.names[int(cls)])
                            plot_one_box(xyxy, showimg, label=label,
                                         color=self.colors[int(cls)], line_thickness=1)

                            conf_list = conf.tolist()
                            converted_list = [x.item() if isinstance(x, torch.Tensor) else x for x in xyxy]

                            bbox_list = converted_list + [conf_list]

                            print("bbox plotted")
                            with open(filename, 'a') as f:
                                f.write(' '.join(str(i) for i in bbox_list) + "\n")
                    else:
                        self.box_empty_status = 0
            image_name= img_name.split('/')[-1]
            if self.box_empty_status == 0:
                self.no_face_detected(self.window1)
                self.box_empty_status = 1
            else:
                results, landmarks_data, image_with_landmarks, self.dio_window_list = test_on_image_or_firstframe.main(img_name, 0, showimg)
                self.window_status=1
            # showimg = image_with_axis
            # print('/Users/apple/Documents/Project/Full_pipeline_software_Zejian/predict_img/'+image_name)
            #Estimate pose based on bbox
            # results,landmarks_data, image_with_axis = test_on_image_or_firstframe.main(img_name, 0)
            # crop_img = showimg[int(converted_list[1]):int(converted_list[3]),int(converted_list[0]):int(converted_list[2])]
            # crop_img = cv2.resize(crop_img, (224, 224))
            # image_with_landmarks = crop_img
            #     image_with_landmarks = showimg
            # image_with_landmarks = cv2.cvtColor(image_with_landmarks, cv2.COLOR_BGR2RGB)

                # marked_showimg=self.show_img(showimg, landmarks_data)
                marked_showimg = image_with_landmarks
                cv2.imwrite(self.opt.out_img_dir + '/' + image_name, marked_showimg)
                str_results = " ".join(results)
                # self.setText(str_results)
                self.result = cv2.cvtColor(marked_showimg, cv2.COLOR_BGR2BGRA)

                # exec('self.dio_window{} = Ui_Dialog(self.result, str_results, converted_list)'.format(j))
                # exec('self.dio_window{}={}'.format(j,Ui_Dialog(self.result, str_results, converted_list)))
                # j+=1
                self.open_large_interface(self.dio_window_list)

                # self.open_large_interface(self.result,str_results,converted_list)
                # self.dio_window= Ui_Dialog(self.result, str_results, converted_list)
                # self.result = cv2.resize(self.result, (640, 480), interpolation=cv2.INTER_AREA)
                # self.QtImg = QtGui.QImage(
                #     self.result.data, self.result.shape[1], self.result.shape[0], QtGui.QImage.Format_RGB32)
                # self.graphicsView.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
    def button_video_open(self):
        self.frameid = 0
        video_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "upload video", "", "All Files (*);;Video File (*.mp4 *.avi *.ogv *.mpg)")

        if not video_name:
            return

        flag = self.cap.open(video_name)
        if flag == False:
            QtWidgets.QMessageBox.warning(
                self, u"Warning", u"fail to open the video", buttons=QtWidgets.QMessageBox.Ok,
                defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            self.vid_name = video_name.split('/')[-1]
            self.out = cv2.VideoWriter('predict_video' + self.vid_name, cv2.VideoWriter_fourcc(
                *'MJPG'), 10, (int(self.cap.get(3)), int(self.cap.get(4))))
            # print('/Users/apple/Documents/Project/Full_pipeline_software_Zejian/predict_video/'+vid_name)
            self.cap1.set(5, 30)
            # print(self.cap1.get(5))
            self.timer_video.start(30)
            # self.video.setDisabled(True)
            self.image.setDisabled(True)

    def show_video_frame(self):
        name_list = []
        self.frameid+=1
        frameid=self.frameid
        # print(frameid)
        flag, img = self.cap.read()
        if img is not None:
            showimg = img
            filename = self.opt.out_frame_bbox_dir + '/' + self.vid_name.split('/')[-1].split(".")[0]+'frame_'+ str(frameid) + '_bbox.txt'
            with torch.no_grad():
                img = letterbox(img, new_shape=self.opt.img_size)[0]
                # Convert
                # BGR to RGB, to 3x416x416
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                # Inference
                pred = self.model(img, augment=self.opt.augment)[0]

                # Apply NMS
                pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                           agnostic=self.opt.agnostic_nms)
                # Process detections
                # print(pred[0].nelement())
                if pred[0].nelement() != 0:
                    with open(filename, "w") as file:
                        file.truncate(0)
                for i, det in enumerate(pred):  # detections per image
                    if det.shape[0]!=0:
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(
                            img.shape[2:], det[:, :4], showimg.shape).round()
                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            label = '%s %.2f' % (self.names[int(cls)], conf)
                            name_list.append(self.names[int(cls)])
                            # print(label)
                            plot_one_box(
                                xyxy, showimg, label=label, color=self.colors[int(cls)], line_thickness=2)
                # Process pose estimation
                            conf_list = conf.tolist()
                            converted_list = [x.item() if isinstance(x, torch.Tensor) else x for x in xyxy]

                            bbox_list =  converted_list + [conf_list]
                            bbox_list.insert(0,frameid)
                            print(len(bbox_list))
                            if len(bbox_list)!= 0:
                                with open(filename, 'a') as f:
                                    f.write(' '.join(str(i) for i in bbox_list) + "\n")

            self.out.write(showimg)
            show = cv2.resize(showimg, (640, 480))
            self.result = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                     QtGui.QImage.Format_RGB888)
            self.overallwindow.setPixmap(QtGui.QPixmap.fromImage(showImage))
            if os.path.exists(filename):
                results,landmarks_data, image_with_axis =test_on_image_or_firstframe.main(filename, 1, showimg)
                str_results = " ".join(results)
                self.label_2.setText(str_results)
        else:
            self.timer_video.stop()
            self.cap.release()
            self.out.release()
            self.label.clear()
            # self.video.setDisabled(False)
            self.image.setDisabled(False)
    def init_slots(self):
        self.help.clicked.connect(self.open_help)
        self.image.clicked.connect(self.button_image_open)
        # self.video.clicked.connect(self.button_video_open)
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(545, 358)
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semibold")
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        MainWindow.setFont(font)
        MainWindow.setStyleSheet("\n"
"\n"
"\n"
"#MainWindow{border-image:url(:/newPrefix/lawnbackground.jpeg);}")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setSpacing(90)
        self.verticalLayout.setObjectName("verticalLayout")
        self.title = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.title.sizePolicy().hasHeightForWidth())
        self.title.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semibold")
        font.setPointSize(14)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.title.setFont(font)
        self.title.setObjectName("title")
        self.verticalLayout.addWidget(self.title)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setContentsMargins(-1, 9, -1, 9)
        self.horizontalLayout.setSpacing(50)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.image = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.image.sizePolicy().hasHeightForWidth())
        self.image.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semibold")
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.image.setFont(font)
        self.image.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.image.setMouseTracking(False)
        self.image.setStyleSheet("QPushButton{background:rgb(255, 250, 103);border-radius:5px;}QPushButton:hover{background:rgb(255, 255, 127);}")
        self.image.setObjectName("image")
        self.image.resize(100,200)
        self.horizontalLayout.addWidget(self.image)
        # self.video = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        # sizePolicy.setHeightForWidth(self.video.sizePolicy().hasHeightForWidth())
        # self.video.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semibold")
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        # self.video.setFont(font)
        # self.video.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
#         self.video.setStyleSheet("QPushButton{background:#F7D674;border-radius:5px;}\n"
# "QPushButton:hover{background:rgb(255, 216, 117);}")
        # self.video.setObjectName("video")
        # self.horizontalLayout.addWidget(self.video)
        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 1)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setSpacing(400)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.help = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.help.sizePolicy().hasHeightForWidth())
        self.help.setSizePolicy(sizePolicy)
        self.help.setMinimumSize(QtCore.QSize(40, 40))
        self.help.setMaximumSize(QtCore.QSize(40, 37))
        self.help.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.help.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"color: rgb(255,255,255);  \n"
"border-radius: 30px;  border: 2px groove gray;\n"
"font: 9pt \"AcadEref\";\n"
"border-style: outset;\n"
"")
        self.help.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/newPrefix/Bbox_pose_Pipeline/Full_pipeline_software_Zejian/question_mark5.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.help.setIcon(icon)
        self.help.setIconSize(QtCore.QSize(32, 32))
        self.help.setObjectName("help")
        self.horizontalLayout_2.addWidget(self.help)
        self.logo = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.logo.sizePolicy().hasHeightForWidth())
        self.logo.setSizePolicy(sizePolicy)
        self.logo.setObjectName("logo")
        self.horizontalLayout_2.addWidget(self.logo)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.verticalLayout.setStretch(1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Open Sheep Face"))
        self.title.setText(_translate("MainWindow", "Sheep Face Analyzer"))
        self.image.setText(_translate("MainWindow", "Click to choose an image"))
        # self.video.setText(_translate("MainWindow", "Click to choose a video"))
        self.logo.setText(_translate("MainWindow", ""))
class No_face_detected(QWidget):
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        # self.label = QLabel("Another Window")
        self.label = QLabel("No face detected.")
        layout.addWidget(self.label)
        self.setLayout(layout)
# help page
class help(QtWidgets.QMainWindow):
    def __init__(self,parent = None):
        super(help, self).__init__(parent)
        self.setupUi(self)
        self.retranslateUi(self)
        self.init_help()
    def init_help(self):
        intro='''
                The Open Sheep Face appliction is a comprehensive,
            automated pipline for detecting and analyzing sheep
            faces by simply uploading images or videos of sheep
            with face detection, pose estimation, landmarks
            predictions and pain estimation, jointly designed by
            Marwa Mahmoud, Zejian Feng and Martina Karaskova.
              '''
        self.label.setText(intro)
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semibold")
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        font.setPointSize(10)
        self.label.setFont(font)
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(200, 400)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("About", "TextLabel"))
# large_page_with_result
class large_page_with_result(QtWidgets.QMainWindow):
    def __init__(self,image, info, boundingbox,dio_window_list, parent = None):
        super(large_page_with_result, self).__init__(parent)
        self.setupUi(self)
        self.retranslateUi(self)
        self.init_logo()
        self.results = info
        self.boundingbox = boundingbox
        self.dio_window_list=dio_window_list
        self.click_num=len(dio_window_list)
        self.result_crop_now=None
        self.results_now=None
        self.init_slots()
        self.set_info()
        if image is not None:
            if boundingbox != None:
                self.result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
                self.result_crop = self.result[int(boundingbox[1]):int(boundingbox[3]), int(boundingbox[0]):int(boundingbox[2])]
                # print(f'{int(boundingbox[1])},{int(boundingbox[3])},{int(boundingbox[0])},{int(boundingbox[2])}')
                # show = cv2.resize(self.result, (640, 480), interpolation=cv2.INTER_AREA)
                # self.result = show
            else:
                self.result = image
                show = cv2.resize(self.result, (640, 480), interpolation=cv2.INTER_AREA)
                self.result = cv2.cvtColor(show, cv2.COLOR_BGR2BGRA)

            self.show_image()
            show = cv2.resize(self.result_crop, (224, 224), interpolation=cv2.INTER_AREA)
            self.result_crop= show
            self.show_crop_image()
    def init_slots(self):
        self.next.clicked.connect(self.next_sheep_button)
        self.previous.clicked.connect(self.previous_sheep_button)
    def show_image(self):
        self.QtImg = QtGui.QImage(
            self.result.data, self.result.shape[1], self.result.shape[0], QtGui.QImage.Format_RGB32)
        self.display_window.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))


    def show_crop_image(self):
        self.QtImg = QtGui.QImage(
            self.result_crop.data, self.result_crop.shape[1], self.result_crop.shape[0], QtGui.QImage.Format_RGB32)
        self.individual.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
        if self.result_crop_now is not None:
            self.QtImg = QtGui.QImage(
                self.result_crop_now.data, self.result_crop_now.shape[1], self.result_crop_now.shape[0], QtGui.QImage.Format_RGB32)
            self.individual.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
    def set_info(self):
        if self.results_now is None:
            self.results_box.setText(self.results)
            font = QtGui.QFont()
            font.setFamily("Segoe UI Semibold")
            font.setBold(True)
            font.setItalic(True)
            font.setWeight(75)
            font.setPointSize(10)
            self.results_box.setFont(font)
        else:
            self.results_box.setText(self.results_now)
            font = QtGui.QFont()
            font.setFamily("Segoe UI Semibold")
            font.setBold(True)
            font.setItalic(True)
            font.setWeight(75)
            font.setPointSize(10)
            self.results_box.setFont(font)
    def next_sheep_button(self):
        print("next")
        win = self.dio_window_list

        if self.click_num < len(win)-1:
            self.click_num += 1
        else:
            self.click_num = 0

        self.result_crop_now= win[self.click_num].result_crop
        self.results_now="\n".join(win[self.click_num].results.split("\n")[-4:])
        print(self.click_num)
        self.show_crop_image()
        self.set_info()
    def previous_sheep_button(self):
        print("previous")
        win = self.dio_window_list

        if self.click_num==0:
            self.click_num =len(win)
        if self.click_num > 0:
            self.click_num -= 1

        self.result_crop_now= win[self.click_num].result_crop
        self.results_now = "\n".join(win[self.click_num].results.split("\n")[-4:])
        print(self.click_num)
        self.show_crop_image()
        self.set_info()
    # def previous_sheep_button(self):

    def init_logo(self):
        self.logo.setMaximumSize(300, 40)
        pix=QtGui.QPixmap("UI/BHAI_logo")
        self.logo.setScaledContents(True)
        pix = pix.scaled(280, 40, QtCore.Qt.KeepAspectRatio)
        self.logo.setPixmap(pix)
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(900, 600)
        MainWindow.setStyleSheet("QMainWindow{border-image:url(:/newPrefix/game_background.png);}")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setContentsMargins(-1, 3, -1, 3)
        self.verticalLayout.setSpacing(3)
        self.verticalLayout.setObjectName("verticalLayout")
        self.title = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.title.sizePolicy().hasHeightForWidth())
        self.title.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semibold")
        font.setPointSize(14)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.title.setFont(font)
        self.title.setObjectName("title")
        self.verticalLayout.addWidget(self.title)
        self.display_window = QtWidgets.QLabel(self.centralwidget)
        self.display_window.setObjectName("display_window")
        self.verticalLayout.addWidget(self.display_window)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setContentsMargins(-1, 6, 10, -1)
        self.horizontalLayout.setSpacing(30)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.previous = QtWidgets.QPushButton(self.centralwidget)
        self.previous.setStyleSheet("QPushButton{background:#F7D674;border-radius:5px;}QPushButton:hover{background:rgb(255, 255, 127);}\n"
"")
        self.previous.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/newPrefix/left_arrow.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.previous.setIcon(icon)
        self.previous.setIconSize(QtCore.QSize(32, 32))
        self.previous.setObjectName("previous")
        self.previous.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.horizontalLayout.addWidget(self.previous)
        self.individual = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.individual.sizePolicy().hasHeightForWidth())
        self.individual.setSizePolicy(sizePolicy)
        self.individual.setObjectName("individual")
        self.horizontalLayout.addWidget(self.individual)
        self.next = QtWidgets.QPushButton(self.centralwidget)
        self.next.setStyleSheet("QPushButton{background:#F7D674;border-radius:5px;}QPushButton:hover{background:rgb(255, 255, 127);}\n"
"")
        self.next.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/newPrefix/right_arrow.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.next.setIcon(icon1)
        self.next.setIconSize(QtCore.QSize(32, 32))
        self.next.setObjectName("next")
        self.next.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.horizontalLayout.addWidget(self.next)
        self.results_box = QtWidgets.QLabel(self.centralwidget)
        self.results_box.setObjectName("results_box")
        self.horizontalLayout.addWidget(self.results_box)
        self.logo = QtWidgets.QLabel(self.centralwidget)
        self.logo.setObjectName("logo")
        self.logo.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.horizontalLayout.addWidget(self.logo)
        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 2)
        self.horizontalLayout.setStretch(2, 1)
        self.horizontalLayout.setStretch(3, 6)
        self.horizontalLayout.setStretch(4, 2)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 3)
        self.verticalLayout.setStretch(2, 2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Open Sheep Face"))
        self.title.setText(_translate("MainWindow", "Sheep Face Analyzer"))
        self.display_window.setText(_translate("MainWindow", "TextLabel"))
        self.individual.setText(_translate("MainWindow", "TextLabel"))
        self.results_box.setText(_translate("MainWindow", "TextLabel"))
        self.logo.setText(_translate("MainWindow", "TextLabel"))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = interface()
    ui.show()
    sys.exit(app.exec_())