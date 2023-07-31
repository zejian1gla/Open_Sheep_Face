import os
import cv2
import torch
import glob
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image
import joblib
import numpy as np
import hopenet
import math
from app import landmark_prediction
from skimage.feature import hog

import argparse
from run_interface import large_page_with_result
import csv
def parse_args(image_path, frameflag=0, img=None):
    args = argparse.Namespace()
    args.gpu = 0
    args.snapshot ='hopenet_weights/best_epoch_1.pk1'
    args.out_dir ='output_img/output_img_bbox_pose'
    args.out_dir_frame = "predict_video/frame_bbox_pose"
    args.input = image_path
    args.bboxes = f'output_img/output_img_bbox/{image_path.split("/")[-1].split(".")[0]}_bbox.txt'
    args.frame_bboxes = f'predict_video/frame_bbox/{image_path.split("/")[-1].split(".")[0]}.txt'
    args.image_path = image_path
    args.out_dir_img = "output_img/output_image"
    args.frameflag = frameflag
    args.img= img
    return args



# def parse_args():
#     """Parse input arguments"""
#     parser = argparse.ArgumentParser(description = "Head Pose estimation using the Hopenet network.")
#     parser.add_argument('--gpu',dest='gpu_id',help='GPU device id to use[0]',default=0, type=int)
#     parser.add_argument('--snapshot',dest='snapshot',help='Path of model snapshot.',default='',type=str)
#     parser.add_argument('--out_dir',dest='out_dir', help = 'Path of output dir')
#     parser.add_argument('--input',dest='image_path',help='Path of input image')
#     parser.add_argument('--bboxes', dest='bboxes', help='Bounding box annotations')
#     args = parser.parse_args()
#     return args

def main(image_name, frameflag, image_with_box, img=None):
    args = parse_args(image_name,frameflag,img)
    cudnn.enabled = True
    batch_size= 1
    gpu = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    snapshot_path = args.snapshot
    out_dir = args.out_dir
    image_path = args.image_path
    out_dir_img = args.out_dir_img
    out_dir_frame = args.out_dir_frame
    frameflag = args.frameflag
    results=[]
    images_with_landmarks=[]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # ResNet50 structure
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck,[3, 4, 6, 3], 66)
    # print('Loading snapshot.')
    # Load snapshot
    saved_state_dict = torch.load(snapshot_path,map_location= torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(saved_state_dict)
    # print('Loading data.')
    transformations = transforms.Compose([transforms.Resize(224),
                                          transforms.CenterCrop(224),transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    model.to(device)
    # model.cuda(gpu)
    print(f'pose_estimation_gpu:{torch.cuda.is_available()}')
    # print('Ready to test network.')
    # Test the model
    model.eval() # Change model to 'eval' mode (BN uses moving mean/var).
    total = 0
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor)
    idx_tensor = idx_tensor.to(device)
    # idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
    # set your output filefolder path
    if frameflag == 0:
        # txt_out = open(f'{out_dir}/{image_path.split("/")[-1]}bbox_pose.txt', "w")
        # txt_out.truncate(0)
        # txt_out.write("face_num, yaw_predicted, pitch_predicted, roll_predicted, x_min, y_min, x_max, y_max, confidence, pain_status\n")
        with open(f'{out_dir}/{image_path.split("/")[-1]}bbox_pose.csv', "w+", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["face_num", "yaw_predicted", "pitch_predicted", "roll_predicted", "x_min", "y_min", "x_max", "y_max", "boundingbox_confidence", "pain_status", "pain_score", "landmarks"])
    else:
        # txt_out = open(f'{out_dir_frame}/{image_path.split("/")[-1]}bbox_pose.txt', "w")
        # txt_out.truncate(0)
        # txt_out.write(
        #     "face_num, yaw_predicted, pitch_predicted, roll_predicted, x_min, y_min, x_max, y_max, confidence, pain_status\n")
        with open(f'{out_dir_frame}/{image_path.split("/")[-1]}bbox_pose.csv', "w+", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["face_num", "yaw_predicted", "pitch_predicted", "roll_predicted", "x_min", "y_min", "x_max", "y_max", "boundingbox_confidence", "pain_status", "pain_score", "landmarks"])
    # read bboxes file
    bboxes = args.bboxes if frameflag ==0 else args.frame_bboxes
    with open(bboxes, 'r') as f:
        bbox_lines = f.read().splitlines()
    # read image
    if frameflag == 0:
        image = cv2.imread(image_path)
        height = image.shape[0]
        width = image.shape[1]
    else:
        height = args.img.shape[0]
        width = args.img.shape[1]
    index = 0
    dio_window_list=[]
    while index < len(bbox_lines):
        line = bbox_lines[index]
        line = line.strip('\n').split(' ')
        face_num = index+1
        if True:
            x_min, y_min, x_max, y_max, conf = float(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4])
            # print(x_min, y_min, x_max, y_max, conf)
            if conf >= 0.5:
                bbox_wid = abs(x_max - x_min)
                bbox_height = abs(y_max - y_min)
                # x_min -= 2 * bbox_wid / 4
                # x_max += 2 * bbox_wid / 4
                # y_min -= 3 * bbox_height / 4
                # y_max += 3 * bbox_height / 4
                # y_max += bbox_height / 4
                # x_min = max(x_min, 0)
                # y_min = max(y_min, 0)
                # x_max = min(width, x_max)
                # y_max = min(height, y_max)

                # Crop image
                img = image[int(y_min):int(y_max),int(x_min):int(x_max)] if frameflag == 0 else args.img
                img = Image.fromarray(img)
                # Trasnform
                img= transformations(img)
                img_shape = img.size()
                img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
                img = Variable(img)
                img = img.to(device)
                # img = Variable(img).cuda(gpu)
                yaw, pitch, roll = model(img)
                yaw_predicted = F.softmax(yaw)
                pitch_predicted = F.softmax(pitch)
                roll_predicted = F.softmax(roll)
                # yaw_predicted=yaw_predicted.to(device)
                # pitch_predicted=pitch_predicted.to(device)
                # roll_predicted=roll_predicted.to(device)
                yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
                pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
                roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99
                top_left = [x_min, y_min]
                top_right = [x_max, y_min]
                bottom_right = [x_max, y_max]
                bottom_left = [x_min, y_max]

                box = [top_left, top_right, bottom_right, bottom_left]

                img_with_landmarks, landmarks_data = process_landmarks(image_path,box,yaw_predicted)
                pain_score = process_data(img_with_landmarks,landmarks_data)
                if pain_score>= 0.5:
                    pain_status = 1
                else:
                    pain_status =0
                # save results in txt file, or just pass the value of yaw_predicted to your codes
                if frameflag ==0:
                    with open(f'{out_dir}/{image_path.split("/")[-1]}bbox_pose.csv', "a", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([face_num, yaw_predicted.item(), pitch_predicted.item(), roll_predicted.item(), x_min, y_min, x_max, y_max, conf, pain_status, pain_score, landmarks_data])
                    # txt_out = open(f'{out_dir}/{image_path.split("/")[-1]}bbox_pose.txt', "a")
                    # txt_out.write(str(face_num)+' '+'%f %f %f %f %f %f %f %f %f\n' % (yaw_predicted, pitch_predicted, roll_predicted,x_min, y_min, x_max, y_max, conf, pain_status))
                else:
                    with open(f'{out_dir_frame}/{image_path.split("/")[-1]}bbox_pose.csv', "a", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([face_num, yaw_predicted.item(), pitch_predicted.item(), roll_predicted.item(), x_min, y_min, x_max, y_max, conf, pain_status, pain_score, landmarks_data])
                    # txt_out = open(f'{out_dir_frame}/{image_path.split("/")[-1]}bbox_pose.txt', "a")
                    # txt_out.write(str(face_num) + ' ' + '%f %f %f %f %f %f %f %f %f\n' % (
                    # yaw_predicted, pitch_predicted, roll_predicted, x_min, y_min, x_max, y_max, conf,pain_status))
                # if don't want to show the angles on images, just remove it.
                if frameflag == 0:
                    # image_with_axis =hopenet_utils.draw_axis(image, yaw_predicted, pitch_predicted, roll_predicted, tdx=(x_min + x_max) / 2,
                    #             tdy=(y_min + y_max) / 2, size=bbox_height / 2)
                    image_with_landmarks = show_img(image_with_box,landmarks_data)
                    # if image_with_axis is not None:
                    #     print("pose axis plotted")
                    if image_with_landmarks is not None:
                        print("landmarks plotted")
                # j = 2
                # These two lines can be removed
                if frameflag == 0:
                    cv2.imwrite(out_dir_img + '/' + image_path.split('/')[-1], image_with_landmarks)

                # else:
                #     for i in range(face_num):
                #         if not os.path.exists(f'{out_dir}/{j+i}_{image_path.split("/")[-1]}'):
                #             cv2.imwrite(f'{out_dir}/{j+i}_{image_path.split("/")[-1]}',image)
                print(f"pain score is {pain_score}")
                print(f"pain status is {pain_status}")
                # images_with_landmarks.append(image_with_landmarks)
                results.append(f"The confidence of sheep face detection is: {100*round(conf,2)}%\n")
                results.append(f"The pain probability of this sheep is: {100*round(pain_score,2)}%\n")
                if pain_status == 1:
                    results.append(f"The sheep is most likely in pain\n")
                else:
                    results.append(f"The sheep is most likely not in pain\n")

                str_results = " ".join(results)
                converted_list=[x_min,y_min,x_max,y_max]
                exec('dio_window{} = large_page_with_result(image_with_landmarks, str_results, converted_list, dio_window_list)'.format(index))
                # exec('result{} = '.format(index))
                # names = locals()
                # names['dio_window' + str(index)]
                exec('dio_window_list.append(dio_window{})'.format(index))
                # dio_window_list.append(dio_window)
                # dio_window= Ui_Dialog(image_with_landmarks, results)
        index += 1
    return results, landmarks_data, image_with_landmarks, dio_window_list
def process_landmarks(image,box,yaw_angle):
    # delete this when getting yaw angle & bounding box implemented
    # file_name = image.split('/')[-1].split('.')[0]
    # npy_path = os.path.abspath( os.path.dirname(image) +"/" + file_name + '.npy')
    # try:
    #     landmarks = np.load(npy_path, allow_pickle=True)[1]
    #     yaw_angle = np.load(npy_path, allow_pickle=True)[2][2]
    # except:
    #     landmarks = np.load(npy_path, allow_pickle=True)[2]
    #     yaw_angle = np.load(npy_path, allow_pickle=True)[3][2]
    # call functions for getting bounding box coordinates
    # x_values = [coord[0] for coord in landmarks]
    # y_values = [coord[1] for coord in landmarks]

    # Calculate min and max values
    # min_x = round(min(x_values) - 10)
    # max_x = round(max(x_values) + 10)
    # min_y = round(min(y_values) - 10)
    # max_y = round(max(y_values) + 10)
    #
    # top_left = [min_x, min_y]
    # top_right = [max_x, min_y]
    # bottom_right = [max_x, max_y]
    # bottom_left = [min_x, max_y]
    #
    # box = [top_left, top_right, bottom_right, bottom_left]

    # call cascade forest regressor to predict landmarks
    # save them into npy and plot them over
    landmarks_data = landmark_prediction.evaluate_landmarks(image, yaw_angle, box)
    img_format = image.split('.')[-1]
    if img_format in ["mp4", "avi", "mpg", "mkv", "mov"]:
        image = glob.glob('frames/frame_0000.jpg')
    else:
        image = glob.glob(image)
    image = image[0]
    image = cv2.imread(image)
    image_processed = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # show_img(image, landmarks_data)

    return image_processed, landmarks_data

def process_data(image, data):
    # angles between tips and roots of ears and distance between roots
    angles = [math.degrees(math.atan2(data[0][1] - data[1][1], data[0][0] - data[1][0])),
              math.degrees(math.atan2(data[4][1] - data[5][1], data[4][0] - data[5][0])),
              math.dist(data[1], data[5])]

    landmarks = {
        "left_ear": np.array([[data[0]], [data[21]], [data[22]], [data[1]]]),
        "right_ear": np.array([[data[5]], [data[23]], [data[24]], [data[4]]]),
        "left_eye": np.array([[data[1]], [data[2]]]),
        "right_eye": np.array([[data[4]], [data[3]]]),
        "nose": np.array(
            [[data[13]], [data[19]], [data[6]], [data[20]], [data[17]], [data[7]], [data[18]], [data[15]], [data[16]]])
    }

    regions = []
    # creating bounding box for each facial region and cropping it out
    for key, coord in landmarks.items():
        coord = np.array(coord, dtype=np.int32)
        coord = coord.reshape(-1, 1, 2)
        x, y, w, h = cv2.boundingRect(coord)
        # if coordinates negative, change to 0
        if x < 0:
            x = 0
        elif y < 0:
            y = 0
        crop_img = image[y:y + h, x:x + w]
        crop_img = cv2.resize(crop_img, (100, 100))
        fd, hog_img = hog(crop_img, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
                          visualize=True, channel_axis=-1)
        regions.append(fd)

    regions = np.array(regions).flatten()
    regions = np.array(regions, dtype=object)
    angles = np.array(angles, dtype=object)

    # combined angles and regions hog values
    combined = np.concatenate((regions, angles))
    combined = combined.reshape(1, -1)
    models = joblib.load("svm_models.pkl")
    predictions = []
    for m in models:
        predictions.append(m.predict(combined))
    avg = sum([p[0] for p in predictions]) / len(predictions)
    print(avg)
    if avg >= 0.5:
        print("The sheep is in pain")
    else:
        print("The sheep is not in pain")

    return avg

def show_img(image, data):
    # plt.imshow(image, cmap=plt.cm.gray)
    for point in data:
            # plt.scatter(p, q, color='red')
        print(point)
        p1 = int(point[0])
        p2 = int(point[1])
        image=cv2.circle(image, tuple([p1,p2]),1,(0,0,255))
    return image
    # if 0.5 <= avg:
    #     pain_result = ctk.CTkLabel(root, text="The Sheep is in pain", font=ctk.CTkFont(size=25))
    #     pain_result.grid(row=5, column=1, columnspan=2, pady=25)
    # else:
    #     pain_result = ctk.CTkLabel(root, text="The Sheep is NOT in pain", font=ctk.CTkFont(size=25))
    #     pain_result.grid(row=5, column=1, columnspan=2, pady=25)

    # button = ctk.CTkButton(root, text="Upload Different Sheep", command=lambda: restart_upload(root, file_uploader,
    #                                                                                            [pain_result,
    #                                                                                             img_uploaded_label,
    #                                                                                             button]),
    #                                                                                             border_spacing=10)
    # button.grid(row=6, column=1, columnspan=2)
    #

if __name__ == '__main__':
    main()
    #
    #
    # image = cv2.imread(image_path)
    # height = image.shape[0]
    # width = image.shape[1]
    # img = Image.fromarray(image)
    # img = transformations(img)
    # img_shape = img.size()
    # img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
    # img = Variable(img).cuda(gpu)
    # yaw, pitch, roll = model(img)
    # yaw_predicted = F.softmax(yaw)
    # pitch_predicted = F.softmax(pitch)
    # roll_predicted = F.softmax(roll)
    # yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) *3 -99
    # pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) *3 -99
    # roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) *3 -99
    # print(yaw_predicted, pitch_predicted,roll_predicted)
    # utils.draw_axis(image, yaw_predicted, pitch_predicted, roll_predicted, tdx= width/2, tdy = height/2, size =height/3)
# sh Example: py test_on_image.py --snapshot C:\Users\zejian\Downloads\output\snapshots\Good_snapshots\modelhopenet_lr_0.0005_alpha_0.0001_epoch-num_16_gpu0_batch-size_32_Sheep\_epoch_16.pk1 --input C:\Users\zejian\Downloads\test_image_input/test12.jpg --out_dir C:\Users\zejian\Downloads\test_image_output --bboxes C:\Users\zejian\Downloads\Project\Full_pipeline_software_Zejian\output_img\output_img_bbox/test12_bbox.txt




