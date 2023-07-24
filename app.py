import glob
import tkinter.filedialog as fd
import logging
from skimage.feature import hog
import matplotlib.pyplot as plt
import joblib
import customtkinter as ctk
import io
import cv2
import numpy as np
import math
import os
import landmark_prediction


class FileUploader:
    def __init__(self, root):
        # Initialize the file path variables
        self.img_uploaded = False
        self.file_path = ''
        self.root = root
        self.img_uploaded_label = ctk.CTkLabel(self.root, text='', font=ctk.CTkFont(size=10))
        self.img_uploaded_label.grid(row=2, column=1, columnspan=2)

    def upload_img(self):
        try:
            self.file_path = fd.askopenfilename(filetypes=[("All files", "*.*")])
            if self.file_path:
                ext = self.file_path.split(".")[-1]
                if ext in ["jpg", "jpeg", "png", "gif"]:
                    img = cv2.imread(self.file_path)
                    self.img_uploaded = True
                    self.img_uploaded_label = ctk.CTkLabel(self.root, text='', font=ctk.CTkFont(size=10))
                    self.img_uploaded_label.grid(row=4, column=1, columnspan=2)
                    self.img_uploaded_label.configure(text='Image file successfully uploaded.')
                elif ext in ["mp4", "avi", "mpg", "mkv", "mov"]:
                    cap = cv2.VideoCapture(self.file_path)
                    success, frame = cap.read()
                    if success:
                        self.img_uploaded_label = ctk.CTkLabel(self.root, text='', font=ctk.CTkFont(size=10))
                        self.img_uploaded_label.grid(row=4, column=1, columnspan=2)
                        self.img_uploaded_label.configure(text='Video file successfully uploaded.')
                        # Convert the frame to jpg image and process like an image file
                        self.img_uploaded = True
                    else:
                        self.img_uploaded_label = ctk.CTkLabel(self.root, text='', font=ctk.CTkFont(size=10))
                        self.img_uploaded_label.grid(row=4, column=1, columnspan=2)
                        self.img_uploaded_label.configure(
                            text='The video was not uploaded successfully. Try again, please.')
                else:
                    self.img_uploaded_label.configure(text='Invalid file format, please select an image or video file.')
                create_buttons(self.root, self, self.img_uploaded, self.file_path, self.img_uploaded_label)
        except BaseException as be:
            self.img_uploaded_label.configure(text="There was an error uploading the file.")
            logging.error(be)


# process the inputted files and analyze whether the sheep is in pain
def process_data(root, file_uploader, image, data, result, button3, button4, img_uploaded_label):
    result.destroy()
    button3.destroy()
    button4.destroy()

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
    if 0.5 <= avg:
        pain_result = ctk.CTkLabel(root, text="The Sheep is in pain", font=ctk.CTkFont(size=25))
        pain_result.grid(row=5, column=1, columnspan=2, pady=25)
    else:
        pain_result = ctk.CTkLabel(root, text="The Sheep is NOT in pain", font=ctk.CTkFont(size=25))
        pain_result.grid(row=5, column=1, columnspan=2, pady=25)

    button = ctk.CTkButton(root, text="Upload Different Sheep", command=lambda: restart_upload(root, file_uploader,
                                                                                               [pain_result,
                                                                                                img_uploaded_label,
                                                                                                button]),
                                                                                                border_spacing=10)
    button.grid(row=6, column=1, columnspan=2)


# outputs a window of an image of sheep with plotted landmarks
def show_img(image, data):
    plt.imshow(image, cmap=plt.cm.gray)
    for p, q in data:
        plt.scatter(p, q, color='red')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.axis('off')
    plt.show(block=False)
    plt.draw()


# restart upload if files were uploaded incorrectly
def restart_upload(root, file_uploader, buttons):
    for b in buttons:
        b.destroy()
    plt.close()
    # Set the img_uploaded to False
    file_uploader.img_uploaded = False


def process_landmarks(image):
    # delete this when getting yaw angle & bounding box implemented
    file_name = image.split('/')[-1].split('.')[0]
    npy_path = os.path.abspath( os.path.dirname(image) +"/" + file_name + '.npy')
    try:
        landmarks = np.load(npy_path, allow_pickle=True)[1]
        yaw_angle = np.load(npy_path, allow_pickle=True)[2][2]
    except:
        landmarks = np.load(npy_path, allow_pickle=True)[2]
        yaw_angle = np.load(npy_path, allow_pickle=True)[3][2]
    # call functions for getting bounding box coordinates
    x_values = [coord[0] for coord in landmarks]
    y_values = [coord[1] for coord in landmarks]

    # Calculate min and max values
    min_x = round(min(x_values) - 10)
    max_x = round(max(x_values) + 10)
    min_y = round(min(y_values) - 10)
    max_y = round(max(y_values) + 10)

    top_left = [min_x, min_y]
    top_right = [max_x, min_y]
    bottom_right = [max_x, max_y]
    bottom_left = [min_x, max_y]

    box = [top_left, top_right, bottom_right, bottom_left]

    # call cascade forest regressor to predict landmarks
    # save them into npy and plot them over
    data = landmark_prediction.evaluate_landmarks(image, yaw_angle, box)
    img_format = image.split('.')[-1]
    if img_format in ["mp4", "avi", "mpg", "mkv", "mov"]:
        image = glob.glob('frames/frame_0000.jpg')
    else:
        image = glob.glob(image)
    image = image[0]
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    show_img(image, data)

    return image, data


# create buttons for sheep pain analysis
def create_buttons(root, file_uploader, img_uploaded, image_path, img_uploaded_label):
    if img_uploaded:
        img, data = process_landmarks(image_path)
        result = ctk.CTkLabel(root, text='Would you like to continue?', font=ctk.CTkFont(size=15))
        result.grid(row=5, column=1, columnspan=2, pady=(40, 0))
        button3 = ctk.CTkButton(root, text="Yes",
                                command=lambda: process_data(root, file_uploader, img, data, result, button3, button4,
                                                             img_uploaded_label), border_spacing=10)

        button4 = ctk.CTkButton(root, text="No", command=lambda: restart_upload(root, file_uploader,
                                                                                [result, button3, button4,
                                                                                 img_uploaded_label]),
                                border_spacing=10)
        button3.grid(row=6, column=1)
        button4.grid(row=6, column=2)


# Page classes to create a functional GUI
# all inheriting from the Base class Page
class Page(ctk.CTkFrame):
    def __init__(self, master, *args, **kwargs):
        ctk.CTkFrame.__init__(self, master, *args, **kwargs)

    def show(self):
        self.grid(sticky="nsew")
        self.lift()


# home page
class SheepPainAnalysis(Page):
    def __init__(self, master, *args, **kwargs):
        Page.__init__(self, master, *args, **kwargs)

        button1 = ctk.CTkButton(self, text="Upload Image/Video File", command=lambda: uploader.upload_img(),
                                border_spacing=10)
        button1.grid(row=2, column=1, columnspan=2, padx=140, pady=(150, 0))

        # Create an instance of the FileUploader class
        uploader = FileUploader(self)


# about page
class About(Page):
    def __init__(self, *args, **kwargs):
        Page.__init__(self, *args, **kwargs)
        # create textbox
        self.textbox = ctk.CTkTextbox(self, width=380, height=360, fg_color="#d0d0d0", wrap=ctk.WORD)
        self.textbox.grid(row=0, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.textbox.insert("0.0",
                            "About the Project\n\n"
                            + "This application serves as a desktop software that was developed by Martina Karaskova, a student at the University of Glasgow, as a part of her 4th year final project. The project was supervised by Dr. Marwa Mahmoud.  \n\n"
                            + "The software takes an input of an image or video file of a sheep and produces a result indicating whether the sheep is in pain or not. This kind of technology has the potential to improve the lives of farm animals, specifically sheep, and save farmers and veterinarian professionals a lot of time when caring for these animals. \n\n"
                            + "The application is based on the research paper \"Towards automatic monitoring of disease progression in sheep: A hierarchical model for sheep facial expressions analysis from video\" by F. Pessanha, K. McLennan and M. Mahmoud, which was presented at the 2020 15th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2020). \n\n"
                            + "This paper proposes a detailed hierarchical model for estimating pain in sheep, using fine-tuned convolution neural networks to detect sheep faces and subsequent use of CNN-based pose estimation to detect facial landmarks. It particularly focuses on the procedure of extracting facial features using Histograms of Oriented Gradients combined with geometric features in order to train a Support Vector Machine Classifier. The paper concludes with an evaluation of the classifier based on previously evaluated images of sheep in pain and shows that this approach outperforms current state-of-the-art methods. \n\n"
                            )


# application class creating the base of the app
class Application(ctk.CTkFrame):
    def __init__(self, *args, **kwargs):
        ctk.CTkFrame.__init__(self, *args, **kwargs)
        p1 = SheepPainAnalysis(self)
        p1.grid(row=0, rowspan=6, column=1)
        p2 = About(self)
        p2.grid(row=0, rowspan=6, column=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure((0, 1, 2, 3, 4, 5), weight=3)

        # Create a frame for the navbar
        navbar_frame = ctk.CTkFrame(self, width=110, corner_radius=0, fg_color='#bbb')
        navbar_frame.grid(row=0, column=0, rowspan=7, sticky="nsew")
        navbar_frame.grid_rowconfigure(5, weight=1)
        # Create the navbar buttons
        home_button = ctk.CTkButton(navbar_frame, text="Home", command=lambda: p1.show())
        home_button.grid(row=0, column=0, padx=20, pady=(130, 10))
        about_button = ctk.CTkButton(navbar_frame, text="About", command=lambda: p2.show())
        about_button.grid(row=2, column=0, pady=(10, 50))
        p1.show()


if __name__ == "__main__":
    root = ctk.CTk()
    main = Application(root)
    main.pack(side="top", fill="both", expand=True)
    root.resizable(False, False)
    root.title("SOF (Sheep Open Face)")
    root.geometry("600x400")
    root.configure(bg="white")
    ctk.set_appearance_mode("light")
    root.mainloop()