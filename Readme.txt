Instructions to use Open-Sheep-Face:
1. Download the zip file , released zip file or gitclone from github link: https://github.com/zejian1gla/Open_Sheep_Face
2. Extract zip file, after that right click the folder "Open_Sheep_Face-master", select "Open folder as Pycharm Community Edition Project”.
If there is no such option, open Pycharm Community Edition and click "file->Open..",then choose the folder "Open_Sheep_Face-master".
(Pycharm Community Edition IDE is highly recommended to run this application, which is free to use, here is the link for downloading the latest version: https://www.jetbrains.com/pycharm/download, community version is at the bottom)
3. Click from menu: file -> settings, click "Python interpreter" under the Project: Open_Sheep_Face-master. Click add interpreter, you can create a virtual environment here, the option "inherit global site-packages" should not be chosen. Click save.
4. Go back to the project tree, double click on the file "requirements.txt" under the root file folder "Open_Sheep_Face-master", a message with "install requirement" option will appear at the top, install the required packages, this process can take up to 10 min.
If the message doesn't show up, click "Terminal" at the left bottom in 2023 version, right click the root folder, click "copy path/reference" -> "Absolute Path", then type in "cd path" in the terminal and press enter, where "path" should be replaced by the copied path.
After that, type in "pip install -r requirements.txt" in the terminal and press enter, then required packages will be installed.
5. Run run_interface.py, click to choose an image from the folder "test_images", then you can get the result window in seconds. Csv files are saved under the folder "Open_Sheep_Face-master\output_img\output_img_bbox_pose", where file name is the same as the test image. 

Note: if you download by git clone, the root folder name will be "Open_Sheep_Face"; if you download released zip file, the root folder name will be "Open_Sheep_Face-0.1.0-alpha"
