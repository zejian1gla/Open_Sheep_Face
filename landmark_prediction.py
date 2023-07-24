from util import *
import hickle


def evaluate_landmarks(image, yaw, box):
    ibug_exam, rotated = read_image(image, yaw, box, True)
    ibug_exam = ibug_exam[0]
    ibug_exam_boxes = get_bounding_box(ibug_exam)
    # load the trained regressor
    yaw_angle = yaw
    print(f"yaw angle is:{yaw}")
    if yaw < 0:
        yaw_angle *= -1
    if yaw_angle <= 30:
        model = hickle.load("0-30.hkl")
        print("0-30")
    elif 30 < yaw_angle <= 60:
        model = hickle.load("30-60.hkl")
        print("30-60")
    else:
        model = hickle.load("60-90.hkl")
        print("60-90")
    init_shapes, fin_shapes = model.apply(ibug_exam, [ibug_exam_boxes[0]])

    landmarks = fin_shapes[0].points
    # rotate the landmarks back
    if rotated:
        landmarks = [[x, y] for x, y in landmarks]
    return landmarks
