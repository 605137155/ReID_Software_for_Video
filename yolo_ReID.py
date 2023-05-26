import cv2
import numpy as np

# if __name__ == "__main__":
#
# 读取目标检测网络模型yolov5s
def build_model(is_cuda, model_path):
    net = cv2.dnn.readNet(model_path
        )
    if is_cuda:
        print("Attempty to use CUDA")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    else:
        print("Running on CPU")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

# 读取行人重识别模型-resnet50
def reid_model(is_cuda, model_path):
    net = cv2.dnn.readNet(
        model_path)
    if is_cuda:
        print("Attempty to use CUDA")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    else:
        print("Running on CPU")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

def detect(image, net, INPUT_WIDTH, INPUT_HEIGHT):
    # 图像帧预处理
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    # 设置网络
    net.setInput(blob)
    # 运算
    preds = net.forward()
    return preds

def reidFeatrueExtract(image, net):
    # 图像预处理
    # print(image.shape)
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (256, 128), swapRB=True, crop=False)
    net.setInput(blob)
    feature = np.resize(net.forward(), 2048)
    return feature

def load_capture(video_path):
    capture = cv2.VideoCapture(
        video_path)
    return capture

#判断文件是否是视频
def is_video_file(file_path):
    video_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.flv')
    return file_path.lower().endswith(video_extensions) and cv2.VideoCapture(file_path).isOpened()

#判断文件是否是图片
def is_image_file(file_path):
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    return file_path.lower().endswith(image_extensions) and cv2.imread(file_path) is not None

def load_classes(class_path):
    class_list = []
    with open(class_path) as f:
        class_list = [cname.strip() for cname in f.readlines()]
    return class_list

def wrap_detection(input_image, output_data, INPUT_WIDTH, INPUT_HEIGHT):
    class_ids = []
    confidences = []
    boxes = []
    rows = output_data.shape[0]
    image_width, image_height, _ = input_image.shape
    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT

    for r in range(rows):
        row = output_data[r]
        confidence = row[4]
        if confidence >= 0.4:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .25):
                confidences.append(confidence)
                class_ids.append(class_id)
                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)
    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    return result_class_ids, result_confidences, result_boxes

def format_yolov5(frame):
    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result

