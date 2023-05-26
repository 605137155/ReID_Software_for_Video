import cv2
import time
import sys
import numpy as np
import copy
import os.path as osp
import os


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

def detect(image, net):
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

def load_classes(class_path):
    class_list = []
    with open(class_path) as f:
        class_list = [cname.strip() for cname in f.readlines()]
    return class_list

def wrap_detection(input_image, output_data):
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








INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.4


time1 = time.time()
colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]
is_cuda = len(sys.argv) > 1 and sys.argv[1] == "cuda"


#目标检测模型和reid模型
root_path = os.path.dirname(os.getcwd()) #项目根目录

yolov5_path = osp.join(root_path, 'config_files', 'yolov5s.onnx')
reid_path = osp.join(root_path, 'config_files', 'resnet50.onnx')
video_path = osp.join(root_path, 'python', 'sample.mp4')
class_path = osp.join(root_path, 'config_files', 'classes.txt')
target_path = osp.join(root_path, 'python', 'target.jpg')



net = build_model(is_cuda, yolov5_path)
reidModule = reid_model(is_cuda, reid_path)
class_list = load_classes(class_path)
threshold_similarity = 0.90
#加载视频
capture = load_capture(video_path)
start = time.time_ns()
frame_count = 0
total_frames = 0
fps = -1

#视频规格
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(width)
print(height)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fpss = capture.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter('output.mp4', fourcc, fpss, (width, height))
cv2.namedWindow('output', 0)


#获取target的feature
target = cv2.imread(target_path)
target_feature = reidFeatrueExtract(target, reidModule)
target_feature_norm = target_feature/np.linalg.norm(target_feature)


#开始帧处理
start_clock = time.time()
while True:

    # Frame 将获得视频中/相机中的下一帧的(通过“cap”)
    # Ret 将从相机中获取的帧中获得返回值，要么为true，要么为false
    _, frame = capture.read()
    if frame is None:
        print("End of stream")
        break
    # 原帧的深拷贝
    f_old = copy.deepcopy(frame)
    # 目标检测(行人检测)
    inputImage = format_yolov5(frame)
    outs = detect(inputImage, net)
    class_ids, confidences, boxes = wrap_detection(inputImage, outs[0])
    frame_count += 1
    total_frames += 1
    features = []
    coordinate = []
    # 绘制行人框，且提取所有行人特征，及其对应的帧坐标box
    for (classid, confidence, box) in zip(class_ids, confidences, boxes):
        # print(box)
        # 如果是行人person的检测框，则提取该行人的特征
        if classid == 0:
            color = colors[int(classid) % len(colors)]

            a = [box[0], box[1], box[0] + box[2] - 1, box[3] + box[1] - 1]
            # print(a)
            # print(frame.shape)
            # 特征提取,并加入到person中(p1,p2 = (box[0],box[1]),(box[0]+box[2]-1,box[3]+box[1]-1) #对应左上和右下角点的二维坐标)
            # print(box)
            a = np.array(frame[box[1]:box[1] + box[3], box[0]:box[0] + box[2]])
            if a.shape[0] <= 0 or a.shape[1] <= 0:  # 非法box过滤img
                continue
            f = reidFeatrueExtract(a, reidModule)
            # print(f)
            # print(f.shape)
            # print(type(f))
            # confidence = np.dot(target_feature, f)
            f_norm = f / np.linalg.norm(f)
            cos_sim = np.dot(f_norm, target_feature_norm)


            if cos_sim > threshold_similarity:
                print(cos_sim)
                #输出人物出现时间
                p_clock = time.time()
                second = int(p_clock - start_clock)
                minute = int(second/60)
                print("{}分{}秒".format(minute, second))
                #标记任务框框
                cv2.rectangle(frame, box, color, 2)
                cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
                cv2.putText(frame, class_list[classid], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))

            features.append(f)
            # person对应坐标
            coordinate.append(box)


        else:
            continue

    # 对比识别

    if frame_count >= 30:
        end = time.time()
        fps = 1000000000 * frame_count / (end - start)
        frame_count = 0
        start = time.time_ns()

    if fps > 0:
        fps_label = "FPS: %.2f" % fps
        cv2.putText(frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 保存
    out.write(frame)

    cv2.imshow("output", frame)

    if cv2.waitKey(1) > -1:
        print("finished by user")
        break

print("Total frames: " + str(total_frames))
time2 = time.time()
print("total time: " + str(time2 - time1))
capture.release()
out.release()
cv2.destroyAllWindows()

