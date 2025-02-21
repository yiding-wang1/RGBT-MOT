import os
import cv2
import pandas as pd
from ultralytics import YOLO
import glob
from tqdm import tqdm

def get_jpg_files(target_dir):
    # 拼接指定子目录的完整路径
    # target_dir = os.path.join(root_dir, sub_dir)
    # 使用 glob 模块查找指定子目录下所有的 .jpg 文件
    jpg_files = glob.glob(os.path.join(target_dir, '*.jpg'))
    # 从完整文件路径中提取文件名
    file_names = [os.path.basename(file) for file in jpg_files]
    return file_names


def get_video_list(file_path):
    result = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # 去除行末的换行符
                line = line.strip()
                # 查找最后一个下划线的位置
                index = line.rfind('_')
                if index != -1:
                    # 截取下划线前的内容
                    new_str = line[:index]
                    result.append(new_str)
                else:
                    # 如果没有下划线，直接添加整行内容
                    result.append(line)
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到，请检查文件路径。")
    except Exception as e:
        print(f"读取文件时出现异常: {e}")
    return result


def detect_objects(model, image_path, image_index):
    results = model(image_path)
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxyn[0].tolist()
            confidence = box.conf.item()
            class_id = box.cls.item()
            detections.append([image_index, x1, y1, x2, y2, confidence, class_id])
    # print(np.array(detections))
    return detections


def main_(image_dir, model):
    output_csv_path = os.path.join(image_dir, 'det', 'det.csv')

    all_detections = []
    idx=0
    for index, filename in tqdm(enumerate(os.listdir(image_dir))):
        if filename.lower().endswith('.jpg'):
            image_path = os.path.join(image_dir, filename)
            detections = detect_objects(model, image_path, idx)
            idx+=1
            all_detections.extend(detections)

    df = pd.DataFrame(all_detections, columns=['ImageIndex', 'X1', 'Y1', 'X2', 'Y2', 'Confidence', 'ClassID'])
    df.to_csv(output_csv_path, index=False)


if __name__ == "__main__":
    # 调用函数获取图片文件名数组
    def main(root_dir, second_level_dir):  # second_level_dir = visible or infrared
        if second_level_dir == "visible":
            weigth_path = '../tracking/weights/yolov8n.pt'
        elif second_level_dir == 'infrared':
            weigth_path = '../tracking/weights/yolov8n.pt'
        else:
            FileNotFoundError("Wrong second level dir.")
        model = YOLO(weigth_path)
        # 遍历指定目录下的一级子目录
        for entry in os.scandir(root_dir):
            if entry.is_dir():
                second_level_path = os.path.join(entry.path, second_level_dir)
                if os.path.isdir(second_level_path):
                    print(f"detecting:{second_level_path}")
                    main_(second_level_path, model)

    # main("E:/lasher/LasHeR_Unalined_960_0615/seleted/", "visible")
    main("E:/lasher/LasHeR_Unalined_960_0615/seleted/", "infrared")
