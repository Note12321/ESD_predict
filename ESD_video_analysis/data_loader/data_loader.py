import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

class VideoDataset:
    def __init__(self, data_path,begin,end):
        self.data_path = data_path
        self.matched_files = load_dataset_files(data_path,begin,end)
    

def load_dataset_files(self,data_path, begin, end):
    """ 获取 1~10 文件夹下以 M_ 开头的视频文件和对应的标注文件 """


    matched_files = []

    # 遍历 1~10 的文件夹
    for i in range(begin, end + 1):  # 修改为 end + 1，确保包含 'end'
        folder_path = os.path.join(data_path, str(i))  # 构造文件夹路径

        if not os.path.isdir(folder_path):  # 确保该路径是文件夹
            continue

        # 筛选文件夹中的文件并构建标注文件字典
        # txt_files = {os.path.basename(txt).replace(".txt", ""): txt for txt in os.listdir(folder_path) if txt.endswith(".txt")}

        for file in os.listdir(folder_path):
            if file.startswith("M_") and file.lower().endswith('.mp4'):  # 只筛选 M_ 开头的 mp4 文件
                video_path = os.path.join(folder_path, file)
                #video_files.append(video_path)

                base_name = file.replace("M_", "").replace(".MP4", "")  # 去掉 M_ 前缀并去掉后缀
                annotation_file_name = base_name + ".txt"  # 得到相应的 txt 文件名
                 # 拼接绝对路径
                annotation_file_path = os.path.join(folder_path, annotation_file_name)
                
                # 检查文件是否存在
                if os.path.exists(annotation_file_path):
                    matched_files.append((video_path, annotation_file_path))
                else:
                    print(f"{video_path} 视频未找到相对应的文本文件")  # 如果没有对应的标注文件，加入 None

    print(f"找到 {len(matched_files)} 对（视频-文本）文件对")
    self.matched_files = matched_files
    
    
    def load_annotation_file(annotation_file_path):
        """ 打开标注文件并获取标注信息（假设包含 Frame 和 Phase） """
        annotations = []
        try:
            # 打开并读取标注文件
            with open(annotation_file_path, 'r') as file:
                lines = file.readlines()

                # 跳过表头（如果有的话）
                if lines[0].startswith("Frame"):
                    lines = lines[1:]

                # 遍历每一行，提取标注信息
                for line in lines:
                    # 去除行尾换行符，并按空格分隔
                    parts = line.strip().split()

                    if len(parts) == 2:  # 确保每行包含 2 部分 (Frame 和 Phase)
                        try:
                            frame = int(parts[0])  # 帧号
                            phase = parts[1]  # 阶段

                            # 将解析后的数据保存为字典
                            annotations.append({
                                'frame': frame,
                                'phase': phase
                            })
                        except ValueError:
                            print(f"⚠️ 无法解析标注文件中的行: {line}")
                    else:
                        print(f"⚠️ 标注文件格式不正确: {line}")

        except FileNotFoundError:
            print(f"⚠️ 找不到文件: {annotation_file_path}")
        except Exception as e:
            print(f"⚠️ 打开标注文件时发生错误: {e}")

        return annotations
    
    
    def check_video_fps_and_frames(video_path):
        """ 检查视频的帧率和获取视频的总帧数 """
    
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
    
        if not cap.isOpened():
            print(f"无法打开视频文件: {video_path}")
            return None, None
    
        # 获取视频的帧率 (fps) 和总帧数
        fps = cap.get(cv2.CAP_PROP_FPS)  # 获取帧率
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频总帧数
    
        cap.release()  # 释放视频文件资源
    
        return fps, total_frames