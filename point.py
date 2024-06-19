from ultralytics import YOLO
import os
import shutil
import torch
import multiprocessing
from multiprocessing import Process,Pipe
from matplotlib import pyplot as plt
import numpy as np
import cv2
# def get_export(pt_path,data_yaml,fp32_path,fp16_path,int8_path):
#     model = YOLO(pt_path)
#     model_name = pt_path.split("\\")[-1]
#     model_fp32 = model.export(format='engine',
#                               keras=True,
#                               dynamic=True,
#                               )
#     model_fp32_onnx = "\\".join(model_fp32.split("\\")[0:-1]) + "\\" + model_name.split('.')[0]+".onnx"
#     shutil.move(model_fp32_onnx,fp32_path)
#     shutil.move(model_fp32, fp32_path)
#     fp32_en_path = fp32_path + '\\' + model_fp32.split("\\")[-1]
#     model_half = model.export(format='engine',
#                               keras=True,
#                               half=True,
#                               dynamic=True,
#                               )
#     model_half_onnx="\\".join(model_half.split("\\")[0:-1])+"\\"+model_name.split('.')[0]+".onnx"
#     shutil.move(model_half_onnx, fp16_path)
#     shutil.move(model_half,fp16_path)
#     fp16_en_path=fp16_path+'\\'+model_half.split("\\")[-1]
#     model_int = model.export(format='engine',
#                              keras=True,
#                              int8=True,
#                              dynamic=True,
#                              data=data_yaml,
#                              )
#     int8_onnx = "\\".join(model_int.split("\\")[0:-1]) + "\\" + model_name.split('.')[0] + ".onnx"
#     shutil.move(int8_onnx, int8_path)
#     shutil.move(model_int, int8_path)
#     int8_en_path = int8_path + '\\' + model_int.split("\\")[-1]
#
#
# #分层的模型路径
# pt_path=[r"D:\wenjian\yolo\yolov8\models\train\WAKE\best.pt",
#          r"D:\wenjian\yolo\yolov8\models\train\N1\best.pt",
#          r"D:\wenjian\yolo\yolov8\models\train\N2\best.pt"]
#
# #进行int8量化所需的yaml文件
# data_yaml=[r"D:\wenjian\yolo\yolov8\models\WAKE\wake.yaml",
#            r"D:\wenjian\yolo\yolov8\models\N1\N1.yaml",
#            r"D:\wenjian\yolo\yolov8\models\N2\N2.yaml"]
# def task(in_con):
#     num = in_con.recv()
#     pt_path = [r"D:\wenjian\yolo\yolov8\models\train\WAKE\best.pt",
#                r"D:\wenjian\yolo\yolov8\models\train\N1\best.pt",
#                r"D:\wenjian\yolo\yolov8\models\train\N2\best.pt"]
#     data_yaml = [r"D:\wenjian\yolo\yolov8\models\WAKE\wake.yaml",
#                  r"D:\wenjian\yolo\yolov8\models\N1\N1.yaml",
#                  r"D:\wenjian\yolo\yolov8\models\N2\N2.yaml"]
#
#     n1_path = r"D:\wenjian\yolo\yolov8\models\N1"
#     n2_path = r"D:\wenjian\yolo\yolov8\models\N2"
#     wake_path = r"D:\wenjian\yolo\yolov8\models\WAKE"
#     fix0 = [wake_path, n1_path, n2_path]
#     fix1 = ["\\fp32", "\\fp16", "\\int8"]
#     model_use_list = []
#     for up in fix0:
#         model = []
#         for down in fix1:
#             model.append(up + down)
#         model_use_list.append(model)
#     get_export(pt_path[num], data_yaml[num], model_use_list[num][0], model_use_list[num][1], model_use_list[num][2])
# if __name__=="__main__":
#     out_con,in_con =Pipe()
#     results = []
#     for i in range(3):
#         p = Process(target=task, args=(in_con,))
#         p.start()
#         out_con.send(i)
#         p.join()
#     allocated_memory = torch.cuda.memory_allocated()
#     print("已分配的GPU内存：", allocated_memory)
#     cached_memory = torch.cuda.memory_reserved()
#     print("已缓存的GPU内存：", cached_memory)

#这是第二部分用来得到级联系统的结果

def check_level(image, WAKE, N1, N2):
    time_add=[] #用来计算总的运行时间
    wake_por = WAKE(image)
    time_add.append(sum(wake_por[0].speed.values()))
    if 0.0 in wake_por[0].boxes.cls:
        n1_por = N1(image)
        time_add.append(sum(n1_por[0].speed.values()))
        if 0.0 in n1_por[0].boxes.cls:
            n2_por = N2(image)
            time_add.append(sum(n2_por[0].speed.values()))
            return n2_por,time_add
        elif 1.0 in n1_por[0].boxes.cls:
            return n1_por,time_add
        else:
            return n1_por,time_add
    else:
        return wake_por,time_add
# def use_check_level(image, WAKE, N1, N2):
#     wake_por = WAKE(image)
#     if 0.0 in wake_por[0].boxes.cls:
#         n1_por = N1(image)
#         if 0.0 in n1_por[0].boxes.cls:
#             n2_por = N2(image)
#             return n2_por
#         elif 1.0 in n1_por[0].boxes.cls:
#             return n1_por
#         else:
#             return n1_por
#     else:
#         return wake_por

def apply_check(com_list,model_list,img_path):
    WAKE=YOLO(model_list[0][com_list[0]],task='detect')
    N1=YOLO(model_list[1][com_list[1]],task='detect')
    N2=YOLO(model_list[2][com_list[2]],task='detect')
    results = []
    result_times = []
    if(os.path.isdir(img_path)):
        img_list = os.listdir(img_path)
        img_list = [img_path + '\\' + x for x in img_list]
        for image in img_list:
            result, result_time = check_level(image, WAKE, N1, N2)
            results.append(result[0])
            result_times.append(result_time)
    elif(os.path.isfile(img_path)):
        result, result_time = check_level(img_path, WAKE, N1, N2)
        results.append(result[0])
        result_times.append(result_time)
    time = sum([sum(x) for x in result_times])
    results = [x.boxes.cls.cpu().numpy().tolist() for x in results]
    return results,time
# def apply_video(video):
#     # Open the video file
#     WAKE=YOLO(model_list[0][com_list[0]],task='detect')
#     N1=YOLO(model_list[1][com_list[1]],task='detect')
#     N2=YOLO(model_list[2][com_list[2]],task='detect')
#     video_path = video
#     cap = cv2.VideoCapture(video_path)
#     # Loop through the video frames
#     while cap.isOpened():
#         # Read a frame from the video
#         success, frame = cap.read()
#         if success:
#             results = use_check_level(frame,WAKE,N1,N2)
#             annotated_frame = results[0].plot()
#             cv2.imshow("YOLOv8 Inference", annotated_frame)
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 break
#         else:
#             break
#     cap.release()
#     cv2.destroyAllWindows()

def get_check_cls(path_file):
    os.chdir(path_file)
    path_list=os.listdir()
    cls_list=[]
    for j in range(len(path_list)):
        data_add=[]
        with open(path_list[j], "r") as f:
            data = f.readlines()
            for data_line in data:
                data_add.append(data_line[0])
        cls_list.append(data_add)
    return cls_list

def get_score_ori(list1,list2):#该函数用在检测结果与验证对象完全相等
    num=0
    for i in range(len(list1)):
        if list1[i]==list2[i]:
            num = num + 1
    return num
def get_score(list1,list2):#该函数用在验证对象与检验结果为包含关系
    num=0
    for i in range(len(list1)):
        if (len(list2[i])==0):
            if(len(list1[i])==0):
                num=num+1
        elif list2[i] == list1[i]:
            num=num+1
    return num
def get_models_list(up_use,down_use):
    models=[]
    for up in up_use:
        model=[]
        for down in down_use:
            model.append(up+down)
        models.append(model)
    return models
def get_com_list(num):
    com_lists = []
    for i in range(num):
        for j in range(num):
            for k in range(num):
                com_lists.append([i, j, k])
    return com_lists

eng_models = [r"D:\wenjian\yolo\yolov8\models\WAKE",
              r"D:\wenjian\yolo\yolov8\models\N1",
              r"D:\wenjian\yolo\yolov8\models\N2",
              ]
quants = ["\\fp32\\best.engine", "\\fp16\\best.engine", "\\int8\\best.engine"]

use_paths=[r"D:\wenjian\yolo\dateset\end_use\hug_role_dog\MODE1_MORE",
          r"D:\wenjian\yolo\dateset\end_use\hug_role_dog\MODE2_MID",
          r"D:\wenjian\yolo\dateset\end_use\hug_role_dog\MODE3_LESS"]
use_path2=["\\images","\\labels"]
def get_size(path):
    return(int(os.stat(path).st_size/1024))
def get_paixu(cc,num=1):#生成latxt所需的插入代码cc为得到的结果，num为偏置量
    for mode in range(int(len(cc) / 3)):
        num1 = mode + (num-1)*27
        num2 = mode + 27
        num3 = mode + 54
        print(str(cc[num1][2][0]) +"-"+ str(cc[num1][2][1])+ "-"+str(cc[num1][2][2]), "&", cc[num1][0] / 200,"&", int(cc[num1][1]), "&",
              str(cc[num2][2][0])+"-"+str(cc[num2][2][1])+ "-"+ str(cc[num2][2][2]), "&", cc[num2][0] / 200, "&", int(cc[num2][1]), "&",
              str(cc[num3][2][0])+ "-"+ str(cc[num3][2][1])+ "-"+ str(cc[num3][2][2]), "&", cc[num3][0] / 200,  "&", int(cc[num3][1]), "&",
              str(cc[num3][3]) + "kb",r" \\ ")
#在已经获得对应模型的路径后like path
mix_list = get_models_list(use_paths, use_path2)
model_list=get_models_list(eng_models,quants)
com_lists=get_com_list(3)#获取模型组合的结果
pt_path=[[r"D:\wenjian\yolo\yolov8\models\train\WAKE\best.pt"],
         [r"D:\wenjian\yolo\yolov8\models\train\N1\best.pt"],
         [r"D:\wenjian\yolo\yolov8\models\train\N2\best.pt"]]
model_size=[]
for path in model_list:
    model=[]
    for path2 in path:
        model.append(get_size(path2))
    model_size.append(model)
# cc=[]
# for re in cc:
#
#     size=model_size[0][re[2][0]]+model_size[1][re[2][1]]+model_size[2][re[2][2]]
#     re.append(size)

# def task(in_con):
#     com_index,img_index=in_con.recv()
#     com_list=com_lists[com_index]
#     result=[]
#     result.append(apply_check(com_list, model_list, mix_list[img_index][0]))
#     allocated_memory = torch.cuda.memory_allocated()
#     print("已分配的GPU内存：", allocated_memory)
#     cached_memory = torch.cuda.memory_reserved()
#     print("已缓存的GPU内存：", cached_memory)
#     in_con.send(result)
def post_process(results,txt_path):
    data = [list(x[0]) for x in results]
    num=len(data)
    com_list=get_com_list(3)
    list_txt = get_check_cls(txt_path)
    list_txt = [list(map(float, x)) for x in list_txt]
    scores = []
    for i in range(num):
        scores.append(get_score_ori(list_txt, data[i][0]))
        data[i].append(com_list[i])
        data[i].append(scores[i])
    for k in np.unique(scores):
        print("-----------{}-------------".format(k))
        for i in range(num):
            if (k == scores[i]):
                print(data[i][1], data[i][2])
    return [scores[-1],data[-1][1], data[-1][2]]

# if __name__=="__main__":
#
#     out_con,in_con =Pipe()
#     end_results=[]
#     for num in range(len(mix_list)):
#         results = []
#         for i in range(len(com_lists)):
#     # for num in range(2,3):
#     #     results = []
#     #     for i in range(24,27):
#             p = Process(target=task, args=(in_con,))
#             p.start()
#             out_con.send([i,num])
#             results.append(out_con.recv())
#             end_results.append(post_process(results, mix_list[num][1]))
#         p.join()
#
#     print(end_results)


#该部分为使用不同层次不同量化程度的模型进行训练

# pt_models=[r"D:\wenjian\yolo\yolov8\models\train\WAKE\best.pt",
#          r"D:\wenjian\yolo\yolov8\models\train\N1\best.pt",
#          r"D:\wenjian\yolo\yolov8\models\train\N2\best.pt"]
# eng_models = [r"D:\wenjian\yolo\yolov8\models\WAKE",
#               r"D:\wenjian\yolo\yolov8\models\N1",
#               r"D:\wenjian\yolo\yolov8\models\N2",
#               ]
# quants = ["\\fp32\\best.engine", "\\fp16\\best.engine", "\\int8\\best.engine"]
#
#
# use_paths=[r"D:\wenjian\yolo\dateset\end_use\hug_role_dog\ave_WAKE",
#           r"D:\wenjian\yolo\dateset\end_use\hug_role_dog\ave_N1",
#           r"D:\wenjian\yolo\dateset\end_use\hug_role_dog\ave_N2"]
# mix_lists=[]
# for use_path in use_paths:
#     os.chdir(use_path)
#     mix_file = [use_path + "\\" + x for x in os.listdir()]
#     mix_path = ["\\images", "\\labels"]
#     mix_list = get_models_list(mix_file, mix_path)
#     mix_lists.append(mix_list)
#
#
#
# model_list=get_models_list(eng_models,quants)
#
# def task(in_con):
#     mod_cho,model_cho,img_cho= in_con.recv()
#     ends=[]
#     use_model=YOLO(model_cho,task='detect')
#     result=use_model(mix_lists[mod_cho][img_cho][0])
#     results = [x.boxes.cls.cpu().numpy().tolist() for x in result]
#     check_txt=get_check_cls(mix_lists[mod_cho][img_cho][1])
#     list_txt = [list(map(float, x)) for x in check_txt]
#     ends.append(get_score_ori(results,list_txt))
#     allocated_memory = torch.cuda.memory_allocated()
#     print("已分配的GPU内存：", allocated_memory)
#     cached_memory = torch.cuda.memory_reserved()
#     print("已缓存的GPU内存：", cached_memory)
#     print(results)
#     print(list_txt)
#     print(ends)
#
#     in_con.send(ends)
# if __name__=="__main__":
#     out_con,in_con =Pipe()
#     results = []
#     for model_index in range(3):
#         result1=[]
#         model_list[model_index].append(pt_models[model_index])
#         for cho_model in model_list[model_index]:
#             result0=[]
#             for img_index in range(4):
#                 p = Process(target=task, args=(in_con,))
#                 p.start()
#                 out_con.send([model_index,cho_model,img_index])
#                 p.join()
#                 result0.append(out_con.recv())
#             result1.append(result0)
#         results.append(result1)
#     print(results)


