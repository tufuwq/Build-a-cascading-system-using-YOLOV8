import os
import shutil
import random
from tqdm import tqdm
def fenlei(img,txt,cho=0):
    os.chdir(img)
    test_frac = 0.1  # 测试集比例
    val_frac = 0.1
    random.seed(123)  # 随机数种子，便于复现

    img_path = os.listdir()
    random.shuffle(img_path)
    test_number = int(len(img_path) * test_frac)  # 测试集文件个数
    val_number = int(len(img_path) * val_frac)  # 测试集文件个数
    train_number = int(len(img_path) * (1-val_frac-test_frac))  # 测试集文件个数
    print("训练集：{}\n测试集：{}\n验证集：{}".format(train_number,test_number,val_number))
    train_files = img_path[test_number:-val_number]
    test_files = img_path[:test_number]
    val_files = img_path[-val_number:]
    os.chdir('../')
    list_use_0=['test','train','val']
    list_use_1=['images','labels']
    files=[test_files,train_files,val_files]
    for i in range(3):
        os.mkdir(list_use_0[i])
        os.chdir(list_use_0[i]+'\\')
        os.mkdir(list_use_1[0])
        os.mkdir(list_use_1[1])
        os.chdir(list_use_1[1]+'\\')
        labels_path = os.getcwd()
        for each in tqdm(files[i]):
            src_path = txt + '\\' + each.split('.')[0] + '.txt'
            shutil.move(src_path, labels_path)
        os.chdir('../')
        os.chdir(list_use_1[0]+'\\')
        images_path = os.getcwd()
        for each in tqdm(files[i]):
            src_path = img + '\\' + each
            shutil.move(src_path, images_path)
        os.chdir('../../')
    if (cho==0):
        shutil.move(txt + '\\' + "classes.txt", os.getcwd())
    os.removedirs(img)
    os.removedirs(txt)
def cg_index(path,fir,las):
    os.chdir(path)
    pa_list = os.listdir()
    for j in range(len(pa_list)):
        with open(pa_list[j], "r") as f:
            data_lines = f.readlines()
            data_temp = []
            for data in data_lines:
                i_a = data[1:len(data)]
                for pos in range(len(fir)):
                    if (data[0] == fir[pos]):
                        data_temp.append(las[pos] + i_a)
        with open(pa_list[j], "w") as f:
            for i in data_temp:
                f.writelines(i)

def choose(path_img,path_txt,newimg_name,newtxt_name,id,out_name="WAKE"):
    os.chdir(path_img)
    os.mkdir(newimg_name)
    path_new_img = os.getcwd() + '\\' + newimg_name
    os.chdir(path_txt)
    path_new_txt = os.getcwd() + '\\' + newtxt_name
    pa_list = os.listdir()
    data_img = []
    data_txt = []
    for num in range(len(pa_list)):
        with open(pa_list[num], "r") as f:
            data_lines = f.readlines()
            flag=0
            for data in data_lines:
                data_mid=data[0]
                if (data_mid == id):
                    flag=1
                else:
                    flag=0
            if(flag):
                data_img.append(path_img + "\\" + pa_list[num])
                data_txt.append(path_txt + "\\" + pa_list[num])
    for each in tqdm(data_img):
        each=each.split(".")[0]+".jpg"
        new_path = path_new_img + "\\" + each.split("\\")[-1]
        shutil.copy(each, new_path)
    os.mkdir(newtxt_name)
    for each in tqdm(data_txt):
        new_path = path_new_txt + "\\" + each.split("\\")[-1]
        shutil.copy(each, new_path)
    os.chdir("../")
    os.mkdir(out_name)
    wake_path = os.getcwd() + "\\"+out_name
    next_img=shutil.move(path_new_img, wake_path)
    next_txt=shutil.move(path_new_txt,wake_path)
    return next_img,next_txt
def disteibute(img_path,txt_path,fir,las,create=["N1","N2"]):#对图像进行分发
    next_img, next_txt = choose(img_path, txt_path, "image", "label", '1')
    fenlei(next_img, next_txt,cho=1)
    os.chdir(img_path)
    os.chdir('../')
    for num in range(len(create)):
        os.mkdir(create[num])
        create_path=os.getcwd()+"\\"+create[num]
        images_path=create_path + "\\"+"images"
        labels_path=create_path + "\\"+"labels"
        shutil.copytree(img_path,images_path)
        shutil.copytree(txt_path, labels_path)
        cg_index(labels_path,fir[num],las[num])
        fenlei(images_path,labels_path)
        os.chdir("../")


# 填自己的图片和txt路径
# path_1=r"D:\wenjian\yolo\dateset\end_use\end_use\WAKE\images"
# path_2=r"D:\wenjian\yolo\dateset\end_use\end_use\WAKE\labels"
# fenlei(path_1,path_2)
# path=r"D:\wenjian\yolo\dateset\end_use\model_choose\WAKE\val\labels"
# fir=[["0","1","2"],["0","1","2"]]#这个是原有顺序
# las=[["0","1","0"],["2","1","0"]]#这个是目标顺序

# cg_index(path,fir,las)
#进行整个的数据划分
# img_path=r"D:\wenjian\yolo\dateset\end_use\model_choose\images"
# txt_path=r"D:\wenjian\yolo\dateset\end_use\model_choose\labels"
# disteibute(img_path,txt_path,fir,las,create=["N1","N2"])

# img_path=r"D:\wenjian\yolo\dateset\end_use\hug_role_dog\images"
# txt_path=r"D:\wenjian\yolo\dateset\end_use\hug_role_dog\labels"
# #进行单独训练集的提取
# next_img, next_txt = choose(img_path, txt_path, "image", "label", '2',out_name="hug_dog")

def file_name(image_dir,txt_dir):#该函数仅为了创建数据集方便对没有标记的图片进行删除
    jpg_list = []
    txt_list = []
    les_list = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            jpg_list.append(os.path.splitext(file)[0])
    for root, dirs, files in os.walk(txt_dir):
        for file in files:
            txt_list.append(os.path.splitext(file)[0])
    print(len(jpg_list))
    # diff = set(txt_list).difference(set(jpg_list))  # 差集，在a中但不在b中的元素
    # for name in diff:
    #     print("no jpg", name + ".txt")
    diff2 = set(jpg_list).difference(set(txt_list))  # 差集，在b中但不在a中的元素
    print(len(diff2))
    for name in diff2:
        name_mid=name+".jpg"
        name_suf=os.path.join(image_dir,name_mid)
        os.remove(name_suf)
        print(name_suf)
        les_list.append(name_suf)
        # print(name_suf)
    return les_list
        # print("no txt", name + ".jpg")

def change_name(path):#更换图片名称
    os.chdir(path)
    pa_list = os.listdir()
    for i in range(len(pa_list)):
        print(path+pa_list[i])
        oldname=path+'\\'+pa_list[i]
        newname=path+'\\'+'{}.jpg'.format(i+15)
        os.rename(oldname,newname)