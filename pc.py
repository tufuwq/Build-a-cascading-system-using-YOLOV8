import urllib.request
import http.cookiejar
from bs4 import BeautifulSoup
import os
import time
import re
#获取图片网址
def img_list(text,text2):
    with open(text, "r", encoding="UTF") as f:
        res = f.read()
        main_page = BeautifulSoup(res, "html.parser")
        div_in = main_page.find_all("div", attrs={"class", "gallery_inner"})#需要查看对应网页图片位置
        for div_in1 in div_in:
            div_in2 = div_in1.find_all("a", attrs={"class", "imgWaper"})
            for i in range(len(div_in2)):
                urls = "https:" + div_in2[i].find("img")["data-src"] + "\n"
                with open(text2, 'at') as f:
                    f.write(urls)
def q_c(read_dir,write_dir):
    outfile = open(write_dir, "w")
    f = open(read_dir, "r")
    lines_seen = set()  # Build an unordered collection of unique elements.
    for line in f:
        line = line.strip('\n')
        if line not in lines_seen:
            outfile.write(line + '\n')
            lines_seen.add(line)
#获取目标图片地址
path_1="wq"
text_1=os.listdir(path_1)
for txt_1 in text_1:
    text=r"wq/"+txt_1
    text2=r"wq_list_txt/"+txt_1
    print(text,text2)
    img_list(text,text2)
for i,j,k in os.walk(path_1):
    print(k)
#去除重复
path_txt="wq_list_txt"
text_txt=os.listdir(path_txt)
txt_path=r"wq_list_txt/"
set_path=r"wq_list_set/"
for set_b in text_txt:
    read_dir=txt_path+set_b
    write_dir=set_path+set_b
    print(read_dir)
    print(write_dir)
    q_c(read_dir,write_dir)
path_end=r"wq\wq_1.txt"

with open(path_end, 'r', encoding='utf-8') as f:
    num=0
    for line_txt in f.readlines():
        line_txt = line_txt.strip('\n')       #去除文本中的换行符
        print(line_txt)
        name="wq_img/test/"+str(num)+".jpg"
        print(name)
        urllib.request.urlretrieve(line_txt, name)
        num+=1
