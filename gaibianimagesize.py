
######################################padding#######################################
# from PIL import Image
# import cv2
# import numpy as np
# # 打开图像
# image = Image.open("/home/lx/lx/suofang/00000.jpg")
#
# # 创建新的空白图像，尺寸为 1680x1680，背景颜色可以根据需要修改
# new_image = Image.new("RGB", (1680, 1680), (255, 255, 255))
#
# # 计算填充的偏移量，使原始图像居中
# x_offset = (1680 - 1680) // 2
# y_offset = (1680 - 1120) // 2
# x, y, width, height = 100, 200, 500, 400
# y1 = y+280
# # 将原始图像粘贴到新图像上，以居中填充
# new_image.paste(image, (x_offset, y_offset))

# new_image.save("padded_image.jpg")

# 显示填充后的图像
###################################box转换#########################
# image = cv2.imread('/home/lx/lx/suofang/00000.jpg')
# new_image=cv2.imread('/home/lx/lx/py/ori_bb/padded_image.jpg')
#
# x1 = 100
# y1 = 200
# x4= 500
# y4 = 400
# xx1 = 100
# yy1 = 480
# xx4 = 500
# yy4 = 680
# cv2.rectangle(image, (x1, y1), (x4, y4), (0, 255, 0), 2)
# cv2.rectangle(new_image, (xx1, yy1), (xx4, yy4), (0, 255, 0), 2)
# # 保存填充后的图像
# cv2.imwrite('new_image.jpg', new_image)
# cv2.imwrite('image.jpg', image)

###################################变为1024X1024#####################################
# new_size = (1024, 1024)
# resized_image = cv2.resize(new_image, new_size)
# x1 = 100
# y1 = 200
# x4= 500
# y4 = 400
# t = 1024
# xx1 = int((100*t)/1680)
# yy1 = int((480*t)/1680)
# xx4 = int((500*t)/1680)
# yy4 = int((680*t)/1680)
# cv2.rectangle(image, (x1, y1), (x4, y4), (0, 255, 0), 2)
# cv2.rectangle(resized_image, (xx1, yy1), (xx4, yy4), (0, 255, 0), 2)
# cv2.imwrite('resized_image.jpg', resized_image)
# cv2.imwrite('image.jpg', image)

############################文件夹padding##################################

# import os
# from PIL import Image
# # 输入文件夹路径
# folder_path = "/home/lx/lx/cattle_dota/images"
#
# # 输出文件夹路径
# output_folder = "/home/lx/lx/cattle_dota1/padding"
#
# # 确保输出文件夹存在
# os.makedirs(output_folder, exist_ok=True)
#
# for filename in os.listdir(folder_path):
#     if filename.endswith(('.jpg')):
#         file_path = os.path.join(folder_path, filename)
#         image = Image.open(file_path)
#         # 创建新的空白图像，尺寸为 1680x1680，背景颜色可以根据需要修改
#         new_image = Image.new("RGB", (1680, 1680), (255, 255, 255))
#         # 计算填充的偏移量，使原始图像居中
#         x_offset = (1680 - 1680) // 2
#         y_offset = (1680 - 1120) // 2
#         # 将原始图像粘贴到新图像上，以居中填充
#         new_image.paste(image, (x_offset, y_offset))
#         # 生成输出文件路径
#         output_file = os.path.join(output_folder, filename)
#         # 保存处理后的图像到输出文件夹
#         new_image.save(output_file)

        # resized_image = new_image.resize((1024,1024))
        # cv2.imwrite(os.path.join(new_folder_path,filename),resized_image)

######################################文件夹放suo############################################
# import os
# import cv2
# # 输入文件夹路径
# folder_path =  "/home/lx/lx/cattle_dota1/padding"
#
# # 输出文件夹路径
# output_folder = "/home/lx/lx/cattle_dota1/fangsuo"
#
# # 确保输出文件夹存在
# os.makedirs(output_folder, exist_ok=True)
#
# for filename in os.listdir(folder_path):
#     if filename.endswith(('.jpg')):
#         file_path = os.path.join(folder_path, filename)
#         new_image = cv2.imread(file_path)
#         resized_image = cv2.resize(new_image, (1024,1024))
#         cv2.imwrite(os.path.join(output_folder,filename),resized_image)

####################################标签缩放#################################################
import os
# folder_path = '/home/lx/lx/data1/DOTA/dota/test/labelTxt'
folder_path = '/home/lx/lx/data1/DOTA/dota/val/labelTxt'
# folder_path = '/home/lx/lx/data1/DOTA/dota/train/labelTxt'
output_folder='/home/lx/lx/cattle_dota1/fangsuo'
os.makedirs(output_folder,exist_ok=True)
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        output_file = os.path.join(output_folder,filename)
        file_path = os.path.join(folder_path,filename)
        with open (file_path,'r') as file:
            lines = file.readlines()
            with open(output_file, 'w') as files:
                for line in lines:
                    elements = line.strip().split(' ')
                    data = []
                    for ele in elements[:8]:
                        ele = float(ele)
                        ele = (ele*1024)/1680
                        data.append(ele)
                    data.append(elements[8])
                    data.append(elements[9])
                    files.write(' '.join(map(str, data)) + '\n')


