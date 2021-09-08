#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
from com.robot.algorithm.yolov4_airport.yolo import YOLO
from PIL import Image
import numpy as np
import torch
import cv2
import os
import time
from com.robot.algorithm.yolov4_airport.img_box_change import num_crop,img_crop,box_merge
from com.robot.algorithm.yolov4_airport.utils.utils import bbox_iou,merge_with_overlap,cropimage_with_overlap
template = []
Dst = [1, 2]
def cropimage(img,num):         #返回切割后的图片
    if num == 0:
        return img[:440, :440]
    elif num == 1:
        return img[:440, 421:860]
    elif num == 2:
        return img[:440,841:]
    elif num == 3:
        return img[220:,:440]
    elif num == 4:
        return img[220:,421:860]
    else:
        return img[220:,841:]


def merge(box,num):    #返回图片合并后的box坐标
    if num == 0:
        return box
    elif num == 1:
        box[:, 0] = box[:, 0] + 420
        box[:, 2] = box[:, 2] + 420
        return box
    elif num == 2:
        box[:, 0] = box[:, 0] + 840
        box[:, 2] = box[:, 2] + 840
        return box
    elif num == 3:
        box[:, 1] = box[:, 1] + 220
        box[:, 3] = box[:, 3] + 220
        return box
    elif num == 4:
        box[:, 0] = box[:, 0] + 420
        box[:, 2] = box[:, 2] + 420
        box[:, 1] = box[:, 1] + 220
        box[:, 3] = box[:, 3] + 220
        return box
    else:
        box[:, 0] = box[:, 0] + 840
        box[:, 2] = box[:, 2] + 840
        box[:, 1] = box[:, 1] + 220
        box[:, 3] = box[:, 3] + 220
        return box

def sift_flann_match(template_img,target_img,MIN_MATCH_COUNT):
    counter = []
    dst=[]
    flag=1
    result_img=target_img.copy()
    template=cv2.cvtColor(template_img,cv2.COLOR_BGR2GRAY)
    target=cv2.cvtColor(target_img,cv2.COLOR_BGR2GRAY)
    # 基于FLANN的匹配器(FLANN based Matcher)定位图片
    # Initiate SIFT detector创建sift检测器
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(template, None)
    kp2, des2 = sift.detectAndCompute(target, None)
    # 创建设置FLANN匹配
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    # 舍弃大于0.7欧氏距离的匹配
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    if len(good) > MIN_MATCH_COUNT:
        # 获取关键点的坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # 计算变换矩阵和MASK
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w = template.shape
        # 使用得到的变换矩阵对原图像的四个角进行变换，获得在目标图像上对应的坐标
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        # cv2.polylines(result_img, [np.int32(dst)], True, (0, 0, 255), 1, cv2.LINE_AA)
    else:
        # print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        counter.append(len(good)/MIN_MATCH_COUNT)
        matchesMask = None
        flag=0
        print(counter)
    return dst,flag,result_img

def draw_runway(name,img,Dst):

    if(name=='huang hua airport'):
        dst=Dst[0]
        k1 = dst[0] + 0.5 * (dst[3] - dst[0])
        k2 = dst[1] + 0.5 * (dst[2] - dst[1])
        ka = (k1, k2, dst[1], dst[0])
        cv2.polylines(img, [np.int32(ka)], True, (0, 0, 255), 1, cv2.LINE_AA)
        k1 = dst[0] + 0.5 * (dst[3] - dst[0])
        k2 = dst[1] + 0.5 * (dst[2] - dst[1])
        kb = (k1, k2, dst[2], dst[3])
        cv2.polylines(img, [np.int32(kb)], True, (0, 0, 255), 1, cv2.LINE_AA)
    if(name=='wu han airport'):
        dst = Dst[0]
        ka1 = dst[0] + 0.12 * (dst[3] - dst[0])
        ka2 = dst[1] + 0.12 * (dst[2] - dst[1])
        ka = (ka1, ka2, dst[1], dst[0])
        cv2.polylines(img, [np.int32(ka)], True, (0, 0, 255), 1, cv2.LINE_AA)
        kb1 = dst[0] + 0.88 * (dst[3] - dst[0])
        kb2 = dst[1] + 0.88 * (dst[2] - dst[1])
        kb = (kb1, kb2, dst[2], dst[3])
        cv2.polylines(img, [np.int32(kb)], True, (0, 0, 255), 1, cv2.LINE_AA)
    if(name=='xi an airport'):
        dst = Dst[0]
        ka1 = dst[0] + 0.12 * (dst[3] - dst[0])
        ka2 = dst[1] + 0.12 * (dst[2] - dst[1])
        ka = (ka1, ka2, dst[1], dst[0])
        cv2.polylines(img, [np.int32(ka)], True, (0, 0, 255), 1, cv2.LINE_AA)
        kb1 = dst[0] + 0.88 * (dst[3] - dst[0])
        kb2 = dst[1] + 0.88 * (dst[2] - dst[1])
        kb = (kb1, kb2, dst[2], dst[3])
        cv2.polylines(img, [np.int32(kb)], True, (0, 0, 255), 1, cv2.LINE_AA)
    if(name=='chong qing airport'):
        dst = Dst[0]
        ka2 = dst[1] + 0.15 * (dst[2] - dst[1])
        ka3 = dst[0] + 0.37 * (dst[1] - dst[0])
        ka1 = ka3 + 0.15 * (dst[3] - dst[0])
        ka = (ka1, ka2, dst[1], ka3)
        cv2.polylines(img, [np.int32(ka)], True, (0, 0, 255), 1, cv2.LINE_AA)
        kb1 = ka3 + 0.3 * (dst[3] - dst[0])
        kb2 = dst[1] + 0.3 * (dst[2] - dst[1])
        kb = (ka1, ka2, kb2, kb1)
        cv2.polylines(img, [np.int32(kb)], True, (0, 0, 255), 1, cv2.LINE_AA)
        kc1 = dst[0] + 0.88 * (dst[3] - dst[0])
        kc3 = dst[3] + 0.72 * (dst[1] - dst[0])
        kc2 = kc3 - 0.12 * (dst[2] - dst[1])
        kc = (kc1, kc2, kc3, dst[3])
        cv2.polylines(img, [np.int32(kc)], True, (0, 0, 255), 1, cv2.LINE_AA)
    if(name=='shou du airport'):
        dst = Dst[0]
        ka1 = dst[0] + 0.12 * (dst[3] - dst[0])
        ka3 = dst[0] + 0.58 * (dst[1] - dst[0])
        ka2 = ka3 + 0.12 * (dst[3] - dst[0])
        ka = (ka1, ka2, ka3, dst[0])
        cv2.polylines(img, [np.int32(ka)], True, (0, 0, 255), 1, cv2.LINE_AA)
        kb2 = dst[2] - 0.37 * (dst[3] - dst[0])
        kb1 = kb2 - 0.12 * (dst[3] - dst[0])
        kb0 = kb1 - 0.75 * (dst[1] - dst[0])
        kb3 = kb0 + 0.12 * (dst[3] - dst[0])
        kb = (kb0, kb1, kb2, kb3)
        cv2.polylines(img, [np.int32(kb)], True, (0, 0, 255), 1, cv2.LINE_AA)
        kc3 = dst[2] - 0.78 * (dst[1] - dst[0])
        kc1 = kc3 - 0.10 * (dst[3] - dst[0])
        kc2 = kc1 + 0.72 * (dst[1] - dst[0])
        kc4 = kc2 + 0.12 * (dst[3] - dst[0])
        kc = (kc1, kc2, kc4, kc3)
        cv2.polylines(img, [np.int32(kc)], True, (0, 0, 255), 1, cv2.LINE_AA)
    if(name=='da xing airport'):
        dst1=Dst[0]
        dst2=Dst[1]
        ka0 = dst1[0] + 0.07 * (dst1[3] - dst1[0])
        ka0 = ka0 + 0.32 * (dst1[1] - dst1[0])
        ka1 = ka0 + 0.6 * (dst1[1] - dst1[0])
        ka2 = ka1 + 0.04 * (dst1[3] - dst1[0])
        ka3 = ka0 + 0.04 * (dst1[3] - dst1[0])
        ka = (ka0, ka1, ka2, ka3)
        cv2.polylines(img, [np.int32(ka)], True, (0, 0, 255), 1, cv2.LINE_AA)
        kb0 = dst1[0] + 0.24 * (dst1[3] - dst1[0])
        kb0 = kb0 + 0.09 * (dst1[1] - dst1[0])
        kb1 = kb0 + 0.6 * (dst1[1] - dst1[0])
        kb2 = kb1 + 0.04 * (dst1[3] - dst1[0])
        kb3 = kb0 + 0.04 * (dst1[3] - dst1[0])
        kb = (kb0, kb1, kb2, kb3)
        cv2.polylines(img, [np.int32(kb)], True, (0, 0, 255), 1, cv2.LINE_AA)
        kc0 = kb0 + 0.075 * (dst1[3] - dst1[0])
        kc1 = kb1 + 0.075 * (dst1[3] - dst1[0])
        kc2 = kc1 + 0.04 * (dst1[3] - dst1[0])
        kc3 = kc0 + 0.04 * (dst1[3] - dst1[0])
        kc = (kc0, kc1, kc2, kc3)
        cv2.polylines(img, [np.int32(kc)], True, (0, 0, 255), 1, cv2.LINE_AA)
        kd0 = dst1[0] + 0.55 * (dst1[3] - dst1[0])
        kd0 = kd0 + 0.42 * (dst1[1] - dst1[0])
        kd1 = kd0 + 0.56 * (dst1[1] - dst1[0])
        kd2 = kd1 + 0.04 * (dst1[3] - dst1[0])
        kd3 = kd0 + 0.04 * (dst1[3] - dst1[0])
        kd = (kd0, kd1, kd2, kd3)
        cv2.polylines(img, [np.int32(kd)], True, (0, 0, 255), 1, cv2.LINE_AA)

        ke0 = dst2[0] + 0.06 * (dst2[3] - dst2[0])
        ke1 = ke0 + 0.85 * (dst2[1] - dst2[0])
        ke2 = ke1 + 0.06 * (dst2[3] - dst2[0])
        ke3 = ke0 + 0.06 * (dst2[3] - dst2[0])
        ke = (ke0, ke1, ke2, ke3)
        cv2.polylines(img, [np.int32(ke)], True, (0, 0, 255), 1, cv2.LINE_AA)
    if (name == 'jia xing airport'):
        dst = Dst[0]
        ka0=dst[0]+0.27*(dst[3]-dst[0])
        ka1=dst[1]+0.27*(dst[2]-dst[1])
        ka2 = ka1 + 0.17 * (dst[2] - dst[1])
        ka3 = ka0 + 0.17 * (dst[3] - dst[0])
        ka = (ka0,ka1, ka2, ka3)
        cv2.polylines(img, [np.int32(ka)], True, (0, 0, 255), 1, cv2.LINE_AA)
    if (name == 'xi yuan airport'):
        dst = Dst[0]
        ka0 = dst[0] + 0.6 * (dst[3] - dst[0])
        ka1 = dst[1] + 0.6 * (dst[2] - dst[1])
        ka2 = ka1 + 0.12 * (dst[2] - dst[1])
        ka3 = ka0 + 0.12 * (dst[3] - dst[0])
        ka = (ka0, ka1, ka2, ka3)
        cv2.polylines(img, [np.int32(ka)], True, (0, 0, 255), 1, cv2.LINE_AA)
    if (name == 'shao dong airport'):
        dst = Dst[0]
        ka0 = dst[0] + 0.15 * (dst[3] - dst[0])
        ka1 = ka0 + 0.92 * (dst[1] - dst[0])
        ka2 = ka1 + 0.14 * (dst[2] - dst[1])
        ka3 = ka0 + 0.14 * (dst[3] - dst[0])
        ka = (ka0, ka1, ka2, ka3)
        cv2.polylines(img, [np.int32(ka)], True, (0, 0, 255), 1, cv2.LINE_AA)
    if (name == 'lai yang airport'):
        dst = Dst[0]
        ka0 = dst[3] - 0.17 * (dst[3] - dst[0])
        ka1 = dst[2] - 0.17 * (dst[2] - dst[1])
        ka2 = ka1 + 0.11 * (dst[2] - dst[1])
        ka3 = ka0 + 0.11 * (dst[3] - dst[0])
        ka = (ka0, ka1, ka2, ka3)
        cv2.polylines(img, [np.int32(ka)], True, (0, 0, 255), 1, cv2.LINE_AA)
    if (name == 'lu shan airport'):
        dst = Dst[0]
        cv2.polylines(img, [np.int32(ka)], True, (0, 0, 255), 1, cv2.LINE_AA)
    if (name == 'ling bei airport'):
        dst = Dst[0]
        ka2 = dst[2] - 0.13 * (dst[2] - dst[1])
        ka3 = dst[3] - 0.13 * (dst[3] - dst[0])
        ka0 = ka3 - 0.26 * (dst[3] - dst[0])
        ka1 = ka2 - 0.26 * (dst[2] - dst[1])
        ka = (ka0, ka1, ka2, ka3)
        cv2.polylines(img, [np.int32(ka)], True, (0, 0, 255), 1, cv2.LINE_AA)
    return img

ap_runway={
    'chong qing airport':["jb_o1.png"]
    ,
    'huang hua airport':["hh_o1.png"]
    ,
    'wu han airport': ["th_o1.png"]
    ,
    'xi an airport': ["xy_o1.png"]
    ,
    'shou du airport': ["sd_o1.png"]
    ,
    'da xing airport': ["dx_o1.png","dx_o2.png"]
    ,
    'jia xing airport':["jx_o1.png"]
    ,
    'lai yang airport':["ly_o1.png"]
    ,
    'lu shan airport':["ls_o1.png"]
    ,
    'shao dong airport':["sy_o1.png"]
    ,
    'xi yuan airport':["xy_o1.png"]
    ,
    'ling bei airport':["lb_o1.png"]
    }


threshold = 0.1
yolo = YOLO()
# img = input('Input image filename:')
# image = Image.open(img)
# image_pre = np.array(image.convert("RGB"))    #此时变为opencv可读
image_pre = cv2.imread("C:/Users/user/Desktop/test/airport/552.png")
box_all = []
h_num,w_num = num_crop(image_pre)            #显示h和w上切割的次数
print("图片大小为:{}".format(image_pre.shape))   #h,w,3
for num in range(h_num*w_num):
    #image_crop = cropimage_with_overlap(image_pre,num)      #将图片按序号进行切割，注意不要改变原来的image  此时已经时array (h,w,3)
    image_crop,x_start,y_start = img_crop(image_pre,num,h_num,w_num)
    #print("x_start:{}  y_start:{}".format(x_start,y_start))
    r_image = yolo.detect_image(image_crop)
    measurevalue = np.zeros((yolo.measure.shape[0], yolo.measure.shape[1]))
    measurevalue[:, 0] = yolo.measure[:, 1]  # yolo.measure中的坐标值为 y1,x1,y2,x2    ---> x1,y1,x2,y2,conf
    measurevalue[:, 1] = yolo.measure[:, 0]
    measurevalue[:, 2] = yolo.measure[:, 3]
    measurevalue[:, 3] = yolo.measure[:, 2]
    measurevalue[:, 4] = yolo.measure[:, 4]
    measurevalue[:, 5] = yolo.measure[:, 5]
    #box_changed = merge(measurevalue,num)   # n,5

    #box_changed = merge_with_overlap (measurevalue, num)
    box_changed = box_merge(measurevalue,x_start,y_start)
    box_changed_torch = torch.from_numpy(box_changed)
    if num == 0:
        box_all = box_changed_torch
        continue
    #box_changed_torch与box_all作iou运算
    box_all_copy = box_all   #新建一个所有物体的框用来与新的框计算IOU
    while box_changed_torch.shape[0]>0:
        result = bbox_iou(box_changed_torch[0,:4].unsqueeze(0),box_all_copy[:,:4])
        #还需要加入一个判断，如果相邻两个图片的有交集，但是IOU不大，且面积相近，认为时同一个物体
        if result[result>threshold].shape[0] == 0:  #说明该框与之前所有框重叠率均小于阈值
            box_all = torch.cat((box_all,box_changed_torch[0].unsqueeze(0)),0)  #把当前框加入到box_all中
            #进行判断，选择面积较大的框，滤除面积较小的框
        else:
            _,max_index =  torch.max(result,0)
            #print("max_index",max_index)
            #print(result[result>threshold])
            area1 = (box_all[max_index][2] - box_all[max_index][0]) * (box_all[max_index][3] - box_all[max_index][1])          #有最大IOU的面积
            area2 = (box_changed_torch[0][2] - box_changed_torch[0][0]) *(box_changed_torch[0][3] - box_changed_torch[0][1])    #当前框的面积

            #if area2 > area1 and box_changed_torch[0][4] == box_all[max_index][4]:
            if area2 > area1:
            #if box_changed_torch[0][4] > box_all[max_index][4]:
                box_all[max_index] = box_changed_torch[0]
                box_all_copy[max_index] = box_changed_torch[0]
        box_changed_torch = box_changed_torch[1:]      #每次减少一个
    #print("box_all的长度",box_all)
    #r_image.show()    #对每一块进行展示
#此时box_all为所有符合条件的框  x1,y1,x2,y2,confidence
#|------------------------------------------------------------------------------------------------------------
#需要对box_all进行筛选将不符合的框去掉  box_all为torch.tensor
mask = torch.ones(len(box_all))
#1.左上角坐标超过最大值，右下角坐标小于最小值
for i in range(len(box_all)):
    if box_all[i][0] > image_pre.shape[1] or box_all[i][1] > image_pre.shape[0] or box_all[i][2] < 0 or \
            box_all[i][3] < 0:
        mask[i] = 0
box_all = box_all[mask == 1]  # 第一次筛选
mask = torch.ones(len(box_all))  # 开始第二次筛选
box_all[box_all[:, 0] < 0, 0] = 0
box_all[box_all[:, 1] < 0, 1] = 0
box_all[box_all[:, 2] > image_pre.shape[1], 2] = image_pre.shape[1]
box_all[box_all[:, 3] > image_pre.shape[0], 3] = image_pre.shape[0]
mask[box_all[:, 2] < 10] = 0
mask[box_all[:, 3] < 10] = 0
mask[image_pre.shape[1] - box_all[:, 0] < 20] = 0
mask[image_pre.shape[0] - box_all[:, 1] < 20] = 0
box_all = box_all[mask == 1]

#-------------------------------------------------------------------------------------------------------------|
#print("box_all", box_all)
for i in range(len(box_all)):
        x1 = int(box_all[i][0])
        y1 = int(box_all[i][1])
        x2 = int(box_all[i][2])
        y2 = int(box_all[i][3])
        cv2.rectangle(image_pre,(x1,y1),(x2,y2),(0,255,0),thickness=2)
        # if box_all[i][5]==0:
        #     label = "shaoyang airport"
        # elif box_all[i][5]==1:
        #     label = "changsha airport"
        # elif box_all[i][5]==2:
        #     label = "beijing airport"
        # elif box_all[i][5]==3:
        #     label = "wuhan airport"
        # else:
        #     label = "xian airport"
        if box_all[i][5] == 0:
            label = "jia xing airport"  # 嘉兴机场
        elif box_all[i][5] == 1:
            label = "lai yang airport"  # 莱阳机场
        elif box_all[i][5] == 2:
            label = "lu shan airport"  # 九江庐山机场
        elif box_all[i][5] == 3:
            label = "shao dong airport"  # 邵东军民合用机场
        elif box_all[i][5] == 4:
            label = "ling bei airport"  # 陵北机场
        elif box_all[i][5] == 5:
            label = "xi yuan airport"  # 北京西苑机场
        elif box_all[i][5] == 6:
            label = "huang hua airport"  # 长沙黄花国际机场
        elif box_all[i][5] == 7:
            label = "chong qing airport"  # 重庆江北国际机场
        elif box_all[i][5] == 8:
            label = "da xing airport"  # 北京大兴机场
        elif box_all[i][5] == 9:
            label = "shou du airport"  # 首都国际机场
        elif box_all[i][5] == 10:
            label = "xi an airport"  # 西安咸阳国际机场
        elif box_all[i][5] == 11:
            label = "wu han airport"  # 武汉天河国际机场
        else:
            label = "无法识别"
        if image_pre.shape[1] - x1 <100 and y1 <25:
            cv2.putText(image_pre, '{}:{:.3f}'.format(label, box_all[i][4]), (image_pre.shape[1] - 110, y1 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            continue
        if image_pre.shape[1] - x1 > 100:
            cv2.putText(image_pre, '{}:{:.3f}'.format(label, box_all[i][4]), (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(image_pre, '{}:{:.3f}'.format(label, box_all[i][4]), (image_pre.shape[1] - 110, y1 - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#跑道检测

for name ,info in ap_runway.items():
    if name==label:
        for i in range(0,len(info)):
            target=image_pre.copy()
            template.append(cv2.imread("match_img/"+info[i]))   # queryImage
            Dst[i],flag,result=sift_flann_match(template[i],target,8)
        if(flag==1):
            draw_runway(label,result,Dst)
    elif(ap_runway.get(label)==None):
        print("sorry,The airport you search is not in the list, we are doing related work")
cv2.imshow("img_",result)
cv2.waitKey(0)


