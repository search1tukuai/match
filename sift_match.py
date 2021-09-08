import numpy as np
import cv2
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

def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver
ap_runway={
    '1': ["pen0.jpg", "pen1.jpg"]
    }

template = []
Dst = [1, 2]
flag = [1, 2]
result = [1, 2]
for name, info in ap_runway.items():
    for i in range(0,len(info)):
        target0 = cv2.imread(r"D:\yu_project\match\template_img\pen0.jpg")
        template.append(cv2.imread("C:/yu_project/match/template_img/"+info[i]))   # queryImage
        cv2.imshow("img_", target0)
        cv2.waitKey(0)
        Dst[i],flag[i],result[i]=sift_flann_match(template[i],target0,8)
        while(flag[i]==1):
            cv2.polylines(result[i], [np.int32(Dst[i])], True, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow('img_', result[i])
            if cv2.waitKey(1) == ord('q'):
                break

# template=cv2.imread("./template_img/pen0.jpg")
# target0=cv2.imread("C:/Users/Yu/Desktop/test/0.jpg")
# target1=cv2.imread("C:/Users/Yu/Desktop/test/2.jpg")
# dst0,flag0,res0=sift_flann_match(template,target0,8)
# dst1,flag1,res1=sift_flann_match(template,target1,8)
# if(flag0==1):
#     cv2.polylines(res0, [np.int32(dst0)], True, (0, 0, 255), 1, cv2.LINE_AA)
# if(flag1==1):
#     cv2.polylines(res1, [np.int32(dst1)], True, (0, 0, 255), 1, cv2.LINE_AA)
# # print(dst1[2],dst0[2])
# # print(dst0[2]-dst0[3])
# imgstack=stackImages(1,[res0,res1])
# # cv2.putText(imgstack,"h:"+str(h),(350,100),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 180, 0), 2)
# cv2.imshow("img",imgstack)
# cv2.waitKey(0)

