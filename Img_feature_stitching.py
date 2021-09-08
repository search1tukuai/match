# # coding: utf-8
# import numpy as np
# import cv2
#
# leftgray = cv2.imread('desk1.jpg')
# rightgray = cv2.imread('desk2.jpg')
# hessian = 400
# surf = cv2.xfeatures2d.SURF_create(hessian)  # 将Hessian Threshold设置为400,阈值越大能检测的特征就越少
# kp1, des1 = surf.detectAndCompute(leftgray, None)  # 查找关键点和描述符
# kp2, des2 = surf.detectAndCompute(rightgray, None)
#
# FLANN_INDEX_KDTREE = 0  # 建立FLANN匹配器的参数
# indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)  # 配置索引，密度树的数量为5
# searchParams = dict(checks=50)  # 指定递归次数
# # FlannBasedMatcher：是目前最快的特征匹配算法（最近邻搜索）
# flann = cv2.FlannBasedMatcher(indexParams, searchParams)  # 建立匹配器
# matches = flann.knnMatch(des1, des2, k=2)  # 得出匹配的关键点
#
# good = []
# # 提取优秀的特征点
# for m, n in matches:
#     if m.distance < 0.7 * n.distance:  # 如果第一个邻近距离比第二个邻近距离的0.7倍小，则保留
#         good.append(m)
# src_pts = np.array([kp1[m.queryIdx].pt for m in good])  # 查询图像的特征描述子索引
# dst_pts = np.array([kp2[m.trainIdx].pt for m in good])  # 训练(模板)图像的特征描述子索引
# H = cv2.findHomography(src_pts, dst_pts)  # 生成变换矩阵
# h, w = leftgray.shape[:2]
# h1, w1 = rightgray.shape[:2]
# shft = np.array([[1.0, 0, w], [0, 1.0, 0], [0, 0, 1.0]])
# M = np.dot(shft, H[0])  # 获取左边图像到右边图像的投影映射关系
# dst_corners = cv2.warpPerspective(leftgray, M, (w * 2, h))  # 透视变换，新图像可容纳完整的两幅图
# cv2.imshow('tiledImg1', dst_corners)  # 显示，第一幅图已在标准位置
# dst_corners[0:h, w:w * 2] = rightgray  # 将第二幅图放在右侧
# # cv2.imwrite('tiled.jpg',dst_corners)
# cv2.imshow('tiledImg', dst_corners)
# cv2.imshow('leftgray', leftgray)
# cv2.imshow('rightgray', rightgray)
# cv2.waitKey()
# cv2.destroyAllWindows()


import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

MIN = 10
starttime = time.time()
img1 = cv2.imread('2.bmp')  # query
img2 = cv2.imread('1.bmp')  # train
img1 = cv2.resize(img1, (720, 540))
img2 = cv2.resize(img2, (720, 540))

# stitcher = cv2.createStitcher()
# _result, pano = stitcher.stitch((img1, img2))
# cv2.imshow('pano', pano)
# cv2.waitKey(0)

# img1gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
# # img2gray=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
surf = cv2.xfeatures2d.SURF_create(4000, nOctaves=4, extended=False, upright=True)
#surf = cv2.xfeatures2d.SURF_create(400, nOctaves=4, extended=False, upright=True)
# surf=cv2.xfeatures2d.SIFT_create()#可以改为SIFT
kp1, descrip1 = surf.detectAndCompute(img1, None)
kp2, descrip2 = surf.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 0
indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
searchParams = dict(checks=50)
flann = cv2.FlannBasedMatcher(indexParams, searchParams)
match = flann.knnMatch(descrip1, descrip2, k=2)

good = []
for i, (m, n) in enumerate(match):
    if (m.distance < 0.7 * n.distance):
        good.append(m)

if len(good) > MIN:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    ano_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, ano_pts, cv2.RANSAC, 5.0)
    warpImg = cv2.warpPerspective(img2, np.linalg.inv(M), (img1.shape[1] + img2.shape[1], img2.shape[0]))
    direct = warpImg.copy()
    direct[0:img1.shape[0], 0:img1.shape[1]] = img1
    simple = time.time()

    # cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    # cv2.imshow("Result",warpImg)
    rows, cols = img1.shape[:2]

    for col in range(0, cols):
        if img1[:, col].any() and warpImg[:, col].any():  # 开始重叠的最左端
            left = col
            break
    for col in range(cols - 1, 0, -1):
        if img1[:, col].any() and warpImg[:, col].any():  # 重叠的最右一列
            right = col
            break

    res = np.zeros([rows, cols, 3], np.uint8)
    for row in range(0, rows):
        for col in range(0, cols):
            if not img1[row, col].any():  # 如果没有原图，用旋转的填充
                res[row, col] = warpImg[row, col]
            elif not warpImg[row, col].any():
                res[row, col] = img1[row, col]
            else:
                srcImgLen = float(abs(col - left))
                testImgLen = float(abs(col - right))
                alpha = srcImgLen / (srcImgLen + testImgLen)
                res[row, col] = np.clip(img1[row, col] * (1 - alpha) + warpImg[row, col] * alpha, 0, 255)

    warpImg[0:img1.shape[0], 0:img1.shape[1]] = res
    final = time.time()
    cv2.imshow("simp", direct)
    cv2.imshow("best", warpImg)
    cv2.waitKey(0)
    # img3 = cv2.cvtColor(direct, cv2.COLOR_BGR2RGB)
    # plt.imshow(img3, ), plt.show()
    # img4 = cv2.cvtColor(warpImg, cv2.COLOR_BGR2RGB)
    # plt.imshow(img4, ), plt.show()
    print("simple stich cost %f" % (simple - starttime))
    print("\ntotal cost %f" % (final - starttime))
    # cv2.imwrite("simplepanorma.png", direct)
    # cv2.imwrite("bestpanorma.png", warpImg)

else:
    print("not enough matches!")