import os
import cv2

img_dir = r'D:\yu_project\match\stitch'
names = os.listdir(img_dir)

images = []
for name in names:
    img_path = os.path.join(img_dir, name)
    image = cv2.imread(img_path)
    images.append(image)
stitcher = cv2.createStitcher()
status, stitched = stitcher.stitch(images)
print(status)
cv2.imshow("", stitched)
cv2.waitKey(0)
if status == 0:
    cv2.imwrite('D:/yu_project/match/res/stitch.jpg', stitched)