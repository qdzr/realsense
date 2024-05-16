import cv2
import os
from realsense_save_rgbd import depth2RGB, inpaint

path = './out/2021_12_24_14_59_49/0000d.tiff'
if not os.path.exists(path):
    print(path + ' not exists')
    raise os.error

im_depth = cv2.imread(path, -1)
cv2.imshow('depth', depth2RGB(im_depth))
cv2.waitKey()