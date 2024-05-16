'''
Description: 
Author: wangdx
Date: 2021-11-24 21:38:37
LastEditTime: 2021-11-24 21:39:03
'''
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
from imageio import imsave


def depth2Gray(im_depth):
    """
    将深度图转至三通道8位灰度图
    (h, w, 3)
    """
    # 16位转8位
    x_max = np.max(im_depth)
    x_min = np.min(im_depth)
    if x_max == x_min:
        print('图像渲染出错 ...')
        raise EOFError

    k = 255 / (x_max - x_min)
    b = 255 - k * x_max

    ret = (im_depth * k + b).astype(np.uint8)
    return ret


def depth2RGB(im_depth):
    """
    将深度图转至三通道8位彩色图
    先将值为0的点去除，然后转换为彩图，然后将值为0的点设为红色
    (h, w, 3)
    im_depth: 单位 mm或m
    """
    im_depth = depth2Gray(im_depth)
    im_color = cv2.applyColorMap(im_depth, cv2.COLORMAP_JET)
    return im_color


def inpaint(img, missing_value=0):
    """
    Inpaint missing values in depth image.
    :param missing_value: Value to fill in teh depth image.
    """
    img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    mask = (img == missing_value).astype(np.uint8)

    # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
    scale = np.abs(img).max()
    img = img.astype(np.float32) / scale  # Has to be float32, 64 not supported.
    img = cv2.inpaint(img, mask, 1, cv2.INPAINT_NS)

    # Back to original size and value range.
    img = img[1:-1, 1:-1]
    img = img * scale

    return img


def run():
    pipeline = rs.pipeline()

    # Create a config并配置要流​​式传输的管道
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    align_to = rs.stream.color
    align = rs.align(align_to)

    # 按照日期创建文件夹
    save_path = os.path.join(os.getcwd(), "out", time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    os.mkdir(save_path)

    # 保存的图片和实时的图片界面
    cv2.namedWindow("live", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("save", cv2.WINDOW_AUTOSIZE)
    saved_count = 0

    pipeline.start(config)
    # 主循环
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            # 获取RGB图像
            color_frame = aligned_frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            # 获取深度图
            aligned_depth_frame = aligned_frames.get_depth_frame()
            depth_image = np.asanyarray(aligned_depth_frame.get_data()).astype(np.float32) / 1000.  # 单位为m
            # 可视化图像
            depth_image = inpaint(depth_image)  # 补全缺失值
            depth_image_color = depth2RGB(depth_image)
            cv2.imshow("live", np.hstack((color_image, depth_image_color)))
            key = cv2.waitKey(30)

            # s 保存图片
            if key & 0xFF == ord('s'):
                cv2.imwrite(os.path.join(save_path, "{:04d}r.png".format(saved_count)), color_image)  # 保存RGB为png文件
                imsave(os.path.join(save_path, "{:04d}d.tiff".format(saved_count)), depth_image)  # 保存深度图为tiff文件
                saved_count += 1
                cv2.imshow("save", np.hstack((color_image, depth_image_color)))

            # q 退出
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()


if __name__ == '__main__':
    run()
