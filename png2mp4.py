#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 15:17:37 2023

@author: satoshihoshika
"""

############################################################################
import sys
import cv2

# encoder(for mp4)
# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')    
# output file name, encoder, fps, size(fit to image size)
# video = cv2.VideoWriter('./fig/video.mp4',fourcc, 20.0, (1240, 1360))
tmp = cv2.imread('./fig/figure010.png').shape

video = cv2.VideoWriter('./fig/video.mov',fourcc,20, (tmp[1],tmp[0]))


if not video.isOpened():
    print("can't be opened")
    sys.exit()

for i in range(10, 790, 10):
    print(i)
    print('./fig/figure%03d.png'% i)
    # hoge0000.png, hoge0001.png,..., hoge0090.png
    img = cv2.imread('./fig/figure%03d.png' % i)

    # can't read image, escape 
    if img is None:
        print("can't read")
        break

    # add
    video.write(img)
    print(i)

video.release()
print('written')