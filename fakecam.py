# not my code
# see here: https://elder.dev/posts/open-source-virtual-background/

import os
import cv2
import numpy as np
import requests
import pyfakewebcam


def get_mask(frame, bodypix_url="http://localhost:9000"):
    _, data = cv2.imencode(".jpg", frame)
    r = requests.post(
        url=bodypix_url,
        data=data.tobytes(),
        headers={"Content-Type": "application/octet-stream"},
    )
    mask = np.frombuffer(r.content, dtype=np.uint8)
    mask = mask.reshape((frame.shape[0], frame.shape[1]))
    return mask


def post_process_mask(mask):
    mask = cv2.dilate(mask, np.ones((10, 10), np.uint8), iterations=1)
    mask = cv2.blur(mask.astype(float), (30, 30))
    return mask



def get_frame(cap, background_scaled):
    _, frame = cap.read()
    # fetch the mask with retries (the app needs to warmup and we're lazy)
    # e v e n t u a l l y c o n s i s t e n t
    mask = None
    while mask is None:
        try:
            mask = get_mask(frame)
        except Exception as e:
            print("mask request failed, retrying")
            print(e)
    # post-process mask and frame
    mask = post_process_mask(mask)
    # frame = hologram_effect(frame)
    # composite the foreground and background
    inv_mask = 1 - mask
    for c in range(frame.shape[2]):
        frame[:, :, c] = frame[:, :, c] * mask + background_scaled[:, :, c] * inv_mask
    return frame


# setup access to the *real* webcam
cap = cv2.VideoCapture("/dev/video0")
height, width = 360, 640
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, 60)

# setup the fake camera
fake = pyfakewebcam.FakeWebcam("/dev/video2", width, height)

# load the virtual background
background = cv2.imread("data/background.jpg")
background_scaled = cv2.resize(background, (width, height))

# frames forever
while True:
    frame = get_frame(cap, background_scaled)
    # fake webcam expects RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fake.schedule_frame(frame)
