import cv2
import os

from PyQt5.QtGui import QImage, QPixmap

def read_video():
    # Need to download video from website google drive link and replace name with path here
    cap = cv2.VideoCapture('videos/video.mp4')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        process_frame(frame) # slam logic

def read_imgs():
    image_folder = 'directory/'
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg'))])

    for file in image_files:
        frame = cv2.imread(os.path.join(image_folder, file))
        process_frame(frame) # slam logic

def display_frame_in_label(frame_bgr):
    h, w, ch = frame_bgr.shape
    bytes_per_line = ch * w
    qimg = QImage(frame_bgr.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
    your_label.setPixmap(QPixmap.fromImage(qimg))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread
from time import sleep


class VisualSLAM:
    def __init__(self, fx, fy, cx, cy):
        self.prev_gray = None
        self.prev_kp = None
        self.prev_des = None
        self.orb = cv2.ORB_create(2000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.pose = np.eye(4)
        self.trajectory = [self.pose[:3, 3]]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_kp = kp
            self.prev_des = des
            return None

        matches = self.bf.match(self.prev_des, des)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) < 8:
            return None

        pts1 = np.float32([self.prev_kp[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp[m.trainIdx].pt for m in matches])

        E, mask = cv2.findEssentialMat(pts1, pts2, focal=self.fx, pp=(self.cx, self.cy), method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            return None

        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, focal=self.fx, pp=(self.cx, self.cy))

        Rt = np.eye(4)
        Rt[:3, :3] = R
        Rt[:3, 3] = t.squeeze()
        self.pose = self.pose @ np.linalg.inv(Rt)

        self.trajectory.append(self.pose[:3, 3].copy())

        self.prev_gray = gray
        self.prev_kp = kp
        self.prev_des = des

        return {
            "pose": self.pose.copy(),
            "trajectory": np.array(self.trajectory)
        }
    
def live_plot_thread(traj_buffer):
    plt.ion()
    fig, ax = plt.subplots()
    while True:
        if len(traj_buffer) > 1:
            traj = np.array(traj_buffer)
            ax.clear()
            ax.plot(traj[:, 0], traj[:, 2], color='blue')  # X-Z top-down view
            ax.set_title("Camera Trajectory (Top-Down)")
            ax.set_xlabel("X")
            ax.set_ylabel("Z")
            ax.set_aspect('equal')
            plt.pause(0.01)
        sleep(0.05)


# --- Main SLAM loop ---
def run_slam(video_source):
    fx, fy = 718.856, 718.856
    cx, cy = 607.1928, 185.2157

    slam = VisualSLAM(fx, fy, cx, cy)
    cap = cv2.VideoCapture(video_source)

    traj_buffer = []

    # Start live plot
    plot_thread = Thread(target=live_plot_thread, args=(traj_buffer,), daemon=True)
    plot_thread.start()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = slam.process_frame(frame)
        if result:
            traj_buffer.clear()
            traj_buffer.extend(result["trajectory"])

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_slam("videos/map_test.mp4")  # or use 0 for
    # download video from website here from google drive link and rename this path: 