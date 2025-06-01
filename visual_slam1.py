import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class VisualSLAM:
    def __init__(self, fx, fy, cx, cy):
        self.K = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0,  0,  1]])
        self.orb = cv2.ORB_create(2000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_gray = None
        self.prev_kp = None
        self.prev_des = None
        self.prev_pose = np.eye(4)
        self.pose = np.eye(4)
        self.trajectory = []
        self.landmarks = []

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_kp = kp
            self.prev_des = des
            self.trajectory.append(self.pose[:3, 3].copy())
            return
        matches = sorted(self.bf.match(self.prev_des, des), key=lambda x: x.distance)
        if len(matches) < 8:
            return
        pts1 = np.float32([self.prev_kp[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp[m.trainIdx].pt for m in matches])
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, cv2.RANSAC, 0.999, 1.0)
        if E is None or mask is None:
            return
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.K)
        Rt = np.eye(4)
        Rt[:3, :3] = R
        Rt[:3, 3] = t.squeeze()
        self.pose = self.pose @ np.linalg.inv(Rt)
        self.trajectory.append(self.pose[:3, 3].copy())
        proj1 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        proj2 = self.K @ np.hstack((R, t))
        pts4d = cv2.triangulatePoints(proj1, proj2, pts1.T, pts2.T)
        pts3d = pts4d[:3] / pts4d[3]
        for pt in pts3d.T:
            if 0 < pt[2] < 50:
                pt_hom = np.append(pt, 1)
                world_point = (self.prev_pose @ pt_hom.reshape(4, 1))[:3].flatten()
                self.landmarks.append(world_point)
        self.prev_gray = gray
        self.prev_kp = kp
        self.prev_des = des
        self.prev_pose = self.pose.copy()

    def run(self, video_source):
        cap = cv2.VideoCapture(video_source)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        traj_line, = ax.plot([], [], [], 'b-')
        land_scatter = ax.scatter([], [], [], s=1, c='gray')
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.process_frame(frame)
            frame_idx += 1
            if frame_idx % 10 == 0:
                traj = np.array(self.trajectory)
                if traj.size:
                    traj_line.set_data(traj[:, 0], traj[:, 1])
                    traj_line.set_3d_properties(traj[:, 2])
                if self.landmarks:
                    lm = np.array(self.landmarks)
                    land_scatter._offsets3d = (lm[:, 0], lm[:, 1], lm[:, 2])
                ax.set_xlim(traj[:,0].min(), traj[:,0].max())
                ax.set_ylim(traj[:,1].min(), traj[:,1].max())
                ax.set_zlim(traj[:,2].min(), traj[:,2].max())
                ax.set_title(f'Frame {frame_idx}/{total}')
                plt.draw()
                plt.pause(0.001)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        np.save('landmarks.npy', np.array(self.landmarks))
        np.save('trajectory.npy', np.array(self.trajectory))
        return np.array(self.landmarks), np.array(self.trajectory)

if __name__ == "__main__":
    fx, fy, cx, cy = 1425, 1425, 956, 538
    slam = VisualSLAM(fx, fy, cx, cy)
    landmarks, trajectory = slam.run("videos/slamvid.mp4") 
    print("Saved landmarks.npy and trajectory.npy")