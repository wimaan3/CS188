import cv2
import numpy as np
import matplotlib.pyplot as plt


class VisualSLAM:
    def __init__(self, fx, fy, cx, cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.K = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1]])

        self.orb = cv2.ORB_create(2000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.prev_gray = None
        self.prev_kp = None
        self.prev_des = None
        self.prev_pose = np.eye(4)

        self.pose = np.eye(4)
        self.trajectory = [self.pose[:3, 3]]
        self.landmarks = []

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

        # Estimate essential matrix and pose
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            return None

        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.K)

        # Update pose
        Rt = np.eye(4)
        Rt[:3, :3] = R
        Rt[:3, 3] = t.squeeze()
        self.pose = self.pose @ np.linalg.inv(Rt)

        self.trajectory.append(self.pose[:3, 3].copy())
        # Triangulate points (for landmarks)
        proj1 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))  # Prev cam
        proj2 = self.K @ np.hstack((R, t))                         # Curr cam

        pts4d = cv2.triangulatePoints(proj1, proj2, pts1.T, pts2.T)
        pts3d = pts4d[:3] / pts4d[3]  # Normalize homogeneous coordinates

        # Only keep points with positive depth and reasonable range
        for pt in pts3d.T:
            if 0 < pt[2] < 50:
                # Transform to world coordinates
                pt_world = (self.prev_pose @ np.append(pt, 1).reshape(4, 1))[:3].flatten()
                self.landmarks.append(pt_world)

        # Save current as previous
        self.prev_gray = gray
        self.prev_kp = kp
        self.prev_des = des
        self.prev_pose = self.pose.copy()

        return {
            "trajectory": np.array(self.trajectory),
            "landmarks": np.array(self.landmarks)
        }


def run_slam(video_source):
    fx, fy = 650, 650
    cx, cy = 320, 240

    slam = VisualSLAM(fx, fy, cx, cy)
    cap = cv2.VideoCapture(video_source)

    plt.ion()
    fig, ax = plt.subplots()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = slam.process_frame(frame)
        if result:
            traj = result["trajectory"]
            landmarks = result["landmarks"]

            ax.clear()
            ax.plot(traj[:, 0], traj[:, 2], 'b-', label='Camera Trajectory')
            if len(landmarks) > 0:
                ax.scatter(landmarks[:, 0], landmarks[:, 2], s=1, c='gray', label='Landmarks')
            ax.set_title("SLAM: Top-Down View")
            ax.set_xlabel("X")
            ax.set_ylabel("Z")
            ax.set_aspect('equal')
            ax.legend()
            plt.pause(0.01)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import time

    print("Start")
    time.sleep(10)  # Wait for 10 seconds
    print("End")
    run_slam(0)  # or 0 for webca


    # ran this on laptop recording 
    # but run on real time laptop spin circle 
    # gets a new png
    # make pgm with that 
    # run pgm on rrt.pt