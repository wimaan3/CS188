import cv2
import numpy as np

# Set the size of the checkerboard (internal corners, not squares!)
CHECKERBOARD = (7, 6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare 3D object points
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Arrays to store object and image points
objpoints = []
imgpoints = []

# Open webcam
cap = cv2.VideoCapture(0)  # Change to 1 if Hiwonder is not on index 0

if not cap.isOpened():
    print("Camera not detected!")
    exit()

print("Press SPACE to capture frame for calibration.")
print("Press 'c' to compute calibration once you've captured enough.")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if found:
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        preview = cv2.drawChessboardCorners(frame.copy(), CHECKERBOARD, corners2, found)
    else:
        preview = frame

    cv2.putText(preview, f"Captured frames: {len(objpoints)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Calibration Preview", preview)
    key = cv2.waitKey(1)

    if key == ord(' '):  # Capture frame
        if found:
            print(f"Captured frame #{len(objpoints)+1}")
            objpoints.append(objp)
            imgpoints.append(corners2)
        else:
            print("Checkerboard not found in this frame. Try a better angle or focus.")

    elif key == ord('c'):  # Calibrate
        if len(objpoints) < 5:
            print("Not enough captures. Try at least 10 varied views.")
            continue

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )

        print("\n=== Calibration Complete ===")
        print("Camera matrix:\n", mtx)
        print("Distortion coefficients:\n", dist)
        break

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()