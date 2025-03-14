import cv2
import glob
import numpy as np
import os
import pickle
from ximea import xiapi


# Definujte priečinok, kde sú uložené obrázky
save_directory = "chess_images"

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Príprava objektových bodov (3D body v reálnom svete)
objp = np.zeros((5 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:5].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# Získanie zoznamu obrázkov v priečinku
image_paths = glob.glob(os.path.join(save_directory, '*.png'))

if not image_paths:
    print("Žiadne obrázky na kalibráciu neboli nájdené v priečinku", save_directory)
else:
    print(f"Nájdených {len(image_paths)} obrázkov na kalibráciu.")

for fname in image_paths:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Nájdite rohy šachovnice
    ret, corners = cv2.findChessboardCorners(gray, (7, 5), None)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Zobrazí rohy na obrázku
        cv2.drawChessboardCorners(img, (7, 5), corners2, ret)
        cv2.imshow('Chessboard', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

if objpoints and imgpoints:
    # Spustenie kalibrácie kamery
    print("Spúšťanie kalibrácie kamery...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Výpis výsledkov kalibrácie
    print("Kalibrácia bola úspešná.")
    print("Matica kamery:", mtx)
    print("Distortion koeficienty:", dist)

    # Uloženie matíc do súboru pomocou pickle
    calibration_data = {
        "camera_matrix": mtx,
        "distortion_coefficients": dist,
    }
    with open("camera_calibration.pkl", "wb") as f:
        pickle.dump(calibration_data, f)
    print("Kalibračné dáta boli uložené do camera_calibration.pkl")

    img = cv2.imread('chess_images/image_1.png')
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv2.imwrite('calibresult.png', dst)
    print("Opravený obrázok uložený ako calibresult.png")
else:
    print("Nepodarilo sa nájsť dostatok bodov pre kalibráciu.")

cv2.destroyAllWindows()

