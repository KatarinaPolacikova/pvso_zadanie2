from ximea import xiapi
import cv2
import glob
import os
import numpy as np

# ROZMERY ŠACHOVNICE 5x8


# Definujte priečinok, kam sa majú obrázky ukladať
save_directory = "chess_images"
os.makedirs(save_directory, exist_ok=True)

# Vytvorenie inštancie pre prvú pripojenú kameru
cam = xiapi.Camera()

# Spustenie komunikácie s kamerou
print('Opening first camera...')
cam.open_device()

# Nastavenie parametrov kamery
cam.set_exposure(100000)
cam.set_param('imgdataformat', 'XI_RGB32')
cam.set_param('auto_wb', 1)
print('Exposure was set to %i us' % cam.get_exposure())

# Vytvorenie inštancie pre obrázok
img = xiapi.Image()

# Spustenie získavania dát
print('Starting data acquisition...')
cam.start_acquisition()

captured_images = 0
images = []

while True:
    cam.get_image(img)
    image = img.get_image_data_numpy()
    image = cv2.resize(image, (480, 480))
    cv2.imshow("Live Feed", image)

    key = cv2.waitKey(1)

    if key == ord(' '):  # Stlačenie medzerníka zachytí snímku
        image_path = os.path.join(save_directory, f"image_{captured_images + 1}.png")
        cv2.imwrite(image_path, image)
        print(f"Image {captured_images + 1} saved to {image_path}")
        images.append(image)
        captured_images += 1

        if captured_images >= 20:
            break  # Po zachytení 4 snímok sa program ukončí

    elif key == ord('q'):  # Stlačením 'q' sa program ukončí
        break

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob('*.png')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()


# Zastavenie získavania dát
print('Stopping acquisition...')
cam.stop_acquisition()

# Ukončenie komunikácie s kamerou
cam.close_device()
cv2.destroyAllWindows()
print('Done.')