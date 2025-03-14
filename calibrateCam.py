import cv2
import glob
import numpy as np
import os
import pickle
from ximea import xiapi

# veľkosť obrazu
size = 480

calib_file = "camera_calibration.pkl"
if os.path.exists(calib_file):
    # Načítanie
    with open("camera_calibration.pkl", "rb") as f:
        calibration_data = pickle.load(f)
else:
    raise FileNotFoundError(f"Subor {calib_file} nenajdenz!")

# Prístup k premenným
mtx = calibration_data["camera_matrix"]
dist = calibration_data["distortion_coefficients"]

# Výpis premennzch
print("Camera Matrix:\n", mtx)
print("Distortion Coefficients:\n", dist)


# Vytvorenie inštancie pre pripojenú kameru
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

firstTime = True
while True:
    cam.get_image(img)  # Získa obraz z kamery
    frame = img.get_image_data_numpy()  # Konvertuje obraz na numpy array

    frame = cv2.resize(frame, (size, size))

    # mozno bezpecnejšie to robit po kazdom opakovani, nie len prvz krat
    if firstTime:
        # Získanie veľkosti obrázka - moznost zrychlenia pri fixne zadanzch hodnotach
        h, w = frame.shape[:2]
        # Vytvorenie optimalizovanej kamery
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix( mtx, dist, (w, h), 1, (w, h))
        x, y, w_roi, h_roi = roi
        firstTime = False

    # Korekcia skreslenia
    undistorted_frame = cv2.undistort(frame, mtx, dist, None, new_camera_matrix)
    undistorted_frame = undistorted_frame[y:y + h_roi, x:x + w_roi]
    cv2.imshow('Undistorted Frame', undistorted_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Ukončí slučku, ale kamera sa stále drží otvorená


# Uvoľnenie kamery a zatvorenie okien

# Zastavenie získavania dát
print('Stopping acquisition...')
cam.stop_acquisition()

# Ukončenie komunikácie s kamerou
cam.close_device()
cv2.destroyAllWindows()
print('Done.')# Vytvorenie inštancie pre prvú pripojenú kameru
