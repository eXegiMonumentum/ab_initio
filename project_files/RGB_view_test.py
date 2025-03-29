import depthai as dai
import cv2

# Tworzenie pipeline
pipeline = dai.Pipeline()

# Dodajemy kamerę kolorową
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setFps(30)

# Output
xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("video")
cam_rgb.video.link(xout.input)

# Uruchomienie urządzenia
with dai.Device(pipeline) as device:
    q = device.getOutputQueue(name="video", maxSize=4, blocking=False)
    print("✅ Kamera działa — naciśnij 'q', aby wyjść.")

    while True:
        in_frame = q.get()
        frame = in_frame.getCvFrame()
        cv2.imshow("OAK-D RGB", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("👋 Zakończono podgląd.")
            break

cv2.destroyAllWindows()
