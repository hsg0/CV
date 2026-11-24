import os

from fastapi import HTTPException, Response
from config.open_cv.configureCV import assetsDirectory, cameraIndex, configureCV

openCV = configureCV()


def nextWebcamPath():
    imageNumber = 1
    filePath = os.path.join(assetsDirectory, f"webcam{imageNumber}.jpg")

    while os.path.exists(filePath):
        imageNumber += 1
        filePath = os.path.join(assetsDirectory, f"webcam{imageNumber}.jpg")

    return filePath


def getCameraFrame():
    camera = openCV.VideoCapture(cameraIndex)

    try:
        if not camera.isOpened():
            raise HTTPException(
                status_code=503,
                detail="Unable to find and open camera"
            )

        success, frame = camera.read()

        if not success or frame is None:
            raise HTTPException(
                status_code=503,
                detail="Unable to read frame from camera"
            )

        encoded, image = openCV.imencode(".jpg", frame)

        if not encoded:
            raise HTTPException(
                status_code=500,
                detail="Unable to encode camera frame"
            )

        imageBytes = image.tobytes()
        filePath = nextWebcamPath()

        with open(filePath, "wb") as imageFile:
            imageFile.write(imageBytes)

        return Response(
            content=imageBytes,
            media_type="image/jpeg"
        )

    finally:
        camera.release()
