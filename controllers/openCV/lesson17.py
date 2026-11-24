# FILE: lesson17.py
# LOCATION: python-CV/controllers/openCV/lesson17.py
# WHY: Learn to detect movement using live camera frames.
# WHAT: Detects moving regions and draws bounding boxes.
# HOW: Uses frame differencing, MOG2, masks, and contours.
# WHEN: Runs when the Lesson 17 video endpoint is opened.
# WHERE: Uses Webcam 0 and streams results through FastAPI.

from threading import Lock  # one key for the webcam, so only one stream can hold it
from time import perf_counter  # a precise clock used to count frames per second
from typing import Literal  # a list of allowed names, here "difference" or "mog2"

import numpy as np  # number arrays; an image is an array of pixel numbers
from fastapi import HTTPException  # a clear error the browser can read, such as camera unavailable
from fastapi.responses import StreamingResponse  # sends JPEG frames one after another without ending the page
from config.open_cv.configureCV import configureCV  # our setup function that returns the installed OpenCV module

# Load the OpenCV build selected by our project config.
openCV = configureCV()

# Webcam 0 is the default camera on this machine.
cameraIndex = 0

# Neon green boxes stay easy to see on the live picture.
# BGR order: blue 20, green 255, red 57.
neonGreen = (20, 255, 57)

# Ignore motion regions smaller than 900 square pixels.
minimumArea = 900

# One viewer at a time, so two requests cannot open the same webcam.
cameraLock = Lock()
activeCamera = None

# A newer request increments this number and the old stream stops.
streamVersion = 0

# difference compares one frame with the last. mog2 is the older name for background.
MotionMode = Literal["difference", "mog2"]

# background is MOG2. difference compares this frame with the previous frame.
MotionMethod = Literal["difference", "background"]
currentMethod = "difference"

# boxes draws rectangles. mask shows the black-and-white motion image.
currentView = "boxes"

# boxes draws rectangles on the camera picture. mask shows the black-and-white motion image.
ViewMode = Literal["boxes", "mask"]


# Open Webcam 0 and ask it for a smaller picture so each frame stays fast.
def openCamera():
    camera = openCV.VideoCapture(cameraIndex)

    if not camera.isOpened():
        camera.release()
        raise HTTPException(status_code=503, detail="Webcam 0 is unavailable")

    camera.set(openCV.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(openCV.CAP_PROP_FRAME_HEIGHT, 480)
    return camera


# Turn the colour frame into a blurred gray image so tiny brightness changes are ignored.
def prepareGrayFrame(frame):
    grayImage = openCV.cvtColor(frame, openCV.COLOR_BGR2GRAY)
    return openCV.GaussianBlur(grayImage, (5, 5), 0)


# Frame differencing: white pixels are places that changed since the previous frame.
def differenceMask(grayFrame, previousFrame):
    if previousFrame is None:
        return np.zeros(grayFrame.shape, dtype=np.uint8)

    changedPixels = openCV.absdiff(previousFrame, grayFrame)
    _, mask = openCV.threshold(changedPixels, 25, 255, openCV.THRESH_BINARY)
    return mask


# MOG2 learns which pixels stay still, then marks the pixels that do not fit that background.
def mog2Mask(frame, subtractor):
    mask = subtractor.apply(frame)
    _, mask = openCV.threshold(mask, 200, 255, openCV.THRESH_BINARY)
    return mask


# Remove specks and join nearby moving pixels so one object becomes one white blob.
def cleanMask(mask):
    kernel = openCV.getStructuringElement(openCV.MORPH_RECT, (3, 3))
    openedMask = openCV.morphologyEx(mask, openCV.MORPH_OPEN, kernel)
    return openCV.dilate(openedMask, kernel, iterations=2)


# Find the outline of each white blob and draw a box around the ones large enough to matter.
def drawMotionBoxes(frame, mask):
    resultImage = frame.copy()
    contours, hierarchy = openCV.findContours(
        mask, openCV.RETR_EXTERNAL, openCV.CHAIN_APPROX_SIMPLE,
    )
    movingRegions = 0

    for contour in contours:
        # Area is the number of pixels inside the contour.
        area = openCV.contourArea(contour)
        if area < minimumArea:
            continue

        movingRegions += 1
        leftX, topY, width, height = openCV.boundingRect(contour)
        openCV.rectangle(
            resultImage,
            (leftX, topY),
            (leftX + width, topY + height),
            neonGreen,
            2,
        )
        openCV.putText(
            resultImage,
            f"Motion {movingRegions}",
            (leftX, max(75, topY - 10)),
            openCV.FONT_HERSHEY_SIMPLEX,
            0.6,
            neonGreen,
            2,
        )

    return resultImage, movingRegions


# Choose the motion mask, then draw boxes and the frames-per-second number on the picture.
def markMotion(frame, mode, previousFrame, subtractor):
    grayFrame = prepareGrayFrame(frame)
    mask = differenceMask(grayFrame, previousFrame) if mode == "difference" else mog2Mask(frame, subtractor)
    resultImage, movingRegions = drawMotionBoxes(frame, cleanMask(mask))
    return resultImage, grayFrame


# Write the mode name and the current speed in the corner of the frame.
def drawFrameLabel(frame, mode, fps):
    labeledFrame = frame.copy()
    label = f"{mode}  FPS: {fps:.1f}"
    openCV.putText(labeledFrame, label, (15, 30), openCV.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    openCV.putText(labeledFrame, label, (15, 30), openCV.FONT_HERSHEY_SIMPLEX, 0.7, neonGreen, 2)
    return labeledFrame


# Detect pixels that changed, using either the previous frame or the learned background.
def createMotionMask(frame, previousGray, backgroundModel, method):
    # METHOD 1: compare this frame with the frame before it.
    if method == "difference":
        grayImage = openCV.cvtColor(frame, openCV.COLOR_BGR2GRAY)
        # Blur hides tiny camera noise that is not real movement.
        grayImage = openCV.GaussianBlur(grayImage, (7, 7), 0)

        if previousGray is None:
            # The first frame has nothing to compare, so the mask stays black.
            return np.zeros_like(grayImage), grayImage

        difference = openCV.absdiff(previousGray, grayImage)
        # Differences stronger than 25 become white. The rest stay black.
        _, motionMask = openCV.threshold(difference, 25, 255, openCV.THRESH_BINARY)
        return motionMask, grayImage

    # METHOD 2: MOG2 learns the still background and marks pixels that do not fit it.
    foregroundMask = backgroundModel.apply(frame)
    # Shadow pixels are gray, about 127. Keep only the bright foreground.
    _, motionMask = openCV.threshold(foregroundMask, 200, 255, openCV.THRESH_BINARY)
    return motionMask, previousGray


# Remove specks, then expand the remaining white regions so a moving object stays connected.
def cleanMotionMask(motionMask):
    kernel = openCV.getStructuringElement(openCV.MORPH_RECT, (5, 5))
    cleanedMask = openCV.morphologyEx(motionMask, openCV.MORPH_OPEN, kernel)
    return openCV.dilate(cleanedMask, kernel, iterations=2)


# Keep reading Webcam 0, mark movement, and hand each JPEG to the browser.
def generateVideoFrames(camera, mode, streamId, view="boxes"):
    global activeCamera
    previousTime = perf_counter()
    previousGray = None
    activeCamera = camera
    # history is how many recent frames MOG2 remembers.
    # varThreshold is how different a pixel must be before it counts as motion.
    backgroundModel = openCV.createBackgroundSubtractorMOG2(
        history=200,
        varThreshold=25,
        detectShadows=True,
    )
    lastMethod = None

    try:
        while streamId == streamVersion:
            success, frame = camera.read()
            if not success:
                break

            # Read the page's current choice on every frame, so the buttons do not open a second camera.
            method = "background" if currentMethod == "background" else "difference"
            view = currentView
            if method != lastMethod:
                previousGray = None
                lastMethod = method

            motionMask, previousGray = createMotionMask(
                frame, previousGray, backgroundModel, method,
            )
            motionMask = cleanMotionMask(motionMask)
            resultImage, movingRegions = drawMotionBoxes(frame, motionMask)

            # The mask view shows the black-and-white motion image instead of the boxes.
            if view == "mask":
                resultImage = openCV.cvtColor(motionMask, openCV.COLOR_GRAY2BGR)

            currentTime = perf_counter()
            elapsedTime = currentTime - previousTime
            previousTime = currentTime
            fps = 1 / elapsedTime if elapsedTime > 0 else 0
            labels = [
                f"Method: {method}",
                f"Motion regions: {movingRegions}",
                f"FPS: {fps:.1f}",
            ]

            for index, label in enumerate(labels):
                textPosition = (15, 25 + index * 25)
                openCV.putText(resultImage, label, textPosition, openCV.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
                openCV.putText(resultImage, label, textPosition, openCV.FONT_HERSHEY_SIMPLEX, 0.6, neonGreen, 2)

            encoded, imageBytes = openCV.imencode(
                ".jpg", resultImage, [openCV.IMWRITE_JPEG_QUALITY, 80],
            )
            if not encoded:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"\r\n" + imageBytes.tobytes() + b"\r\n"
            )
    finally:
        activeCamera = None
        camera.release()
        cameraLock.release()


# Open the camera and stream the chosen motion view until the browser closes it.
def streamMotion(mode: MotionMode = "difference", view: ViewMode = "boxes"):
    global streamVersion, currentMethod, currentView
    streamVersion += 1
    streamId = streamVersion

    if not cameraLock.acquire(timeout=2):
        raise HTTPException(status_code=409, detail="Webcam 0 is already in use")

    if streamId != streamVersion:
        cameraLock.release()
        raise HTTPException(status_code=409, detail="A newer camera stream replaced this one")

    camera = None

    try:
        # The first frame uses the choices from this request. Later button clicks update the same stream.
        currentMethod = "background" if mode == "mog2" else "difference"
        currentView = view
        camera = openCamera()
        return StreamingResponse(
            generateVideoFrames(camera, mode, streamId, view),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store"},
        )
    except Exception:
        if camera is not None:
            camera.release()
        cameraLock.release()
        raise


# Same stream. method=background is MOG2. mode=mog2 is the older name for that same choice.
def streamMotionVideo(
    method: MotionMethod = "background",
    view: ViewMode = "boxes",
    mode: MotionMode | None = None,
):
    if mode == "mog2":
        method = "background"
    elif mode == "difference":
        method = "difference"

    chosenMode = "mog2" if method == "background" else "difference"
    return streamMotion(chosenMode, view)


# Stop the current stream and wait until Webcam 0 is released.
def stopMotion():
    global streamVersion, activeCamera
    streamVersion += 1

    # Releasing the open camera unblocks a read that is still waiting for a frame.
    camera = activeCamera
    if camera is not None:
        camera.release()

    if cameraLock.acquire(timeout=3):
        cameraLock.release()

    return {"camera": "released"}


# Change the live picture without opening Webcam 0 a second time.
def updateMotionSettings(method: MotionMethod = "difference", view: ViewMode = "boxes"):
    global currentMethod, currentView
    currentMethod = method
    currentView = view
    return {"method": currentMethod, "view": currentView}
