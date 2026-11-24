# FILE: lesson16.py
# LOCATION: python-CV/controllers/openCV/lesson16.py
# WHY: Combine our earlier computer vision lessons using live video.
# WHAT: Processes Webcam 0 with filters, masks, contours, and ORB.
# HOW: Reads frames, applies a selected mode, and streams JPEGs.
# WHEN: Runs when Lesson 16 opens a camera stream or selects an ROI.
# WHERE: Used by routes/opencv/lesson16.py and FastAPI main.py.

from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Literal

import numpy as np
from fastapi import HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from config.open_cv.configureCV import assetsDirectory, configureCV

openCV = configureCV()
assetsPath = Path(assetsDirectory)
cameraIndex = 0
neonGreen = (20, 255, 57)
orange = (0, 165, 255)
cameraLock = Lock()
streamVersion = 0

ViewMode = Literal[
    "original", "gray", "blur", "brightness",
    "blue", "green", "red", "crop",
    "canny", "binary", "adaptive", "range",
    "erode", "dilate", "opening", "closing",
    "color", "contours", "features", "match",
    "flip", "rotate", "affine", "perspective",
]
ColorName = Literal["red", "green", "blue", "yellow"]


# Camera engine: Opens Webcam 0, captures frames, calculates FPS and streams JPEGs.
def openCamera():
    camera = openCV.VideoCapture(cameraIndex)

    if not camera.isOpened():
        camera.release()
        raise HTTPException(status_code=503, detail="Webcam 0 is unavailable")

    camera.set(openCV.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(openCV.CAP_PROP_FRAME_HEIGHT, 480)
    return camera


def saveImage(fileName, image):
    savePath = assetsPath / fileName

    if not openCV.imwrite(str(savePath), image):
        raise RuntimeError(f"Cannot save image: {fileName}")

    return fileName


# Image-processing functions: Reuse grayscale, blur, brightness, thresholding, HSV and morphology.
def convertToGrayscale(frame):
    return openCV.cvtColor(frame, openCV.COLOR_BGR2GRAY)


def blurImage(frame):
    return openCV.GaussianBlur(frame, (7, 7), 0)


def createBinaryMask(frame, mode="otsu"):
    grayImage = convertToGrayscale(frame)
    blurredImage = blurImage(grayImage)

    if mode == "binary":
        _, mask = openCV.threshold(blurredImage, 127, 255, openCV.THRESH_BINARY)
    elif mode == "adaptive":
        mask = openCV.adaptiveThreshold(
            blurredImage, 255, openCV.ADAPTIVE_THRESH_GAUSSIAN_C,
            openCV.THRESH_BINARY, 11, 2,
        )
    elif mode == "range":
        mask = openCV.inRange(blurredImage, 80, 180)
    else:
        _, mask = openCV.threshold(
            blurredImage, 0, 255, openCV.THRESH_BINARY_INV + openCV.THRESH_OTSU,
        )

    return mask


def applyMorphology(mask, mode):
    kernel = openCV.getStructuringElement(openCV.MORPH_RECT, (3, 3))

    if mode == "erode":
        return openCV.erode(mask, kernel)
    if mode == "dilate":
        return openCV.dilate(mask, kernel)
    if mode == "opening":
        return openCV.morphologyEx(mask, openCV.MORPH_OPEN, kernel)
    if mode == "closing":
        return openCV.morphologyEx(mask, openCV.MORPH_CLOSE, kernel)

    return mask


def createColorMask(frame, colorName):
    hsvImage = openCV.cvtColor(frame, openCV.COLOR_BGR2HSV)
    colorRanges = {
        "red": [((0, 100, 100), (10, 255, 255)), ((170, 100, 100), (179, 255, 255))],
        "green": [((36, 80, 70), (85, 255, 255))],
        "blue": [((90, 80, 70), (130, 255, 255))],
        "yellow": [((20, 100, 100), (35, 255, 255))],
    }
    mask = np.zeros(hsvImage.shape[:2], dtype=np.uint8)

    for lowerRange, upperRange in colorRanges[colorName]:
        mask = openCV.bitwise_or(mask, openCV.inRange(hsvImage, lowerRange, upperRange))

    mask = applyMorphology(mask, "opening")
    mask = applyMorphology(mask, "closing")
    return mask


# Geometry and drawing: Resize, crop, rotate, flip, perspective, contours, circles and bounding boxes.
def drawContoursAndBoxes(frame, mask):
    resultImage = frame.copy()
    contours, hierarchy = openCV.findContours(
        mask.copy(), openCV.RETR_EXTERNAL, openCV.CHAIN_APPROX_SIMPLE,
    )
    detectedObjects = 0
    contours = sorted(contours, key=openCV.contourArea, reverse=True)[:30]

    for contour in contours:
        area = openCV.contourArea(contour)
        if area < 350:
            continue

        detectedObjects += 1
        leftX, topY, width, height = openCV.boundingRect(contour)
        centerX = leftX + width // 2
        centerY = topY + height // 2
        openCV.drawContours(resultImage, [contour], -1, neonGreen, 2)
        openCV.rectangle(
            resultImage, (leftX, topY), (leftX + width, topY + height), orange, 2,
        )
        openCV.circle(resultImage, (centerX, centerY), 5, neonGreen, -1)
        openCV.putText(
            resultImage,
            f"Region {detectedObjects}: {int(area)}",
            (leftX, max(25, topY - 10)),
            openCV.FONT_HERSHEY_SIMPLEX,
            0.5,
            neonGreen,
            2,
        )

    return resultImage


def transformFrame(frame, mode):
    height, width = frame.shape[:2]

    if mode == "flip":
        return openCV.flip(frame, 1)

    if mode == "crop":
        cropX = int(width * 0.15)
        cropY = int(height * 0.15)
        croppedImage = frame[cropY:height - cropY, cropX:width - cropX]
        return openCV.resize(croppedImage, (width, height))

    if mode == "rotate":
        rotationMatrix = openCV.getRotationMatrix2D((width / 2, height / 2), 25, 0.8)
        return openCV.warpAffine(frame, rotationMatrix, (width, height))

    if mode == "affine":
        sourcePoints = np.float32([[0, 0], [width - 1, 0], [0, height - 1]])
        destinationPoints = np.float32([
            [width * 0.08, height * 0.08],
            [width * 0.92, height * 0.03],
            [width * 0.03, height * 0.92],
        ])
        matrix = openCV.getAffineTransform(sourcePoints, destinationPoints)
        return openCV.warpAffine(frame, matrix, (width, height))

    if mode == "perspective":
        sourcePoints = np.float32([
            [0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1],
        ])
        destinationPoints = np.float32([
            [width * 0.13, height * 0.12],
            [width * 0.82, height * 0.04],
            [width * 0.93, height * 0.90],
            [width * 0.06, height * 0.95],
        ])
        matrix = openCV.getPerspectiveTransform(sourcePoints, destinationPoints)
        return openCV.warpPerspective(frame, matrix, (width, height))

    return frame.copy()


# ORB feature engine: Detects and compares descriptors, estimates homography and draws matching lines.
def detectFeatures(frame, orbDetector):
    return orbDetector.detectAndCompute(convertToGrayscale(frame), None)


def loadReference(orbDetector):
    referencePath = assetsPath / "lesson16-reference.png"
    if not referencePath.is_file():
        referencePath = assetsPath / "6-head-crop.png"

    referenceImage = openCV.imread(str(referencePath))
    if referenceImage is None:
        raise HTTPException(status_code=422, detail="Select a reference image before matching")

    keypoints, descriptors = detectFeatures(referenceImage, orbDetector)
    if descriptors is None or len(keypoints) < 8:
        raise HTTPException(status_code=422, detail="Reference needs more visual detail")

    return referenceImage, keypoints, descriptors


def matchReference(frame, orbDetector, referenceData):
    referenceImage, referenceKeypoints, referenceDescriptors = referenceData
    frameKeypoints, frameDescriptors = detectFeatures(frame, orbDetector)
    resultImage = frame.copy()
    goodMatches = []
    inlierCount = 0
    headLocated = False

    if frameDescriptors is not None:
        matcher = openCV.BFMatcher(openCV.NORM_HAMMING, crossCheck=False)
        candidateMatches = matcher.knnMatch(referenceDescriptors, frameDescriptors, k=2)

        for pair in candidateMatches:
            if len(pair) < 2:
                continue
            bestMatch, secondMatch = pair
            if bestMatch.distance < 0.75 * secondMatch.distance:
                goodMatches.append(bestMatch)

        if len(goodMatches) >= 8:
            referencePoints = np.float32([
                referenceKeypoints[match.queryIdx].pt for match in goodMatches
            ]).reshape(-1, 1, 2)
            framePoints = np.float32([
                frameKeypoints[match.trainIdx].pt for match in goodMatches
            ]).reshape(-1, 1, 2)
            matrix, inlierMask = openCV.findHomography(
                referencePoints, framePoints, openCV.RANSAC, 5.0,
            )

            if matrix is not None and inlierMask is not None:
                inlierCount = int(inlierMask.sum())
                if inlierCount >= 8:
                    height, width = referenceImage.shape[:2]
                    corners = np.float32([
                        [0, 0], [width - 1, 0],
                        [width - 1, height - 1], [0, height - 1],
                    ]).reshape(-1, 1, 2)
                    projectedCorners = openCV.perspectiveTransform(corners, matrix)
                    frameHeight, frameWidth = frame.shape[:2]
                    validCorners = (
                        np.isfinite(projectedCorners).all()
                        and np.max(np.abs(projectedCorners)) < 3 * max(frameWidth, frameHeight)
                    )
                    if validCorners and abs(openCV.contourArea(projectedCorners)) >= 100:
                        points = np.int32(projectedCorners)
                        openCV.polylines(resultImage, [points], True, neonGreen, 3)
                        boxX, boxY, boxWidth, boxHeight = openCV.boundingRect(points)
                        openCV.rectangle(
                            resultImage,
                            (boxX, boxY),
                            (boxX + boxWidth, boxY + boxHeight),
                            orange,
                            2,
                        )
                        openCV.circle(
                            resultImage,
                            (boxX + boxWidth // 2, boxY + boxHeight // 2),
                            max(1, min(boxWidth, boxHeight) // 2),
                            neonGreen,
                            2,
                        )
                        headLocated = True

    goodMatches.sort(key=lambda match: match.distance)
    matchingImage = openCV.drawMatches(
        referenceImage,
        referenceKeypoints,
        resultImage,
        frameKeypoints,
        goodMatches[:20],
        None,
        flags=openCV.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    openCV.putText(
        matchingImage,
        f"Matches: {len(goodMatches)} Inliers: {inlierCount} Located: {headLocated}",
        (15, matchingImage.shape[0] - 15),
        openCV.FONT_HERSHEY_SIMPLEX,
        0.5,
        neonGreen,
        2,
    )
    return matchingImage


def processFrame(frame, mode, colorName, orbDetector, referenceData):
    if mode == "original":
        resultImage = frame.copy()
    elif mode == "gray":
        resultImage = convertToGrayscale(frame)
    elif mode == "blur":
        resultImage = blurImage(frame)
    elif mode == "brightness":
        resultImage = openCV.convertScaleAbs(frame, alpha=1.0, beta=35)
    elif mode in ("blue", "green", "red"):
        blueChannel, greenChannel, redChannel = openCV.split(frame)
        resultImage = {"blue": blueChannel, "green": greenChannel, "red": redChannel}[mode]
    elif mode == "canny":
        resultImage = openCV.Canny(blurImage(convertToGrayscale(frame)), 50, 150)
    elif mode in ("binary", "adaptive", "range"):
        resultImage = createBinaryMask(frame, mode)
    elif mode in ("erode", "dilate", "opening", "closing"):
        resultImage = applyMorphology(createBinaryMask(frame), mode)
    elif mode == "color":
        mask = createColorMask(frame, colorName)
        filteredImage = openCV.bitwise_and(frame, frame, mask=mask)
        resultImage = drawContoursAndBoxes(filteredImage, mask)
    elif mode == "contours":
        mask = applyMorphology(createBinaryMask(frame), "opening")
        resultImage = drawContoursAndBoxes(frame, mask)
    elif mode == "features":
        keypoints, descriptors = detectFeatures(frame, orbDetector)
        resultImage = openCV.drawKeypoints(
            frame, keypoints, None, color=neonGreen,
            flags=openCV.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
    elif mode == "match":
        resultImage = matchReference(frame, orbDetector, referenceData)
    else:
        resultImage = transformFrame(frame, mode)

    if len(resultImage.shape) == 2:
        resultImage = openCV.cvtColor(resultImage, openCV.COLOR_GRAY2BGR)

    return resultImage


def drawVideoInformation(frame, mode, fps):
    resultImage = frame.copy()
    labels = [f"Mode: {mode}", f"FPS: {fps:.1f}"]

    for index, label in enumerate(labels):
        textPosition = (15, 30 + index * 27)
        openCV.putText(resultImage, label, textPosition, openCV.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
        openCV.putText(resultImage, label, textPosition, openCV.FONT_HERSHEY_SIMPLEX, 0.65, neonGreen, 2)

    return resultImage


def generateVideoFrames(camera, mode, colorName, orbDetector, referenceData, streamId):
    previousTime = perf_counter()

    try:
        while streamId == streamVersion:
            success, frame = camera.read()
            if not success:
                break

            processedFrame = processFrame(frame, mode, colorName, orbDetector, referenceData)
            currentTime = perf_counter()
            elapsedTime = currentTime - previousTime
            previousTime = currentTime
            fps = 1 / elapsedTime if elapsedTime > 0 else 0
            processedFrame = drawVideoInformation(processedFrame, mode, fps)
            encoded, imageBytes = openCV.imencode(
                ".jpg", processedFrame, [openCV.IMWRITE_JPEG_QUALITY, 80],
            )
            if not encoded:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"\r\n" + imageBytes.tobytes() + b"\r\n"
            )
    finally:
        camera.release()
        cameraLock.release()


def streamVideo(mode: ViewMode = "original", colorName: ColorName = "green"):
    global streamVersion
    streamVersion += 1
    streamId = streamVersion

    if not cameraLock.acquire(timeout=2):
        raise HTTPException(status_code=409, detail="Webcam 0 is already in use")

    if streamId != streamVersion:
        cameraLock.release()
        raise HTTPException(status_code=409, detail="A newer camera stream replaced this one")

    camera = None

    try:
        orbDetector = openCV.ORB_create(nfeatures=800)
        referenceData = loadReference(orbDetector) if mode == "match" else None
        camera = openCamera()
        return StreamingResponse(
            generateVideoFrames(
                camera, mode, colorName, orbDetector, referenceData, streamId,
            ),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store"},
        )
    except Exception:
        if camera is not None:
            camera.release()
        cameraLock.release()
        raise


# Reference selector: Uses your mouse to select an object from a real webcam frame, then saves it for matching.
def captureFrame():
    global streamVersion
    streamVersion += 1

    if not cameraLock.acquire(timeout=2):
        raise HTTPException(status_code=409, detail="Webcam 0 is already in use")

    camera = None

    try:
        camera = openCamera()
        success = False
        frame = None

        for frameNumber in range(8):
            success, frame = camera.read()
            if not success:
                break

        if not success or frame is None:
            raise HTTPException(status_code=503, detail="Unable to capture camera frame")

        saveImage("lesson16-frame.png", frame)
        return FileResponse(
            path=assetsPath / "lesson16-frame.png",
            media_type="image/png",
        )
    finally:
        if camera is not None:
            camera.release()
        cameraLock.release()


def saveReference(headX: int, headY: int, headWidth: int, headHeight: int):
    frame = openCV.imread(str(assetsPath / "lesson16-frame.png"))

    if frame is None:
        raise HTTPException(status_code=404, detail="Capture a webcam frame first")

    imageHeight, imageWidth = frame.shape[:2]

    if (
        headX < 0 or headY < 0
        or headWidth <= 0 or headHeight <= 0
        or headX + headWidth > imageWidth
        or headY + headHeight > imageHeight
    ):
        raise HTTPException(status_code=422, detail="Selection is outside the frame")

    headImage = frame[headY:headY + headHeight, headX:headX + headWidth].copy()
    saveImage("lesson16-reference.png", headImage)

    markedImage = frame.copy()
    openCV.rectangle(
        markedImage,
        (headX, headY),
        (headX + headWidth, headY + headHeight),
        neonGreen,
        2,
    )
    openCV.circle(
        markedImage,
        (headX + headWidth // 2, headY + headHeight // 2),
        max(1, min(headWidth, headHeight) // 2),
        orange,
        2,
    )
    saveImage("lesson16-selection.png", markedImage)

    return {
        "savedReference": "lesson16-reference.png",
        "savedSelection": "lesson16-selection.png",
        "headBox": {
            "x": headX,
            "y": headY,
            "width": headWidth,
            "height": headHeight,
        },
    }


def selectReferenceFromWebcam():
    if not cameraLock.acquire(blocking=False):
        raise RuntimeError("Webcam 0 is already in use")

    camera = None

    try:
        camera = openCamera()
        success = False
        frame = None

        for frameNumber in range(10):
            success, frame = camera.read()
            if not success:
                break

        if not success:
            raise RuntimeError("Unable to capture camera frame")

        headBox = openCV.selectROI(
            "Select Dog Head - Press Enter", frame, showCrosshair=True, fromCenter=False,
        )
        headX, headY, headWidth, headHeight = (int(value) for value in headBox)

        if headWidth <= 0 or headHeight <= 0:
            raise ValueError("No reference region selected")

        headImage = frame[headY:headY + headHeight, headX:headX + headWidth].copy()
        saveImage("lesson16-reference.png", headImage)
        markedImage = frame.copy()
        openCV.rectangle(
            markedImage, (headX, headY), (headX + headWidth, headY + headHeight), neonGreen, 2,
        )
        openCV.circle(
            markedImage,
            (headX + headWidth // 2, headY + headHeight // 2),
            min(headWidth, headHeight) // 2,
            orange,
            2,
        )
        saveImage("lesson16-selection.png", markedImage)
        return {
            "savedReference": "lesson16-reference.png",
            "savedSelection": "lesson16-selection.png",
            "headBox": headBox,
        }
    finally:
        if camera is not None:
            camera.release()
        openCV.destroyAllWindows()
        cameraLock.release()


if __name__ == "__main__":
    print(selectReferenceFromWebcam())
