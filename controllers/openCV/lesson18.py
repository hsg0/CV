# FILE: lesson18.py
# LOCATION: python-CV/controllers/openCV/lesson18.py
# WHY: Understand how all our Computer Vision lessons work together.
# WHAT: Processes live video, detects motion, and tracks objects.
# HOW: Uses OpenCV, MOG2, contours, ORB, and centre matching.
# WHEN: Runs when the browser opens the Lesson 18 video endpoint.
# WHERE: Webcam 0 -> OpenCV -> FastAPI -> Browser.
#
# IMPORTANT:
# This file does NOT depend on Lessons 1 through 17.
# We only reuse config/open_cv/configureCV.py.
#
# EVERY SECTION IS MARKED WITH THE LESSON WHERE IT WAS TAUGHT (LESSON 1 - LESSON 18).

from pathlib import Path  # LESSON 4: Build safe file paths to our assets folder
from threading import Lock  # LESSON 15: One lock so only one stream uses Webcam 0 at a time
from time import perf_counter  # LESSON 15: High-precision clock used to calculate FPS
from typing import Literal  # LESSON 12: Restrict URL options to valid mode and colour names

import numpy as np  # LESSON 13: Number arrays used for images, masks, and matrices

from fastapi import HTTPException  # LESSON 1: Return clear HTTP errors to the browser
from fastapi.responses import FileResponse, StreamingResponse  # LESSON 6 & 15: Serve saved PNGs and live JPEG streams

from config.open_cv.configureCV import configureCV, assetsDirectory  # LESSON 1 & 2: Shared OpenCV setup and assets path


# LESSON 1: Load OpenCV using our centralized configuration.
openCV = configureCV()

# LESSON 2: Use our project's assets directory for saved images.
assetsPath = Path(assetsDirectory)

# LESSON 2: Camera 0 is our primary webcam.
cameraIndex = 0

# LESSON 6: OpenCV uses BGR colour order (Blue, Green, Red), not RGB.
neonGreen = (20, 255, 57)
orange = (0, 165, 255)

# LESSON 11: Ignore moving regions smaller than 900 square pixels to filter out noise.
minimumArea = 900

# LESSON 15: Allow only one webcam stream at a time.
cameraLock = Lock()
activeCamera = None

# LESSON 16 & 17: Track the active stream number so a new stream or Stop cleanly ends the previous loop.
streamVersion = 0

# LESSON 17: Choose how movement is detected (frame differencing or MOG2 background).
MotionMethod = Literal["difference", "background"]

# LESSON 16 & 18: Choose which lesson result to display in the browser.
ViewMode = Literal[
    "original", "gray", "blur", "brightness",
    "blue", "green", "red", "crop",
    "canny", "binary", "adaptive", "range",
    "erode", "dilate", "opening", "closing",
    "color", "contours", "features", "match",
    "flip", "rotate", "affine", "perspective",
    "tracking", "mask", "all"
]

# LESSON 12: Supported HSV colour names for colour detection.
ColorName = Literal["red", "green", "blue", "yellow"]

# LESSON 18: Store live settings so the React UI can switch modes without reopening Webcam 0.
currentView: ViewMode = "tracking"
currentMethod: MotionMethod = "background"
currentColorName: ColorName = "green"
referenceVersion = 0


# ==========================================================
# PART 1 — WEBCAM AND BASIC IMAGE PROCESSING (LESSONS 2, 3, 7, 8, 9, 10)
# ==========================================================

# LESSON 2: Connect to Webcam 0 and verify it opened.
def openCamera():

    # Open the default webcam.
    camera = openCV.VideoCapture(cameraIndex)

    # Stop immediately if the webcam cannot be opened.
    if not camera.isOpened():
        camera.release()

        raise HTTPException(
            status_code=503,
            detail="Webcam 0 is unavailable"
        )

    # Ask for a 640x480 frame size so live processing stays fast.
    camera.set(openCV.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(openCV.CAP_PROP_FRAME_HEIGHT, 480)

    return camera


# LESSON 3: Convert a 3-channel BGR colour frame into a 1-channel grayscale image.
def makeGray(frame):

    return openCV.cvtColor(
        frame,
        openCV.COLOR_BGR2GRAY
    )


# LESSON 7: Apply Gaussian blur to smooth out small pixel noise.
def blurImage(image):

    # (5, 5) is the blur kernel size; 0 lets OpenCV calculate the standard deviation.
    return openCV.GaussianBlur(
        image,
        (5, 5),
        0
    )


# LESSON 8: Detect sharp brightness boundaries using the Canny edge detector.
def detectEdges(frame):

    # Convert to grayscale and blur first so noise does not become false edges.
    grayImage = blurImage(makeGray(frame))

    # Pixels with gradient above 160 are strong edges; between 80 and 160 are kept if connected.
    return openCV.Canny(
        grayImage,
        80,
        160
    )


# LESSON 9: Turn a grayscale image into a black-and-white (binary) mask.
def makeBinaryMask(frame, mode="binary"):

    # Convert to grayscale and blur before thresholding.
    grayImage = blurImage(makeGray(frame))

    if mode == "adaptive":

        # LESSON 9: Calculate a threshold for each small 11x11 pixel neighbourhood.
        return openCV.adaptiveThreshold(
            grayImage,
            255,
            openCV.ADAPTIVE_THRESH_GAUSSIAN_C,
            openCV.THRESH_BINARY,
            11,
            2
        )

    if mode == "range":

        # LESSON 9: Keep only pixels whose brightness is between 80 and 180.
        return openCV.inRange(grayImage, 80, 180)

    # LESSON 9: Binary thresholding with Otsu automatically picks the best cutoff value.
    _, mask = openCV.threshold(
        grayImage,
        0,
        255,
        openCV.THRESH_BINARY + openCV.THRESH_OTSU
    )

    return mask


# LESSON 10: Clean up or reshape white regions in a binary mask.
def applyMorphology(mask, mode):

    # A 3x3 rectangular kernel checks each pixel and its 8 neighbours.
    kernel = openCV.getStructuringElement(
        openCV.MORPH_RECT,
        (3, 3)
    )

    if mode == "erode":

        # LESSON 10: Erosion shrinks white areas and removes tiny white specks.
        return openCV.erode(mask, kernel)

    if mode == "dilate":

        # LESSON 10: Dilation grows white areas and connects nearby regions.
        return openCV.dilate(mask, kernel)

    if mode == "opening":

        # LESSON 10: Opening = erode then dilate (removes small background noise).
        return openCV.morphologyEx(
            mask, openCV.MORPH_OPEN, kernel
        )

    if mode == "closing":

        # LESSON 10: Closing = dilate then erode (fills small dark holes inside objects).
        return openCV.morphologyEx(
            mask, openCV.MORPH_CLOSE, kernel
        )

    return mask


# ==========================================================
# PART 2 — COLOUR AND CONTOURS (LESSONS 11 & 12)
# ==========================================================

# LESSON 12: Create a white mask where pixels match the chosen HSV colour.
def makeColorMask(frame, colorName):

    # Convert BGR into HSV (Hue = colour type, Saturation = intensity, Value = brightness).
    hsvImage = openCV.cvtColor(
        frame,
        openCV.COLOR_BGR2HSV
    )

    # Approximate HSV ranges for each colour. Red wraps around hue 0 and 179.
    colorRanges = {
        "red": [
            ((0, 100, 100), (10, 255, 255)),
            ((170, 100, 100), (179, 255, 255))
        ],
        "green": [
            ((36, 80, 70), (85, 255, 255))
        ],
        "blue": [
            ((90, 80, 70), (130, 255, 255))
        ],
        "yellow": [
            ((20, 100, 100), (35, 255, 255))
        ]
    }

    # Start with a black mask of the same height and width.
    mask = np.zeros(hsvImage.shape[:2], dtype=np.uint8)

    # Combine all HSV ranges for the chosen colour.
    for lowerRange, upperRange in colorRanges[colorName]:

        currentMask = openCV.inRange(
            hsvImage,
            lowerRange,
            upperRange
        )

        mask = openCV.bitwise_or(mask, currentMask)

    # LESSON 10: Use morphological opening to remove tiny colour specks.
    return applyMorphology(mask, "opening")


# LESSON 11: Find the outer boundaries of white mask regions and draw boxes around them.
def drawContourBoxes(frame, mask):

    # Work on a copy so the original frame stays untouched.
    resultImage = frame.copy()

    # LESSON 11: Find external contours from a copy of the binary mask.
    contours, hierarchy = openCV.findContours(
        mask.copy(),
        openCV.RETR_EXTERNAL,
        openCV.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:

        # LESSON 11: Measure the enclosed area in square pixels.
        area = openCV.contourArea(contour)

        # LESSON 11: Filter out small noise contours.
        if area < 500:
            continue

        # LESSON 11: Get the upright bounding box (x, y, width, height).
        x, y, width, height = openCV.boundingRect(contour)

        # LESSON 6 & 11: Draw the exact contour outline in neon green.
        openCV.drawContours(
            resultImage,
            [contour],
            -1,
            neonGreen,
            2
        )

        # LESSON 6 & 11: Draw the rectangular bounding box in orange.
        openCV.rectangle(
            resultImage,
            (x, y),
            (x + width, y + height),
            orange,
            2
        )

    return resultImage


# ==========================================================
# PART 3 — GEOMETRIC TRANSFORMATIONS (LESSONS 4 & 13)
# ==========================================================

# LESSON 4 & 13: Flip, crop, resize, rotate, or warp the camera frame.
def transformFrame(frame, mode):

    # Read frame height (rows) and width (columns).
    height, width = frame.shape[:2]

    if mode == "flip":

        # LESSON 13: Flip code 1 mirrors the image left-to-right.
        return openCV.flip(frame, 1)

    if mode == "crop":

        # LESSON 4: Slice rows [y1:y2] and columns [x1:x2] to keep the centre 70%.
        x = int(width * 0.15)
        y = int(height * 0.15)

        croppedImage = frame[
            y:height - y,
            x:width - x
        ]

        # LESSON 4: Resize the cropped area back to the full frame width and height.
        return openCV.resize(
            croppedImage,
            (width, height)
        )

    if mode == "rotate":

        # LESSON 13: Build a 2x3 matrix to rotate 25 degrees around the centre at 80% scale.
        matrix = openCV.getRotationMatrix2D(
            (width / 2, height / 2),
            25,
            0.8
        )

        return openCV.warpAffine(
            frame,
            matrix,
            (width, height)
        )

    if mode == "affine":

        # LESSON 13: Three point pairs define an affine skew while keeping parallel lines parallel.
        sourcePoints = np.float32([
            [0, 0],
            [width - 1, 0],
            [0, height - 1]
        ])

        destinationPoints = np.float32([
            [width * 0.08, height * 0.08],
            [width * 0.92, height * 0.03],
            [width * 0.03, height * 0.92]
        ])

        matrix = openCV.getAffineTransform(
            sourcePoints,
            destinationPoints
        )

        return openCV.warpAffine(
            frame,
            matrix,
            (width, height)
        )

    if mode == "perspective":

        # LESSON 13: Four corner pairs define a 3x3 perspective tilt matrix.
        sourcePoints = np.float32([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ])

        destinationPoints = np.float32([
            [width * 0.13, height * 0.12],
            [width * 0.82, height * 0.04],
            [width * 0.93, height * 0.90],
            [width * 0.06, height * 0.95]
        ])

        matrix = openCV.getPerspectiveTransform(
            sourcePoints,
            destinationPoints
        )

        return openCV.warpPerspective(
            frame,
            matrix,
            (width, height)
        )

    return frame.copy()


# ==========================================================
# PART 4 — MOTION DETECTION (LESSON 17)
# ==========================================================

# LESSON 17: Detect moving pixels by comparing frames or using a learned MOG2 background.
def createMotionMask(frame, previousGray, backgroundModel, method):

    # LESSON 3 & 7: Convert to grayscale and blur so tiny camera flicker is ignored.
    grayImage = blurImage(makeGray(frame))

    if method == "difference":

        # LESSON 17 (Method 1): Frame differencing compares this frame with the previous frame.
        if previousGray is None:

            # On the very first frame there is no previous image yet.
            return np.zeros_like(grayImage), grayImage

        # Measure absolute brightness change at every pixel.
        difference = openCV.absdiff(
            previousGray,
            grayImage
        )

        # LESSON 9 & 17: Pixels that changed by more than 25 become white (255).
        _, mask = openCV.threshold(
            difference,
            25,
            255,
            openCV.THRESH_BINARY
        )

    else:

        # LESSON 17 (Method 2): MOG2 learns the still background over time.
        foregroundMask = backgroundModel.apply(frame)

        # MOG2 marks shadows as gray (127); threshold at 200 to keep only moving foreground (255).
        _, mask = openCV.threshold(
            foregroundMask,
            200,
            255,
            openCV.THRESH_BINARY
        )

    return mask, grayImage


# LESSON 10 & 17: Remove small specks and connect nearby moving regions.
def cleanMotionMask(mask):

    # Opening removes isolated white noise dots.
    cleanedMask = applyMorphology(mask, "opening")

    # Dilation expands white areas so parts of one moving object join together.
    cleanedMask = applyMorphology(
        cleanedMask,
        "dilate"
    )

    return cleanedMask


# LESSON 11 & 17: Find bounding boxes around moving white regions in the motion mask.
def findMovingBoxes(mask):

    contours, hierarchy = openCV.findContours(
        mask.copy(),
        openCV.RETR_EXTERNAL,
        openCV.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    # Check the largest moving regions first.
    contours = sorted(
        contours,
        key=openCV.contourArea,
        reverse=True
    )

    for contour in contours:

        # LESSON 11: Ignore regions smaller than minimumArea (900 square pixels).
        if openCV.contourArea(contour) < minimumArea:
            continue

        x, y, width, height = openCV.boundingRect(contour)
        boxes.append((x, y, width, height))

        # Keep at most 20 moving boxes per frame so tracking stays fast and readable.
        if len(boxes) == 20:
            break

    return boxes


# ==========================================================
# PART 5 — OBJECT TRACKING ACROSS VIDEO FRAMES (LESSON 18)
# ==========================================================

# LESSON 18: Match current moving boxes with remembered objects from earlier frames.
def updateTracks(boxes, tracks, nextId):

    # LESSON 18:
    # tracks stores each active object by its ID:
    # tracks[objectId] = {
    #     "center": (centerX, centerY),
    #     "box": (x, y, width, height),
    #     "missing": 0,
    #     "history": [(centerX, centerY), ...]
    # }
    matchedIds = set()

    # An object must be within 80 pixels of its previous centre to keep the same ID.
    maximumDistance = 80

    for box in boxes:

        x, y, width, height = box

        # LESSON 18: Calculate the centre point of the current bounding box.
        centerX = x + width // 2
        centerY = y + height // 2
        currentCenter = (centerX, centerY)

        closestId = None
        closestDistance = maximumDistance ** 2

        # LESSON 18: Compare this box's centre with every remembered object's centre.
        for objectId, objectData in tracks.items():

            # Each remembered ID can match at most one box per frame.
            if objectId in matchedIds:
                continue

            previousX, previousY = objectData["center"]

            # Squared distance avoids a slow square-root calculation.
            distance = (
                (centerX - previousX) ** 2
                + (centerY - previousY) ** 2
            )

            if distance < closestDistance:
                closestDistance = distance
                closestId = objectId

        if closestId is None:

            # LESSON 18: No existing object was close enough, so start a new track ID.
            objectId = nextId
            nextId += 1

            tracks[objectId] = {
                "center": currentCenter,
                "box": box,
                "missing": 0,
                "history": [currentCenter]
            }

        else:

            # LESSON 18: Reuse the closest object's ID and update its position and trail.
            objectId = closestId
            tracks[objectId]["center"] = currentCenter
            tracks[objectId]["box"] = box
            tracks[objectId]["missing"] = 0
            tracks[objectId]["history"].append(currentCenter)

            # Keep the most recent 20 centre points for the movement trail.
            tracks[objectId]["history"] = (
                tracks[objectId]["history"][-20:]
            )

        matchedIds.add(objectId)

    # LESSON 18: Count missed frames for objects that were not seen in this frame.
    for objectId in list(tracks):

        if objectId not in matchedIds:
            tracks[objectId]["missing"] += 1

            # Remove the track if the object has been gone for more than 10 frames.
            if tracks[objectId]["missing"] > 10:
                del tracks[objectId]

    return nextId


# LESSON 6 & 18: Draw each tracked object's box, persistent ID, centre dot, and trail.
def drawTracks(frame, tracks):

    resultImage = frame.copy()
    visibleObjects = 0

    for objectId, objectData in tracks.items():

        # Only draw objects that are visible in the current frame.
        if objectData["missing"] > 0:
            continue

        visibleObjects += 1
        x, y, width, height = objectData["box"]

        # LESSON 6: Draw the bounding box around the tracked object.
        openCV.rectangle(
            resultImage,
            (x, y),
            (x + width, y + height),
            neonGreen,
            2
        )

        # LESSON 6 & 18: Draw the persistent tracking ID above the box.
        openCV.putText(
            resultImage,
            f"ID {objectId}",
            (x, max(90, y - 10)),
            openCV.FONT_HERSHEY_SIMPLEX,
            0.6,
            neonGreen,
            2
        )

        centerX, centerY = objectData["center"]

        # LESSON 6: Draw a filled circle at the object's centre.
        openCV.circle(
            resultImage,
            (centerX, centerY),
            5,
            neonGreen,
            -1
        )

        # LESSON 18: Connect the object's past centre points to show its path.
        history = objectData["history"]

        for index in range(1, len(history)):

            openCV.line(
                resultImage,
                history[index - 1],
                history[index],
                neonGreen,
                2
            )

    return resultImage, visibleObjects


# ==========================================================
# PART 6 — ORB FEATURE MATCHING (LESSON 14)
# ==========================================================

# LESSON 14: Detect ORB keypoints and their binary descriptors.
def detectFeatures(frame, orbDetector):

    grayImage = makeGray(frame)

    return orbDetector.detectAndCompute(
        grayImage,
        None
    )


# LESSON 14 & 16: Load our saved reference crop and compute its ORB features once.
def loadReference(orbDetector):

    # Prefer the Lesson 18 reference crop, or fall back to earlier lesson crops if needed.
    referencePath = assetsPath / "lesson18-reference.png"
    if not referencePath.is_file():
        referencePath = assetsPath / "lesson16-reference.png"
    if not referencePath.is_file():
        referencePath = assetsPath / "6-head-crop.png"

    referenceImage = openCV.imread(str(referencePath))

    if referenceImage is None:
        raise HTTPException(
            status_code=422,
            detail="Create the Lesson 18 reference image first"
        )

    keypoints, descriptors = detectFeatures(
        referenceImage,
        orbDetector
    )

    if descriptors is None or len(keypoints) < 8:
        raise HTTPException(
            status_code=422,
            detail="Reference image has too few visual features"
        )

    return referenceImage, keypoints, descriptors


# LESSON 14: Match saved reference descriptors against the live webcam frame.
def matchReference(frame, orbDetector, referenceData):

    referenceImage, referenceKeys, referenceDescriptors = referenceData
    frameKeys, frameDescriptors = detectFeatures(frame, orbDetector)
    resultImage = frame.copy()

    goodMatches = []
    inlierCount = 0

    if frameDescriptors is not None:

        # LESSON 14: Compare binary ORB descriptors using Hamming distance.
        matcher = openCV.BFMatcher(
            openCV.NORM_HAMMING,
            crossCheck=False
        )

        candidates = matcher.knnMatch(
            referenceDescriptors,
            frameDescriptors,
            k=2
        )

        for pair in candidates:

            if len(pair) < 2:
                continue

            bestMatch, secondMatch = pair

            # LESSON 14: Lowe's ratio test keeps only clear, unmistakable matches.
            if bestMatch.distance < 0.75 * secondMatch.distance:
                goodMatches.append(bestMatch)

        # LESSON 14: Use RANSAC homography when at least 8 good matches exist.
        if len(goodMatches) >= 8:

            referencePoints = np.float32([
                referenceKeys[match.queryIdx].pt
                for match in goodMatches
            ]).reshape(-1, 1, 2)

            framePoints = np.float32([
                frameKeys[match.trainIdx].pt
                for match in goodMatches
            ]).reshape(-1, 1, 2)

            try:
                matrix, inlierMask = openCV.findHomography(
                    referencePoints,
                    framePoints,
                    openCV.RANSAC,
                    5.0
                )
            except openCV.error:
                matrix, inlierMask = None, None

            if matrix is not None and inlierMask is not None:

                inlierCount = int(inlierMask.sum())

                if inlierCount >= 8:

                    height, width = referenceImage.shape[:2]

                    corners = np.float32([
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1],
                        [0, height - 1]
                    ]).reshape(-1, 1, 2)

                    # LESSON 13 & 14: Project the reference corners onto the live frame.
                    projectedCorners = openCV.perspectiveTransform(
                        corners,
                        matrix
                    )

                    frameHeight, frameWidth = frame.shape[:2]

                    validCorners = (
                        np.isfinite(projectedCorners).all()
                        and np.max(np.abs(projectedCorners))
                        < 3 * max(frameWidth, frameHeight)
                    )

                    if validCorners:

                        area = abs(openCV.contourArea(projectedCorners))

                        if area >= 100:

                            points = np.int32(projectedCorners)

                            # LESSON 6 & 14: Outline the located object in neon green.
                            openCV.polylines(
                                resultImage,
                                [points],
                                True,
                                neonGreen,
                                3
                            )

    goodMatches.sort(key=lambda match: match.distance)

    # LESSON 14: Draw lines connecting the top 25 matched keypoints side by side.
    matchingImage = openCV.drawMatches(
        referenceImage,
        referenceKeys,
        resultImage,
        frameKeys,
        goodMatches[:25],
        None,
        flags=openCV.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    openCV.putText(
        matchingImage,
        f"Matches: {len(goodMatches)} Inliers: {inlierCount}",
        (15, matchingImage.shape[0] - 15),
        openCV.FONT_HERSHEY_SIMPLEX,
        0.5,
        neonGreen,
        2
    )

    return matchingImage


# ==========================================================
# PART 7 — SELECT THE PROCESSING MODE (LESSONS 3–18)
# ==========================================================

# LESSON 16 & 18: Run the chosen lesson mode on the current webcam frame.
def processFrame(
    frame, motionMask, tracks,
    view, colorName, orbDetector, referenceData
):

    if view == "original":
        # LESSON 2: Raw webcam frame.
        resultImage = frame.copy()

    elif view == "gray":
        # LESSON 3: Grayscale image.
        resultImage = makeGray(frame)

    elif view == "blur":
        # LESSON 7: Gaussian blurred image.
        resultImage = blurImage(frame)

    elif view == "brightness":
        # LESSON 7: Increase brightness by adding 35 to every pixel.
        resultImage = openCV.convertScaleAbs(
            frame,
            alpha=1.0,
            beta=35
        )

    elif view in ("blue", "green", "red"):
        # LESSON 4: Split the BGR image into single colour channels.
        blueChannel, greenChannel, redChannel = openCV.split(frame)

        channels = {
            "blue": blueChannel,
            "green": greenChannel,
            "red": redChannel
        }

        resultImage = channels[view]

    elif view == "canny":
        # LESSON 8: Canny edge detection.
        resultImage = detectEdges(frame)

    elif view in ("binary", "adaptive", "range"):
        # LESSON 9: Thresholding masks.
        resultImage = makeBinaryMask(frame, view)

    elif view in ("erode", "dilate", "opening", "closing"):
        # LESSON 10: Morphological operations on a binary mask.
        mask = makeBinaryMask(frame)
        resultImage = applyMorphology(mask, view)

    elif view == "color":
        # LESSON 12: Keep pixels matching the chosen HSV colour and draw their boxes.
        colorMask = makeColorMask(frame, colorName)

        filteredImage = openCV.bitwise_and(
            frame,
            frame,
            mask=colorMask
        )

        resultImage = drawContourBoxes(
            filteredImage,
            colorMask
        )

    elif view == "contours":
        # LESSON 11: Draw contours and bounding boxes from a binary mask.
        mask = makeBinaryMask(frame)
        resultImage = drawContourBoxes(frame, mask)

    elif view == "features":
        # LESSON 14: Draw ORB keypoints on the live frame.
        keypoints, descriptors = detectFeatures(
            frame,
            orbDetector
        )

        resultImage = openCV.drawKeypoints(
            frame,
            keypoints,
            None,
            color=neonGreen,
            flags=openCV.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

    elif view == "match":
        # LESSON 14: Match the saved reference image against the live frame.
        resultImage = matchReference(
            frame,
            orbDetector,
            referenceData
        )

    elif view == "mask":
        # LESSON 17: Show the black-and-white motion mask.
        resultImage = motionMask.copy()

    elif view == "tracking":
        # LESSON 18: Draw tracked moving objects with persistent IDs and trails.
        resultImage, visibleObjects = drawTracks(
            frame,
            tracks
        )

    elif view == "all":
        # LESSON 18 CAPSTONE: Combine edges, colour contours, ORB features, and tracking.
        resultImage = frame.copy()

        # LESSON 8: Colour detected Canny edges blue.
        edges = detectEdges(frame)
        resultImage[edges > 0] = (255, 0, 0)

        # LESSON 11 & 12: Outline regions matching the selected HSV colour.
        colorMask = makeColorMask(frame, colorName)
        resultImage = drawContourBoxes(
            resultImage,
            colorMask
        )

        # LESSON 14: Draw ORB keypoints in orange.
        keypoints, descriptors = detectFeatures(
            frame,
            orbDetector
        )
        resultImage = openCV.drawKeypoints(
            resultImage,
            keypoints,
            None,
            color=orange
        )

        # LESSON 18: Draw tracked object boxes, IDs, and trails on top.
        resultImage, visibleObjects = drawTracks(
            resultImage,
            tracks
        )

    else:
        # LESSON 4 & 13: Apply geometric transformations (flip, crop, rotate, affine, perspective).
        resultImage = transformFrame(frame, view)

    # LESSON 3 & 6: Convert 1-channel grayscale masks back to 3-channel BGR so coloured text can be drawn.
    if len(resultImage.shape) == 2:
        resultImage = openCV.cvtColor(
            resultImage,
            openCV.COLOR_GRAY2BGR
        )

    return resultImage


# ==========================================================
# PART 8 — STREAM LIVE VIDEO (LESSONS 15, 17, 18)
# ==========================================================

# LESSON 15, 17 & 18: Read webcam frames, update motion tracks, and yield JPEG bytes.
def generateVideoFrames(
    camera, view, method, colorName, orbDetector, referenceData, streamId
):
    global activeCamera
    activeCamera = camera

    # LESSON 17: Store the previous grayscale frame for frame differencing.
    previousGray = None
    lastMethod = method
    loadedReferenceVersion = referenceVersion

    # LESSON 17: Create the MOG2 background model.
    backgroundModel = openCV.createBackgroundSubtractorMOG2(
        history=200,
        varThreshold=25,
        detectShadows=True
    )

    # LESSON 18: Dictionary of tracked objects and the next ID number to assign.
    tracks = {}
    nextId = 1
    frameNumber = 0

    # LESSON 15: Starting timestamp for FPS calculation.
    previousTime = perf_counter()

    try:

        # Keep streaming until the browser disconnects or Stop is called.
        while streamId == streamVersion:

            # LESSON 2: Capture the next frame from Webcam 0.
            success, frame = camera.read()

            if not success:
                break

            frameNumber += 1

            # LESSON 18: Read the latest mode, motion method, and HSV colour chosen in the UI.
            view = currentView
            method = currentMethod
            colorName = currentColorName

            if method != lastMethod:
                previousGray = None
                lastMethod = method

            # LESSON 14: Load or refresh the reference image when match mode is selected.
            if view == "match" and (referenceData is None or loadedReferenceVersion != referenceVersion):
                referenceData = loadReference(orbDetector)
                loadedReferenceVersion = referenceVersion

            # LESSON 17: Build and clean the motion mask.
            motionMask, previousGray = createMotionMask(
                frame,
                previousGray,
                backgroundModel,
                method
            )

            motionMask = cleanMotionMask(motionMask)

            # Let MOG2 settle for the first 30 frames before reporting motion.
            if method == "background" and frameNumber <= 30:
                motionMask[:] = 0

            # LESSON 17: Find bounding boxes around moving regions.
            boxes = findMovingBoxes(motionMask)

            # LESSON 18: Match moving boxes to tracked object IDs.
            nextId = updateTracks(
                boxes,
                tracks,
                nextId
            )

            # LESSON 16 & 18: Build the display image for the selected view mode.
            resultImage = processFrame(
                frame,
                motionMask,
                tracks,
                view,
                colorName,
                orbDetector,
                referenceData
            )

            # LESSON 18: Count how many tracked objects are visible right now.
            visibleObjects = sum(
                objectData["missing"] == 0
                for objectData in tracks.values()
            )

            # LESSON 15: Measure frames per second.
            currentTime = perf_counter()
            elapsedTime = currentTime - previousTime
            previousTime = currentTime
            fps = 1 / elapsedTime if elapsedTime > 0 else 0

            # LESSON 6 & 15: Draw the status overlay in the top-left corner.
            labels = [
                f"Mode: {view} | Motion: {method}",
                f"Tracked regions: {visibleObjects}",
                f"FPS: {fps:.1f}"
            ]

            for index, label in enumerate(labels):

                textPosition = (15, 25 + index * 25)

                openCV.putText(
                    resultImage,
                    label,
                    textPosition,
                    openCV.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    4
                )

                openCV.putText(
                    resultImage,
                    label,
                    textPosition,
                    openCV.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    neonGreen,
                    2
                )

            # LESSON 2 & 15: Encode the finished frame as a JPEG in memory.
            encoded, imageBytes = openCV.imencode(
                ".jpg",
                resultImage,
                [openCV.IMWRITE_JPEG_QUALITY, 80]
            )

            if not encoded:
                continue

            # LESSON 15: Yield one multipart MJPEG frame to the browser.
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"\r\n"
                + imageBytes.tobytes()
                + b"\r\n"
            )

    finally:

        # LESSON 2 & 15: Always release the webcam and lock when the stream stops.
        activeCamera = None
        camera.release()
        cameraLock.release()


# ==========================================================
# PART 9 — MAIN FASTAPI CONTROLLERS (LESSON 15 & 18)
# ==========================================================

# LESSON 18: Update the active view, motion method, or HSV colour without reopening Webcam 0.
def updateVideoSettings(
    view: ViewMode = "tracking",
    method: MotionMethod = "background",
    colorName: ColorName = "green"
):
    global currentView, currentMethod, currentColorName
    currentView = view
    currentMethod = method
    currentColorName = colorName

    return {
        "view": currentView,
        "method": currentMethod,
        "colorName": currentColorName
    }


# LESSON 15 & 17: Stop the active stream and release Webcam 0 immediately.
def stopVideoStream():
    global streamVersion, activeCamera
    streamVersion += 1

    camera = activeCamera
    if camera is not None:
        camera.release()

    if cameraLock.acquire(timeout=3):
        cameraLock.release()

    return {"camera": "released"}


# LESSON 15 & 18: Open the webcam and return the live multipart stream.
def streamVideo(
    view: ViewMode = "tracking",
    method: MotionMethod = "background",
    colorName: ColorName = "green"
):
    global streamVersion, currentView, currentMethod, currentColorName
    streamVersion += 1
    streamId = streamVersion
    currentView = view
    currentMethod = method
    currentColorName = colorName

    # Wait briefly for any previous stream to release Webcam 0.
    if not cameraLock.acquire(timeout=2):
        raise HTTPException(
            status_code=409,
            detail="Webcam 0 is already in use"
        )

    if streamId != streamVersion:
        cameraLock.release()
        raise HTTPException(
            status_code=409,
            detail="A newer camera stream replaced this one"
        )

    camera = None

    try:

        # LESSON 14: Create one ORB detector for this stream.
        orbDetector = openCV.ORB_create(nfeatures=700)

        # Only load the reference image when match mode is selected.
        referenceData = None

        if view == "match":
            referenceData = loadReference(orbDetector)

        # LESSON 2: Open Webcam 0.
        camera = openCamera()

        # LESSON 15: Stream live JPEG frames to the browser.
        return StreamingResponse(
            generateVideoFrames(
                camera,
                view,
                method,
                colorName,
                orbDetector,
                referenceData,
                streamId
            ),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store"}
        )

    except Exception:

        if camera is not None:
            camera.release()

        cameraLock.release()
        raise


# ==========================================================
# PART 10 — REFERENCE SELECTION (LESSONS 2, 4, 5, 6, 14)
# ==========================================================

# LESSON 2 & 16: Capture one clean frame from Webcam 0 so the browser UI can draw a selection box.
def captureReferenceFrame():
    stopVideoStream()

    if not cameraLock.acquire(timeout=2):
        raise HTTPException(
            status_code=409,
            detail="Webcam 0 is already in use"
        )

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
            raise HTTPException(
                status_code=503,
                detail="Unable to capture a frame"
            )

        framePath = assetsPath / "lesson18-frame.png"
        if not openCV.imwrite(str(framePath), frame):
            raise HTTPException(
                status_code=500,
                detail="Cannot save captured frame"
            )

        return FileResponse(
            path=framePath,
            media_type="image/png",
            headers={"Cache-Control": "no-store"}
        )

    finally:
        if camera is not None:
            camera.release()
        cameraLock.release()


# LESSON 4, 6 & 14: Crop and save the reference region selected in the browser UI.
def saveReferenceSelection(x: int, y: int, width: int, height: int):
    global referenceVersion
    framePath = assetsPath / "lesson18-frame.png"
    frame = openCV.imread(str(framePath))

    if frame is None:
        raise HTTPException(
            status_code=404,
            detail="Capture a webcam frame first"
        )

    frameHeight, frameWidth = frame.shape[:2]

    if (
        x < 0 or y < 0
        or width <= 0 or height <= 0
        or x + width > frameWidth
        or y + height > frameHeight
    ):
        raise HTTPException(
            status_code=422,
            detail="Selection is outside the frame"
        )

    # LESSON 4: Crop the selected region.
    referenceImage = frame[y:y + height, x:x + width].copy()
    referencePath = assetsPath / "lesson18-reference.png"

    if not openCV.imwrite(str(referencePath), referenceImage):
        raise HTTPException(
            status_code=500,
            detail="Cannot save reference image"
        )

    # LESSON 6: Draw a rectangle and centre circle on a copy of the frame.
    markedImage = frame.copy()
    openCV.rectangle(
        markedImage,
        (x, y),
        (x + width, y + height),
        neonGreen,
        2
    )
    openCV.circle(
        markedImage,
        (x + width // 2, y + height // 2),
        max(1, min(width, height) // 2),
        orange,
        2
    )

    selectionPath = assetsPath / "lesson18-selection.png"
    if not openCV.imwrite(str(selectionPath), markedImage):
        raise HTTPException(
            status_code=500,
            detail="Cannot save selection image"
        )

    referenceVersion += 1

    return {
        "reference": "lesson18-reference.png",
        "selection": "lesson18-selection.png",
        "box": {"x": x, "y": y, "width": width, "height": height}
    }


# LESSON 5, 6 & 14: Capture a webcam frame, select a box with the mouse, and save it for ORB matching.
def selectReferenceFromWebcam():

    if not cameraLock.acquire(blocking=False):
        raise RuntimeError("Webcam 0 is already in use")

    camera = None

    try:

        # LESSON 2: Open Webcam 0 and let the exposure settle.
        camera = openCamera()
        success = False
        frame = None

        for frameNumber in range(10):
            success, frame = camera.read()

            if not success:
                break

        if not success:
            raise RuntimeError("Unable to capture a frame")

        # LESSON 5: Drag a box around the object and press Enter.
        headBox = openCV.selectROI(
            "Select Reference - Press Enter",
            frame,
            showCrosshair=True,
            fromCenter=False
        )

        x, y, width, height = [int(value) for value in headBox]

        if width <= 0 or height <= 0:
            raise ValueError("No reference was selected")

        # LESSON 4: Crop the selected box from the frame.
        referenceImage = frame[
            y:y + height,
            x:x + width
        ].copy()

        # LESSON 3 & 14: Save the cropped reference image in assets/.
        referencePath = assetsPath / "lesson18-reference.png"

        if not openCV.imwrite(str(referencePath), referenceImage):
            raise RuntimeError("Cannot save reference image")

        markedImage = frame.copy()

        # LESSON 6: Draw a neon-green rectangle and an orange circle on the selection preview.
        openCV.rectangle(
            markedImage,
            (x, y),
            (x + width, y + height),
            neonGreen,
            2
        )

        centerX = x + width // 2
        centerY = y + height // 2

        openCV.circle(
            markedImage,
            (centerX, centerY),
            min(width, height) // 2,
            orange,
            2
        )

        selectionPath = assetsPath / "lesson18-selection.png"

        if not openCV.imwrite(str(selectionPath), markedImage):
            raise RuntimeError("Cannot save selection image")

        return {
            "reference": str(referencePath),
            "selection": str(selectionPath),
            "box": [x, y, width, height]
        }

    finally:

        # LESSON 2 & 5: Always release the camera and close desktop windows.
        if camera is not None:
            camera.release()

        openCV.destroyAllWindows()
        cameraLock.release()


if __name__ == "__main__":

    result = selectReferenceFromWebcam()
    print(result)
