# FILE: lesson14.py
# LOCATION: python-CV/controllers/openCV/lesson14.py
# WHY: Learn how to match a dog's head between two images.
# WHAT: Draws head markers, detects features, and matches images.
# HOW: Uses Canny, ORB, BFMatcher, RANSAC, and homography.
# WHEN: Runs locally or through the Lesson 14 FastAPI route.
# WHERE: Reads assets/6.png and saves results in assets/.

from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import HTTPException
from fastapi.responses import FileResponse
from config.open_cv.configureCV import assetsDirectory, configureCV

# OpenCV uses blue, green, red order.
openCV = configureCV()
assetsPath = Path(assetsDirectory)
neonGreen = (20, 255, 57)
orange = (0, 165, 255)

# These are the saved pictures the route can return.
ImageType = Literal[
    "marked",
    "crop",
    "edges",
    "features",
    "target",
    "matches",
    "located",
]


# Step 1. Load the original photo.
def loadImage():
    imagePath = assetsPath / "6.png"
    image = openCV.imread(str(imagePath))

    if image is None:
        raise HTTPException(
            status_code=404,
            detail="Cannot read image: 6.png"
        )

    return image


# Local only: drag a box around the head, then press Enter.
def selectDogHead(image):
    headBox = openCV.selectROI(
        "Select Dog Head",
        image,
        showCrosshair=True,
        fromCenter=False,
    )
    openCV.destroyAllWindows()
    return tuple(int(value) for value in headBox)


# Step 2. Draw a rectangle and a circle on the selected head.
def drawDogHead(image, headBox):
    markedImage = image.copy()
    headX, headY, headWidth, headHeight = headBox
    centerX = headX + headWidth // 2
    centerY = headY + headHeight // 2
    radius = min(headWidth, headHeight) // 2

    openCV.rectangle(
        markedImage,
        (headX, headY),
        (headX + headWidth, headY + headHeight),
        neonGreen,
        2,
    )
    openCV.circle(markedImage, (centerX, centerY), radius, orange, 2)
    openCV.putText(
        markedImage,
        "DOG HEAD",
        (headX, max(25, headY - 10)),
        openCV.FONT_HERSHEY_SIMPLEX,
        0.7,
        neonGreen,
        2,
    )
    return markedImage


# Step 3. Make a rotated, smaller copy to search inside.
def createTargetImage(image):
    height, width = image.shape[:2]
    rotationMatrix = openCV.getRotationMatrix2D(
        (width / 2, height / 2),
        25,
        0.8,
    )
    return openCV.warpAffine(image, rotationMatrix, (width, height))


# Step 4. Find edges on the cropped head.
def detectEdges(image):
    grayImage = openCV.cvtColor(image, openCV.COLOR_BGR2GRAY)
    return openCV.Canny(grayImage, 100, 200)


# Step 5. Find ORB points and the pattern around each point.
def detectFeatures(image):
    grayImage = openCV.cvtColor(image, openCV.COLOR_BGR2GRAY)
    orbDetector = openCV.ORB_create(nfeatures=1500)
    keypoints, descriptors = orbDetector.detectAndCompute(grayImage, None)

    if keypoints is None:
        keypoints = []

    return keypoints, descriptors


# Step 6. Keep matches that are clearly better than the next choice.
def compareDescriptors(headDescriptors, targetDescriptors):
    if headDescriptors is None or targetDescriptors is None:
        return []

    matcher = openCV.BFMatcher(openCV.NORM_HAMMING, crossCheck=False)
    candidateMatches = matcher.knnMatch(headDescriptors, targetDescriptors, k=2)
    goodMatches = []

    for candidatePair in candidateMatches:
        if len(candidatePair) < 2:
            continue

        bestMatch, secondMatch = candidatePair

        if bestMatch.distance < 0.75 * secondMatch.distance:
            goodMatches.append(bestMatch)

    goodMatches.sort(key=lambda match: match.distance)
    return goodMatches


# Step 7. Draw where the head crop landed in the rotated picture.
def locateDogHead(headImage, targetImage, headKeypoints, targetKeypoints, goodMatches):
    locatedImage = targetImage.copy()
    inlierCount = 0
    headLocated = False

    # Four pairs are the minimum. We ask for eight so the box is better supported.
    if len(goodMatches) < 8:
        return locatedImage, inlierCount, headLocated

    headPoints = np.float32([
        headKeypoints[match.queryIdx].pt
        for match in goodMatches
    ]).reshape(-1, 1, 2)
    targetPoints = np.float32([
        targetKeypoints[match.trainIdx].pt
        for match in goodMatches
    ]).reshape(-1, 1, 2)

    homographyMatrix, inlierMask = openCV.findHomography(
        headPoints,
        targetPoints,
        openCV.RANSAC,
        5.0,
    )

    if homographyMatrix is None or inlierMask is None:
        return locatedImage, inlierCount, headLocated

    inlierCount = int(inlierMask.sum())

    if inlierCount < 8:
        return locatedImage, inlierCount, headLocated

    headHeight, headWidth = headImage.shape[:2]
    headCorners = np.float32([
        [0, 0],
        [headWidth - 1, 0],
        [headWidth - 1, headHeight - 1],
        [0, headHeight - 1],
    ]).reshape(-1, 1, 2)
    transformedCorners = openCV.perspectiveTransform(headCorners, homographyMatrix)

    if not np.isfinite(transformedCorners).all():
        return locatedImage, inlierCount, headLocated

    openCV.polylines(
        locatedImage,
        [np.int32(transformedCorners)],
        True,
        neonGreen,
        3,
    )
    return locatedImage, inlierCount, True


def saveImage(fileName, image):
    savePath = assetsPath / fileName

    if not openCV.imwrite(str(savePath), image):
        raise HTTPException(
            status_code=500,
            detail=f"Cannot save image: {fileName}"
        )

    return fileName


# Run the whole lesson from a head box: x, y, width, height.
def processFeatureMatching(headX: int, headY: int, headWidth: int, headHeight: int):
    try:
        originalImage = loadImage()
        imageHeight, imageWidth = originalImage.shape[:2]

        # The box must sit inside the photo.
        if (
            headX < 0
            or headY < 0
            or headWidth <= 0
            or headHeight <= 0
            or headX + headWidth > imageWidth
            or headY + headHeight > imageHeight
        ):
            raise HTTPException(
                status_code=422,
                detail="Head selection is outside the image"
            )

        headBox = (headX, headY, headWidth, headHeight)
        markedImage = drawDogHead(originalImage, headBox)

        # Crop the unmarked photo so the drawn lines are not matched.
        headImage = originalImage[headY:headY + headHeight, headX:headX + headWidth].copy()
        edges = detectEdges(headImage)
        targetImage = createTargetImage(originalImage)
        headKeypoints, headDescriptors = detectFeatures(headImage)
        targetKeypoints, targetDescriptors = detectFeatures(targetImage)
        featureImage = openCV.drawKeypoints(
            headImage,
            headKeypoints,
            None,
            color=neonGreen,
            flags=openCV.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        goodMatches = compareDescriptors(headDescriptors, targetDescriptors)
        matchingImage = openCV.drawMatches(
            headImage,
            headKeypoints,
            targetImage,
            targetKeypoints,
            goodMatches[:60],
            None,
            flags=openCV.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        locatedImage, inlierCount, headLocated = locateDogHead(
            headImage,
            targetImage,
            headKeypoints,
            targetKeypoints,
            goodMatches,
        )

        savedImages = {
            "marked": saveImage("6-head-marked.png", markedImage),
            "crop": saveImage("6-head-crop.png", headImage),
            "edges": saveImage("6-head-edges.png", edges),
            "features": saveImage("6-head-features.png", featureImage),
            "target": saveImage("6-head-target.png", targetImage),
            "matches": saveImage("6-head-matches.png", matchingImage),
            "located": saveImage("6-head-located.png", locatedImage),
        }

        return {
            "lesson": 14,
            "headBox": {
                "x": headX,
                "y": headY,
                "width": headWidth,
                "height": headHeight,
            },
            "headKeypoints": len(headKeypoints),
            "targetKeypoints": len(targetKeypoints),
            "goodMatches": len(goodMatches),
            "homographyInliers": inlierCount,
            "headLocated": headLocated,
            "savedImages": savedImages,
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Unable to match the dog head"
        ) from error


# Return one saved picture. Process the lesson before calling this.
def getFeatureImage(imageType: ImageType = "matches"):
    imagePath = assetsPath / f"6-head-{imageType}.png"

    if not imagePath.is_file():
        raise HTTPException(
            status_code=404,
            detail="Process Lesson 14 first"
        )

    return FileResponse(path=imagePath, media_type="image/png")


# Run from the terminal to choose the head with the mouse.
if __name__ == "__main__":
    image = loadImage()
    headX, headY, headWidth, headHeight = selectDogHead(image)
    print(processFeatureMatching(headX, headY, headWidth, headHeight))
