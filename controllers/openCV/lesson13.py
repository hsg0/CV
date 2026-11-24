# FILE: lesson13.py
# LOCATION: python-CV/controllers/openCV/lesson13.py
# WHY: Learn how to change image position, size, angle, and perspective.
# WHAT: Creates seven geometric transformation results.
# HOW: Uses OpenCV functions, transformation matrices, and NumPy.
# WHEN: Called by the Lesson 13 FastAPI routes.
# WHERE: Reads assets/6.png and saves results inside assets/.

from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import HTTPException
from fastapi.responses import FileResponse
from config.open_cv.configureCV import assetsDirectory, configureCV

# Load our configured OpenCV library.
openCV = configureCV()

# Identify where our images are stored.
assetsPath = Path(assetsDirectory)

# Allow only these processed image types.
ImageType = Literal[
    "translated",
    "rotated",
    "flipped",
    "scaled",
    "affine",
    "perspective",
    "corrected",
]


# Load the original image.
def loadImage():
    imagePath = assetsPath / "6.png"
    image = openCV.imread(str(imagePath))

    if image is None:
        raise HTTPException(
            status_code=404,
            detail="Cannot read image: 6.png"
        )

    return image


# Move an image horizontally and vertically.
def translateImage(image):
    height, width = image.shape[:2]

    # Move right by 12% and down by 8%.
    moveX = int(width * 0.12)
    moveY = int(height * 0.08)

    # First row moves pixels on x. Second row moves pixels on y.
    translationMatrix = np.float32([
        [1, 0, moveX],
        [0, 1, moveY],
    ])

    # Pixels moved outside the frame are lost.
    return openCV.warpAffine(
        image,
        translationMatrix,
        (width, height),
    )


# Rotate an image around its centre.
def rotateImage(image):
    height, width = image.shape[:2]
    centerX = width / 2
    centerY = height / 2

    # Rotate 25 degrees counterclockwise and keep the original scale.
    rotationMatrix = openCV.getRotationMatrix2D(
        (centerX, centerY),
        25,
        1.0,
    )

    # Some corners may be clipped by the original canvas.
    return openCV.warpAffine(
        image,
        rotationMatrix,
        (width, height),
    )


# Mirror an image horizontally.
# Flip code 1 is horizontal, 0 is vertical, and -1 is both.
def flipImage(image):
    return openCV.flip(image, 1)


# Make an image smaller without stretching it.
def scaleImage(image):
    # INTER_AREA works well when shrinking images.
    return openCV.resize(
        image,
        None,
        fx=0.75,
        fy=0.75,
        interpolation=openCV.INTER_AREA,
    )


# Skew an image using three corresponding points.
def applyAffineTransform(image):
    height, width = image.shape[:2]

    # Three original points form a triangle.
    sourcePoints = np.float32([
        [0, 0],
        [width - 1, 0],
        [0, height - 1],
    ])

    # Where those three points should move.
    destinationPoints = np.float32([
        [width * 0.08, height * 0.08],
        [width * 0.92, height * 0.03],
        [width * 0.03, height * 0.92],
    ])

    affineMatrix = openCV.getAffineTransform(
        sourcePoints,
        destinationPoints,
    )

    return openCV.warpAffine(
        image,
        affineMatrix,
        (width, height),
    )


# Distort an image using four corresponding points.
def applyPerspectiveTransform(image):
    height, width = image.shape[:2]

    # Order: top-left, top-right, bottom-right, bottom-left.
    sourcePoints = np.float32([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ])

    destinationPoints = np.float32([
        [width * 0.13, height * 0.12],
        [width * 0.82, height * 0.04],
        [width * 0.93, height * 0.90],
        [width * 0.06, height * 0.95],
    ])

    perspectiveMatrix = openCV.getPerspectiveTransform(
        sourcePoints,
        destinationPoints,
    )

    perspectiveImage = openCV.warpPerspective(
        image,
        perspectiveMatrix,
        (width, height),
    )

    return perspectiveImage, sourcePoints, destinationPoints


# Correct the perspective distortion we just created.
def correctPerspective(perspectiveImage, sourcePoints, destinationPoints):
    height, width = perspectiveImage.shape[:2]

    # Distorted corners become the source.
    # The original rectangle becomes the destination.
    correctionMatrix = openCV.getPerspectiveTransform(
        destinationPoints,
        sourcePoints,
    )

    return openCV.warpPerspective(
        perspectiveImage,
        correctionMatrix,
        (width, height),
    )


def saveImage(fileName, image):
    savePath = assetsPath / fileName

    if not openCV.imwrite(str(savePath), image):
        raise HTTPException(
            status_code=500,
            detail=f"Cannot save image: {fileName}"
        )

    return fileName


# Run all seven image transformations.
def processImageTransformations():
    try:
        image = loadImage()
        translatedImage = translateImage(image)
        rotatedImage = rotateImage(image)
        flippedImage = flipImage(image)
        scaledImage = scaleImage(image)
        affineImage = applyAffineTransform(image)
        perspectiveImage, sourcePoints, destinationPoints = applyPerspectiveTransform(image)
        correctedImage = correctPerspective(
            perspectiveImage,
            sourcePoints,
            destinationPoints,
        )

        savedImages = {
            "translated": saveImage("6-translated.png", translatedImage),
            "rotated": saveImage("6-rotated.png", rotatedImage),
            "flipped": saveImage("6-flipped.png", flippedImage),
            "scaled": saveImage("6-scaled.png", scaledImage),
            "affine": saveImage("6-affine.png", affineImage),
            "perspective": saveImage("6-perspective.png", perspectiveImage),
            "corrected": saveImage("6-corrected.png", correctedImage),
        }

        return {
            "lesson": 13,
            "totalImages": len(savedImages),
            "savedImages": savedImages,
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Unable to transform image"
        ) from error


# Return one saved transformation through FastAPI.
def getTransformedImage(imageType: ImageType = "rotated"):
    imagePath = assetsPath / f"6-{imageType}.png"

    if not imagePath.is_file():
        raise HTTPException(
            status_code=404,
            detail="Process Lesson 13 first"
        )

    return FileResponse(
        path=imagePath,
        media_type="image/png",
    )
