from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse
from config.open_cv.configureCV import assetsDirectory, configureCV

openCV = configureCV()
assetsPath = Path(assetsDirectory)
neonGreen = (20, 255, 57)
orange = (0, 165, 255)
minimumArea = 300


def loadBinaryMask():
    imagePath = assetsPath / "6-binary.png"
    mask = openCV.imread(str(imagePath), openCV.IMREAD_GRAYSCALE)

    if mask is None:
        raise HTTPException(
            status_code=404,
            detail="Cannot read image: 6-binary.png"
        )

    return mask


def loadOriginalImage(mask):
    imagePath = assetsPath / "6.png"
    image = openCV.imread(str(imagePath))

    if image is None:
        raise HTTPException(
            status_code=404,
            detail="Cannot read image: 6.png"
        )

    maskHeight, maskWidth = mask.shape[:2]
    return openCV.resize(image, (maskWidth, maskHeight))


# --- Contours ---
# Contours represent boundaries rather than recognizing an object's meaning.
def findObjectContours(mask):
    contours, hierarchy = openCV.findContours(
        mask.copy(),
        openCV.RETR_EXTERNAL,
        openCV.CHAIN_APPROX_SIMPLE,
    )
    return contours


def drawObjectContours(image, contours):
    resultImage = image.copy()
    detectedObjects = []

    for contour in contours:
        # --- Area ---
        # Area measures the region enclosed by a contour, in square pixels.
        area = openCV.contourArea(contour)

        # --- Perimeter ---
        # Perimeter measures boundary length, in pixels.
        perimeter = openCV.arcLength(contour, True)

        # --- Bounding box ---
        # A bounding box provides an object's location and dimensions.
        leftX, topY, width, height = openCV.boundingRect(contour)

        # --- Polygon approximation ---
        # Polygon approximation simplifies contours into representative points,
        # which helps with shape analysis.
        epsilon = 0.02 * perimeter
        approximateShape = openCV.approxPolyDP(contour, epsilon, True)
        cornerCount = len(approximateShape)

        # --- Filtering ---
        # Filtering prevents tiny contours from cluttering our results.
        if area < minimumArea:
            continue
        rightX = leftX + width
        bottomY = topY + height
        objectNumber = len(detectedObjects) + 1

        openCV.drawContours(resultImage, [contour], -1, neonGreen, 2)
        openCV.rectangle(
            resultImage,
            (leftX, topY),
            (rightX, bottomY),
            orange,
            2,
        )

        textY = max(25, topY - 10)
        label = f"Object {objectNumber}"

        openCV.putText(
            resultImage,
            label,
            (leftX, textY),
            openCV.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            3,
        )
        openCV.putText(
            resultImage,
            label,
            (leftX, textY),
            openCV.FONT_HERSHEY_SIMPLEX,
            0.6,
            neonGreen,
            2,
        )

        detectedObjects.append({
            "objectNumber": objectNumber,
            "area": round(area, 2),
            "perimeter": round(perimeter, 2),
            "cornerCount": cornerCount,
            "boundingBox": {
                "leftX": leftX,
                "topY": topY,
                "width": width,
                "height": height,
            },
        })

    return resultImage, detectedObjects


def processContours():
    try:
        mask = loadBinaryMask()
        image = loadOriginalImage(mask)
        contours = findObjectContours(mask)
        resultImage, detectedObjects = drawObjectContours(image, contours)
        savePath = assetsPath / "6-contours.png"

        if not openCV.imwrite(str(savePath), resultImage):
            raise HTTPException(
                status_code=500,
                detail="Unable to save contour image"
            )

        return {
            "sections": {
                "contours": "Contours represent boundaries rather than recognizing an object's meaning.",
                "area": "Area measures the region enclosed by a contour, in square pixels.",
                "perimeter": "Perimeter measures boundary length, in pixels.",
                "boundingBox": "A bounding box provides an object's location and dimensions.",
                "polygonApproximation": "Polygon approximation simplifies contours into representative points, which helps with shape analysis.",
                "filtering": "Filtering prevents tiny contours from cluttering our results.",
            },
            "totalContours": len(contours),
            "objectsDetected": len(detectedObjects),
            "objects": detectedObjects,
            "savedImage": savePath.name,
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Unable to find contours"
        ) from error


def getContoursImage():
    imagePath = assetsPath / "6-contours.png"

    if not imagePath.is_file():
        raise HTTPException(
            status_code=404,
            detail="Process the image first"
        )

    return FileResponse(
        path=imagePath,
        media_type="image/png"
    )
