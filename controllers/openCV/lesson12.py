from pathlib import Path
from typing import Literal

from fastapi import HTTPException
from fastapi.responses import FileResponse
from config.open_cv.configureCV import assetsDirectory, configureCV

# Load our configured OpenCV library.
openCV = configureCV()

# Location of our original and processed images.
assetsPath = Path(assetsDirectory)

# Restrict accepted colours to these four choices.
ColorName = Literal["red", "green", "blue", "yellow"]

# Restrict the image types we can return.
ImageType = Literal["mask", "filtered", "detected"]

# Draw bounding boxes in neon green.
# OpenCV uses BGR colour order, not RGB.
neonGreen = (20, 255, 57)

# Ignore detected regions smaller than 300 square pixels.
minimumArea = 300

# Define approximate HSV ranges for each colour.
# Format: (minimum H, S, V), (maximum H, S, V).
# Red needs two ranges because hue wraps around at 179.
colorRanges = {
    "red": [
        ((0, 100, 100), (10, 255, 255)),
        ((170, 100, 100), (179, 255, 255)),
    ],
    "green": [
        ((36, 80, 70), (85, 255, 255)),
    ],
    "blue": [
        ((90, 80, 70), (130, 255, 255)),
    ],
    "yellow": [
        ((20, 100, 100), (35, 255, 255)),
    ],
}


# Step 1. Original image.
# Read the colour photo we start from.
def loadImage():
    imagePath = assetsPath / "6.png"
    image = openCV.imread(str(imagePath))

    if image is None:
        raise HTTPException(
            status_code=404,
            detail="Cannot read image: 6.png"
        )

    return image


# Step 2. HSV conversion.
# HSV separates hue, saturation, and brightness.
# Hue is the colour. That makes a colour range easier to select than BGR.
def convertToHSV(image):
    return openCV.cvtColor(image, openCV.COLOR_BGR2HSV)


# Step 3. Colour mask.
# White pixels match the chosen colour. Black pixels do not.
# Red uses two ranges because hue restarts at 0 after 179.
def createColorMask(hsvImage, colorName):
    selectedRanges = colorRanges[colorName]
    lowerRange, upperRange = selectedRanges[0]
    mask = openCV.inRange(hsvImage, lowerRange, upperRange)

    for lowerRange, upperRange in selectedRanges[1:]:
        additionalMask = openCV.inRange(hsvImage, lowerRange, upperRange)
        mask = openCV.bitwise_or(mask, additionalMask)

    return mask


# Step 4. Morphological cleanup.
# Opening removes small white specks. Closing fills small holes in the white regions.
def cleanColorMask(mask):
    kernel = openCV.getStructuringElement(openCV.MORPH_RECT, (5, 5))
    openedMask = openCV.morphologyEx(mask, openCV.MORPH_OPEN, kernel)
    cleanedMask = openCV.morphologyEx(openedMask, openCV.MORPH_CLOSE, kernel)
    return cleanedMask


# Step 5. Extract colour.
# Keep the original pixels where the mask is white. Everything else becomes black.
def extractColor(image, mask):
    return openCV.bitwise_and(image, image, mask=mask)


# Step 6. Find contours.
# A contour is the boundary of a white region. It does not know the object's name.
def findColorContours(mask):
    contours, hierarchy = openCV.findContours(
        mask.copy(),
        openCV.RETR_EXTERNAL,
        openCV.CHAIN_APPROX_SIMPLE,
    )
    return contours


# Step 7. Draw bounding boxes.
# A bounding box gives each accepted region's location and size.
def drawColorBoundingBoxes(image, contours):
    resultImage = image.copy()
    detectedObjects = []

    for contour in contours:
        area = openCV.contourArea(contour)

        if area < minimumArea:
            continue

        leftX, topY, width, height = openCV.boundingRect(contour)
        rightX = leftX + width
        bottomY = topY + height
        objectNumber = len(detectedObjects) + 1
        label = f"Object {objectNumber}"
        textY = max(25, topY - 10)

        openCV.drawContours(resultImage, [contour], -1, neonGreen, 2)
        openCV.rectangle(
            resultImage,
            (leftX, topY),
            (rightX, bottomY),
            neonGreen,
            2,
        )
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
            "boundingBox": {
                "leftX": leftX,
                "topY": topY,
                "width": width,
                "height": height,
            },
        })

    return resultImage, detectedObjects


# Step 8. Save results.
def saveImage(fileName, image):
    savePath = assetsPath / fileName

    if not openCV.imwrite(str(savePath), image):
        raise HTTPException(
            status_code=500,
            detail=f"Cannot save image: {fileName}"
        )

    return fileName


# Original image → HSV conversion → Colour mask → Morphological cleanup
# → Extract colour → Find contours → Draw bounding boxes → Save results
def processColorDetection(colorName: ColorName = "green"):
    try:
        # Step 1. Original image.
        image = loadImage()

        # Step 2. HSV conversion.
        hsvImage = convertToHSV(image)

        # Step 3. Colour mask.
        mask = createColorMask(hsvImage, colorName)

        # Step 4. Morphological cleanup.
        mask = cleanColorMask(mask)

        # Step 5. Extract colour.
        filteredImage = extractColor(image, mask)

        # Step 6. Find contours.
        contours = findColorContours(mask)

        # Step 7. Draw bounding boxes.
        detectedImage, detectedObjects = drawColorBoundingBoxes(image, contours)

        # Step 8. Save results.
        savedImages = {
            "mask": saveImage(f"6-{colorName}-mask.png", mask),
            "filtered": saveImage(f"6-{colorName}-filtered.png", filteredImage),
            "detected": saveImage(f"6-{colorName}-detected.png", detectedImage),
        }

        return {
            "selectedColor": colorName,
            "totalContours": len(contours),
            "objectsDetected": len(detectedObjects),
            "objects": detectedObjects,
            "savedImages": savedImages,
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Unable to detect colour"
        ) from error


def getColorImage(
    colorName: ColorName = "green",
    imageType: ImageType = "detected",
):
    imagePath = assetsPath / f"6-{colorName}-{imageType}.png"

    if not imagePath.is_file():
        raise HTTPException(
            status_code=404,
            detail="Process this colour first"
        )

    return FileResponse(
        path=imagePath,
        media_type="image/png"
    )
