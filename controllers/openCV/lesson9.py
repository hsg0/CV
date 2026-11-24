from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse
from config.open_cv.configureCV import assetsDirectory, configureCV

openCV = configureCV()
assetsPath = Path(assetsDirectory)
neonGreen = (20, 255, 57)


def loadImage(fileName):
    imagePath = assetsPath / fileName

    if not imagePath.is_file():
        raise HTTPException(
            status_code=404,
            detail="Image not found"
        )

    image = openCV.imread(str(imagePath))

    if image is None:
        raise HTTPException(
            status_code=500,
            detail="Unable to read image"
        )

    return image


def resizeImage(image, width=1280, height=720):
    return openCV.resize(image, (width, height))


def convertToGrayscale(image):
    return openCV.cvtColor(image, openCV.COLOR_BGR2GRAY)


def applyGaussianBlur(image):
    return openCV.GaussianBlur(image, (5, 5), 0)


def detectEdges(image):
    return openCV.Canny(image, 50, 150)


def createBinaryMask(image):
    thresholdValue, binaryMask = openCV.threshold(
        image,
        127,
        255,
        openCV.THRESH_BINARY,
    )
    return binaryMask


def createAdaptiveMask(image):
    return openCV.adaptiveThreshold(
        image,
        255,
        openCV.ADAPTIVE_THRESH_GAUSSIAN_C,
        openCV.THRESH_BINARY,
        11,
        2,
    )


def createRangeMask(image):
    return openCV.inRange(image, 80, 180)


def colorDetectedEdges(image, edges):
    coloredImage = image.copy()
    coloredImage[edges > 0] = neonGreen
    return coloredImage


def cropImage(image, leftX, topY, rightX, bottomY):
    return image[topY:bottomY, leftX:rightX]


def splitColorChannels(image):
    blue, green, red = openCV.split(image)
    return blue, green, red


def saveImage(fileName, image):
    filePath = assetsPath / fileName

    if not openCV.imwrite(str(filePath), image):
        raise HTTPException(
            status_code=500,
            detail=f"Unable to save {fileName}"
        )

    return fileName


def processImage():
    try:
        image = loadImage("6.png")
        resizedImage = resizeImage(image)
        grayImage = convertToGrayscale(resizedImage)
        blurredImage = applyGaussianBlur(grayImage)
        edges = detectEdges(blurredImage)
        binaryMask = createBinaryMask(blurredImage)
        adaptiveMask = createAdaptiveMask(blurredImage)
        rangeMask = createRangeMask(blurredImage)
        coloredEdges = colorDetectedEdges(resizedImage, edges)
        croppedImage = cropImage(resizedImage, 100, 100, 300, 300)
        blue, green, red = splitColorChannels(resizedImage)

        images = {
            "6-resized.png": resizedImage,
            "6-gray.png": grayImage,
            "6-blurred.png": blurredImage,
            "6-canny.png": edges,
            "6-colored-edges.png": coloredEdges,
            "6-binary.png": binaryMask,
            "6-adaptive.png": adaptiveMask,
            "6-range.png": rangeMask,
            "6-cropped.png": croppedImage,
            "6-blue.png": blue,
            "6-green.png": green,
            "6-red.png": red,
        }

        savedFiles = []

        for fileName, processedImage in images.items():
            savedFiles.append(saveImage(fileName, processedImage))

        return {
            "message": "Image processing completed",
            "files": savedFiles,
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Unable to process image"
        ) from error


def getProcessedImage():
    imagePath = assetsPath / "6-colored-edges.png"

    if not imagePath.is_file():
        raise HTTPException(
            status_code=404,
            detail="Process the image first"
        )

    return FileResponse(
        path=imagePath,
        media_type="image/png"
    )
