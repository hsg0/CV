from pathlib import Path

from fastapi import HTTPException
from config.open_cv.configureCV import assetsDirectory, configureCV

openCV = configureCV()
assetsPath = Path(assetsDirectory)


def bringInImage():
    imagePath = assetsPath / "6.png"

    try:
        if not imagePath.exists():
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

        resizedImage = openCV.resize(image, (1280, 720))

        # Rows first, then columns: [y1:y2, x1:x2]
        croppedHeadOfDog = resizedImage[200:450, 520:800]

        blueChannel, greenChannel, redChannel = openCV.split(croppedHeadOfDog)

        imagesToSave = {
            "6-cropped.png": croppedHeadOfDog,
            "6-blue.png": blueChannel,
            "6-green.png": greenChannel,
            "6-red.png": redChannel,
        }

        savedFiles = []

        for fileName, processedImage in imagesToSave.items():
            filePath = assetsPath / fileName

            if not openCV.imwrite(str(filePath), processedImage):
                raise HTTPException(
                    status_code=500,
                    detail=f"Unable to save {fileName}"
                )

            savedFiles.append(fileName)

        return {
            "message": "Image processing complete",
            "files": savedFiles,
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Unable to process image"
        ) from error
