from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse
from config.open_cv.configureCV import assetsDirectory, configureCV

openCV = configureCV()
assetsPath = Path(assetsDirectory)


def labelImage(image, text):
    labeledImage = image.copy()
    openCV.putText(
        labeledImage,
        text,
        (20, 40),
        openCV.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    return labeledImage


def applyGaussianBlurToAnImage():
    imagePath = assetsPath / "6.png"

    try:
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

        images = {
            "6-gaussian-blurred.png": openCV.GaussianBlur(image, (7, 7), 0),
            "6-regular-blurred.png": openCV.blur(image, (7, 7)),
            "6-median-blurred.png": openCV.medianBlur(image, 7),
            "6-bilateral-blurred.png": openCV.bilateralFilter(image, 9, 75, 75),
            "6-white-scale.png": openCV.addWeighted(image, 1.0, image, 0, 50),
            "6-black-scale.png": openCV.addWeighted(image, 1.0, image, 0, -50),
        }

        labeledImages = []

        for fileName, processedImage in images.items():
            filePath = assetsPath / fileName

            if not openCV.imwrite(str(filePath), processedImage):
                raise HTTPException(
                    status_code=500,
                    detail=f"Unable to save {fileName}"
                )

            label = fileName.removeprefix("6-").removesuffix(".png")
            labeledImages.append(labelImage(processedImage, label))

        topRow = openCV.hconcat(labeledImages[:3])
        bottomRow = openCV.hconcat(labeledImages[3:])
        comparisonImage = openCV.vconcat([topRow, bottomRow])
        comparisonPath = assetsPath / "6-blurs.png"

        if not openCV.imwrite(str(comparisonPath), comparisonImage):
            raise HTTPException(
                status_code=500,
                detail="Unable to save 6-blurs.png"
            )

        return FileResponse(
            path=comparisonPath,
            media_type="image/png"
        )

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Unable to blur image"
        ) from error
