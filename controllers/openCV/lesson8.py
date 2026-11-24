from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse
from config.open_cv.configureCV import assetsDirectory, configureCV

openCV = configureCV()
assetsPath = Path(assetsDirectory)
neonGreen = (20, 255, 57)


def applyCannyEdgeDetection():
    imagePath = assetsPath / "6.png"
    savePath = assetsPath / "6-canny.png"

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

        grayImage = openCV.cvtColor(image, openCV.COLOR_BGR2GRAY)
        edges = openCV.Canny(grayImage, 50, 150)

        coloredImage = image.copy()
        coloredImage[edges > 0] = neonGreen

        if not openCV.imwrite(str(savePath), coloredImage):
            raise HTTPException(
                status_code=500,
                detail="Unable to save image"
            )

        return FileResponse(
            path=savePath,
            media_type="image/png"
        )

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Unable to detect edges"
        ) from error
