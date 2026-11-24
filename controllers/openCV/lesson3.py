import os

from fastapi import HTTPException, Response
from config.open_cv.configureCV import assetsDirectory, configureCV

openCV = configureCV()


def webcamImageToGrayScale():
    imagePath = os.path.join(assetsDirectory, "webcam1.jpg")
    savePath = os.path.join(assetsDirectory, "webcam1-gray.jpg")

    try:
        if not os.path.exists(imagePath):
            raise HTTPException(
                status_code=404,
                detail="Image not found"
            )

        image = openCV.imread(imagePath)

        if image is None:
            raise HTTPException(
                status_code=500,
                detail="Unable to read image"
            )

        grayImage = openCV.cvtColor(image, openCV.COLOR_BGR2GRAY)
        saved = openCV.imwrite(savePath, grayImage)

        if not saved:
            raise HTTPException(
                status_code=500,
                detail="Unable to save image"
            )

        encoded, grayBytes = openCV.imencode(".jpg", grayImage)

        if not encoded:
            raise HTTPException(
                status_code=500,
                detail="Unable to encode image"
            )

        return Response(
            content=grayBytes.tobytes(),
            media_type="image/jpeg"
        )

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Unable to convert image to grayscale"
        ) from error
