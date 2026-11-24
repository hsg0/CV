from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse
from config.open_cv.configureCV import assetsDirectory, configureCV

openCV = configureCV()
assetsPath = Path(assetsDirectory)


def drawDogBox(image, leftX, topY, rightX, bottomY):
    annotatedImage = image.copy()

    openCV.rectangle(
        annotatedImage,
        (leftX, topY),
        (rightX, bottomY),
        (0, 255, 0),
        2,
    )
    openCV.putText(
        annotatedImage,
        "DOGGY",
        (leftX, max(topY - 10, 25)),
        openCV.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    openCV.circle(
        annotatedImage,
        (leftX, topY),
        5,
        (0, 0, 255),
        -1,
    )
    return annotatedImage


def drawARectangleAroundTheObject():
    imagePath = assetsPath / "6.png"
    savePath = assetsPath / "6-annotated.png"

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

        try:
            leftX, topY, width, height = openCV.selectROI(
                "Select Rectangle",
                image,
                showCrosshair=False,
            )
        finally:
            openCV.destroyAllWindows()

        if width == 0 or height == 0:
            return {"message": "Window closed without saving"}

        annotatedImage = drawDogBox(
            image,
            leftX,
            topY,
            leftX + width,
            topY + height,
        )

        if not openCV.imwrite(str(savePath), annotatedImage):
            raise HTTPException(
                status_code=500,
                detail="Unable to save image"
            )

        return {
            "message": "Image saved",
            "file": savePath.name,
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Unable to draw rectangle"
        ) from error


def getAnnotatedImage():
    imagePath = assetsPath / "6-annotated.png"

    if not imagePath.is_file():
        raise HTTPException(
            status_code=404,
            detail="Annotated image not found. Open /lesson6/draw-rectangle, draw a box, then press Space or Enter."
        )

    return FileResponse(
        path=imagePath,
        media_type="image/png"
    )
