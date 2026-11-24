from pathlib import Path

from config.open_cv.configureCV import assetsDirectory, configureCV

openCV = configureCV()
assetsPath = Path(assetsDirectory)


def useMouseToCreateABoxAroundObject():
    imagePath = assetsPath / "6.png"
    image = openCV.imread(str(imagePath))

    if image is None:
        raise FileNotFoundError("Unable to read image")

    return image


def selectDogHead():
    image = useMouseToCreateABoxAroundObject()
    windowName = "Select Dog Head"

    try:
        leftX, topY, width, height = openCV.selectROI(
            windowName,
            image,
            showCrosshair=False,
        )
    finally:
        openCV.destroyAllWindows()

    if width == 0 or height == 0:
        return None

    croppedImage = image[topY:topY + height, leftX:leftX + width]
    savePath = assetsPath / "6-selected.png"

    if not openCV.imwrite(str(savePath), croppedImage):
        raise OSError("Unable to save cropped image")

    print(f"Image saved: {savePath}")
    return savePath.name



