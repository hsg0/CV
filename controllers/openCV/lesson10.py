
from pathlib import Path

from config.open_cv.configureCV import assetsDirectory, configureCV

openCV = configureCV()
assetsPath = Path(assetsDirectory)


def createMorphologyKernel():
    return openCV.getStructuringElement(
        openCV.MORPH_RECT,
        (3, 3)
    )


def erodeMask():
    imagePath = assetsPath / "6-binary.png"
    savePath = assetsPath / "6-eroded.png"

    image = openCV.imread(
        str(imagePath),
        openCV.IMREAD_GRAYSCALE
    )

    if image is None:
        raise FileNotFoundError(f"Cannot read image: {imagePath}")

    kernel = createMorphologyKernel()

    erodedImage = openCV.erode(
        image,
        kernel,
        iterations=1
    )

    saved = openCV.imwrite(str(savePath), erodedImage)

    if not saved:
        raise RuntimeError(f"Cannot save image: {savePath}")

    return str(savePath)
