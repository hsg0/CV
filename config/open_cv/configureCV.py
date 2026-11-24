
import os
import cv2 as openCV

cameraIndex = 0
assetsDirectory = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "assets",
)


def configureCV():
    cv_version = os.getenv("CV_VERSION")

    if cv_version:
        if openCV.__version__ != cv_version:
            raise RuntimeError(
                f"Expected OpenCV {cv_version}, "
                f"but installed {openCV.__version__}"
            )

    return openCV



