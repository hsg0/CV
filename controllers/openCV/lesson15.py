
# FILE: lesson15.py
# LOCATION: python-CV/controllers/openCV/lesson15.py
# WHY: Learn to capture, process, and stream live camera video.
# WHAT: Streams webcam frames with a live FPS counter.
# HOW: Uses OpenCV, JPEG encoding, and FastAPI StreamingResponse.
# WHEN: Runs when a client opens the Lesson 15 video endpoint.
# WHERE: Connects a local camera to the FastAPI backend and browser.

from threading import Lock
from time import perf_counter

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from config.open_cv.configureCV import configureCV


# Load our configured OpenCV library.
openCV = configureCV()

# Use camera number 0, normally the default webcam.
# Change this to 1 if your webcam uses a different index.
cameraIndex = 0

# OpenCV uses BGR colour order.
neonGreen = (20, 255, 57)

# Use 80% JPEG quality to balance quality and bandwidth.
jpegQuality = 80

# Allow one camera stream at a time in this learning project.
# This prevents two requests from opening the same webcam.
cameraLock = Lock()


# Open the camera and verify it is available.
def openCamera():

    # Ask OpenCV to open the selected webcam.
    camera = openCV.VideoCapture(cameraIndex)

    # Check whether the camera opened successfully.
    if not camera.isOpened():

        # Release any resources before reporting an error.
        camera.release()

        raise HTTPException(
            status_code=503,
            detail="Unable to open camera"
        )

    return camera


# Calculate how many frames are captured per second.
def calculateFPS(previousTime):

    # Get a high-precision timestamp.
    currentTime = perf_counter()

    # Calculate elapsed time between frames.
    elapsedTime = currentTime - previousTime

    # Avoid division by zero.
    if elapsedTime <= 0:
        return 0.0, currentTime

    # FPS = 1 divided by elapsed seconds.
    fps = 1.0 / elapsedTime

    return fps, currentTime


# Draw the FPS counter on the current video frame.
def drawFPS(frame, fps):

    # Copy the frame so the input is not modified.
    processedFrame = frame.copy()

    # Build the text shown on the video.
    fpsText = f"FPS: {fps:.1f}"

    # Draw a dark outline behind the text.
    openCV.putText(
        processedFrame,
        fpsText,
        (20, 40),
        openCV.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        4
    )

    # Draw neon-green FPS text over the dark outline.
    openCV.putText(
        processedFrame,
        fpsText,
        (20, 40),
        openCV.FONT_HERSHEY_SIMPLEX,
        1,
        neonGreen,
        2
    )

    return processedFrame


# Convert a video frame into JPEG bytes.
def encodeFrame(frame):

    # Configure the JPEG compression quality.
    jpegSettings = [
        openCV.IMWRITE_JPEG_QUALITY,
        jpegQuality
    ]

    # Encode the image in memory.
    # No image file is saved to disk.
    success, encodedImage = openCV.imencode(
        ".jpg",
        frame,
        jpegSettings
    )

    # Return None if encoding fails.
    if not success:
        return None

    # Convert the encoded image into bytes for HTTP.
    return encodedImage.tobytes()


# Capture and process frames continuously.
def generateVideoFrames(camera):

    # Record the time before capturing our first frame.
    previousTime = perf_counter()

    try:

        # Keep processing frames while the stream is active.
        while True:

            # Capture the next camera frame.
            success, frame = camera.read()

            # Stop if the camera stops returning frames.
            if not success:
                break

            # Calculate FPS and update our timestamp.
            fps, previousTime = calculateFPS(previousTime)

            # Draw FPS directly onto the captured frame.
            processedFrame = drawFPS(frame, fps)

            # Convert the processed frame into JPEG bytes.
            imageBytes = encodeFrame(processedFrame)

            # Skip a frame if encoding was unsuccessful.
            if imageBytes is None:
                continue

            # Yield sends one JPEG frame to the browser.
            # The boundary separates consecutive frames.
            # CRLF separates HTTP multipart headers.
            # The browser displays each frame as it arrives.
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"\r\n"
                + imageBytes
                + b"\r\n"
            )

    finally:

        # Release the camera when frame generation ends.
        camera.release()

        # Allow another client to use the webcam.
        cameraLock.release()


# Main controller: provide a live video streaming response.
def streamVideo():

    # Prevent multiple requests from using the webcam.
    cameraAvailable = cameraLock.acquire(
        blocking=False
    )

    if not cameraAvailable:
        raise HTTPException(
            status_code=409,
            detail="Camera is already in use"
        )

    try:

        # Open our webcam.
        camera = openCamera()

        # StreamingResponse consumes frames as they are
        # generated instead of waiting for a complete video.
        return StreamingResponse(
            generateVideoFrames(camera),

            # MJPEG is a sequence of JPEG images.
            # Each JPEG is separated by boundary=frame.
            media_type=(
                "multipart/x-mixed-replace; boundary=frame"
            ),

            # Ask browsers not to cache the live stream.
            headers={
                "Cache-Control": "no-store"
            }
        )

    except Exception:

        # If opening the camera fails, release the lock.
        cameraLock.release()

        # Preserve the original exception for debugging.
        raise
