# FILE: lesson17.py
# LOCATION: python-CV/routes/opencv/lesson17.py
# WHY: Access Lesson 17 through our FastAPI backend.
# WHAT: Streams movement detection from Webcam 0.
# HOW: Connects the video URL to the Lesson 17 controller.
# WHEN: FastAPI receives a Lesson 17 request.
# WHERE: Imported and registered by main.py.

from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from controllers.openCV.lesson17 import streamMotionVideo, updateMotionSettings

lesson17Router = APIRouter()


# One page for both motion methods and both views.
def watchMotion():
    return HTMLResponse(
        """
        <!doctype html>
        <html>
        <head>
          <title>Lesson 17</title>
          <style>
            body { font-family: sans-serif; max-width: 720px; margin: 24px auto; }
            button { margin: 4px; padding: 8px 12px; }
            button.selected { outline: 2px solid #39ff14; }
            img, .off { width: 100%; max-width: 640px; background: #111; }
            .off { color: white; padding: 48px 16px; text-align: center; }
          </style>
        </head>
        <body>
          <h1>Lesson 17</h1>
          <p>One camera. Switch the method or the view on the same stream.</p>
          <h2>Method</h2>
          <button id="difference" type="button">Frame difference</button>
          <button id="background" type="button">MOG2 background</button>
          <h2>View</h2>
          <button id="boxes" type="button">Boxes</button>
          <button id="mask" type="button">Mask</button>
          <p>
            <button id="start" type="button">Start</button>
            <button id="stop" type="button">Stop</button>
          </p>
          <img id="video" alt="Motion" hidden>
          <p id="status" class="off">Camera is off.</p>
          <script>
            let method = "difference";
            let view = "boxes";
            let streaming = false;
            const video = document.getElementById("video");
            const status = document.getElementById("status");

            function markSelected() {
              for (const name of ["difference", "background", "boxes", "mask"]) {
                document.getElementById(name).className =
                  name === method || name === view ? "selected" : "";
              }
            }

            async function choose(nextMethod, nextView) {
              method = nextMethod;
              view = nextView;
              markSelected();
              await fetch("/lesson17/settings?method=" + method + "&view=" + view);
            }

            document.getElementById("difference").onclick = () => choose("difference", view);
            document.getElementById("background").onclick = () => choose("background", view);
            document.getElementById("boxes").onclick = () => choose(method, "boxes");
            document.getElementById("mask").onclick = () => choose(method, "mask");
            document.getElementById("start").onclick = () => {
              if (streaming) return;
              streaming = true;
              status.hidden = true;
              video.hidden = false;
              video.src = "/lesson17/stream?method=" + method + "&view=" + view;
            };
            document.getElementById("stop").onclick = () => {
              streaming = false;
              video.removeAttribute("src");
              video.hidden = true;
              status.hidden = false;
            };
            markSelected();
          </script>
        </body>
        </html>
        """
    )


lesson17Router.get("/lesson17")(watchMotion)
lesson17Router.get("/lesson17/settings")(updateMotionSettings)
lesson17Router.get("/lesson17/stream")(streamMotionVideo)
