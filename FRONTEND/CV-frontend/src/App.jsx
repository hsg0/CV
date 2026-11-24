import { useEffect, useRef, useState } from 'react'
import './App.css'

const api = 'http://127.0.0.1:4020'

const modeGroups = [
  {
    title: '1. Motion & Object Tracking (Lessons 2, 17 & 18)',
    description: 'Detect moving regions, assign persistent tracking IDs, draw trails, or combine every lesson at once.',
    modes: [
      ['tracking', 'Object Tracking (L18)'],
      ['mask', 'Motion Mask (L17)'],
      ['all', 'All Combined (L18)'],
      ['original', 'Original Camera (L2)'],
    ],
  },
  {
    title: '2. Basic Filters, Channels & Edges (Lessons 3, 4, 7 & 8)',
    description: 'Convert to grayscale, smooth noise, boost brightness, split BGR channels, or find Canny edges.',
    modes: [
      ['gray', 'Grayscale (L3)'],
      ['blur', 'Gaussian Blur (L7)'],
      ['brightness', 'Brightness +35 (L7)'],
      ['blue', 'Blue Channel (L4)'],
      ['green', 'Green Channel (L4)'],
      ['red', 'Red Channel (L4)'],
      ['canny', 'Canny Edges (L8)'],
    ],
  },
  {
    title: '3. Thresholding & Morphology (Lessons 9 & 10)',
    description: 'Turn brightness into binary masks and clean them with erosion, dilation, opening, and closing.',
    modes: [
      ['binary', 'Binary Otsu (L9)'],
      ['adaptive', 'Adaptive Mask (L9)'],
      ['range', 'Range 80-180 (L9)'],
      ['erode', 'Erode (L10)'],
      ['dilate', 'Dilate (L10)'],
      ['opening', 'Opening (L10)'],
      ['closing', 'Closing (L10)'],
    ],
  },
  {
    title: '4. Contours, HSV Colour & Geometry (Lessons 4, 11, 12 & 13)',
    description: 'Outline contours, isolate HSV colours, or crop, flip, rotate, and warp the live frame.',
    modes: [
      ['contours', 'Contours & Boxes (L11)'],
      ['color', 'HSV Colour (L12)'],
      ['crop', 'Centre Crop (L4)'],
      ['flip', 'Flip Horizontal (L13)'],
      ['rotate', 'Rotate 25° (L13)'],
      ['affine', 'Affine Skew (L13)'],
      ['perspective', 'Perspective Tilt (L13)'],
    ],
  },
  {
    title: '5. ORB Feature Detection & Matching (Lesson 14)',
    description: 'Detect 700 ORB keypoints live or locate your saved reference object using RANSAC homography.',
    modes: [
      ['features', 'ORB Keypoints (L14)'],
      ['match', 'ORB Match Reference (L14)'],
    ],
  },
]

const motionMethods = [
  ['background', 'MOG2 Background (L17)'],
  ['difference', 'Frame Difference (L17)'],
]

const colors = ['green', 'red', 'blue', 'yellow']

function App() {
  const [view, setView] = useState('tracking')
  const [method, setMethod] = useState('background')
  const [colorName, setColorName] = useState('green')
  const [streaming, setStreaming] = useState(false)
  const [streamKey, setStreamKey] = useState(0)
  const [frameUrl, setFrameUrl] = useState('')
  const [selection, setSelection] = useState(null)
  const [message, setMessage] = useState('')
  const canvasRef = useRef(null)
  const imageRef = useRef(null)
  const dragRef = useRef(null)

  const streamUrl = `${api}/lesson18/stream?view=${view}&method=${method}&colorName=${colorName}&session=${streamKey}`

  useEffect(() => {
    if (!frameUrl) {
      return undefined
    }

    const image = new Image()
    image.onload = () => {
      imageRef.current = image
      const canvas = canvasRef.current
      if (!canvas) {
        return
      }
      canvas.width = image.width
      canvas.height = image.height
      canvas.getContext('2d').drawImage(image, 0, 0)
    }
    image.src = frameUrl

    return () => {
      image.onload = null
    }
  }, [frameUrl])

  async function syncSettings(nextView, nextMethod, nextColorName) {
    if (!streaming) {
      return
    }
    const query = new URLSearchParams({
      view: nextView,
      method: nextMethod,
      colorName: nextColorName,
    })
    await fetch(`${api}/lesson18/settings?${query}`)
  }

  async function chooseView(nextView) {
    setView(nextView)
    await syncSettings(nextView, method, colorName)
  }

  async function chooseMethod(nextMethod) {
    setMethod(nextMethod)
    await syncSettings(view, nextMethod, colorName)
  }

  async function chooseColor(nextColorName) {
    setColorName(nextColorName)
    await syncSettings(view, method, nextColorName)
  }

  async function startCamera(initialView = view) {
    await fetch(`${api}/lesson18/stop`)
    setView(initialView)
    setStreamKey((previousKey) => previousKey + 1)
    setStreaming(true)
  }

  async function stopCamera() {
    setStreaming(false)
    await fetch(`${api}/lesson18/stop`)
  }

  function drawSelection(box) {
    const image = imageRef.current
    const canvas = canvasRef.current
    if (!image || !canvas) {
      return
    }

    const context = canvas.getContext('2d')
    context.drawImage(image, 0, 0)

    if (!box) {
      return
    }

    context.strokeStyle = '#39ff14'
    context.lineWidth = 2
    context.strokeRect(box.x, box.y, box.width, box.height)
  }

  function pointerPosition(event) {
    const canvas = canvasRef.current
    const bounds = canvas.getBoundingClientRect()
    return {
      x: (event.clientX - bounds.left) * (canvas.width / bounds.width),
      y: (event.clientY - bounds.top) * (canvas.height / bounds.height),
    }
  }

  function startDrag(event) {
    const point = pointerPosition(event)
    dragRef.current = point
    setSelection(null)
  }

  function moveDrag(event) {
    if (!dragRef.current) {
      return
    }

    const point = pointerPosition(event)
    const start = dragRef.current
    const box = {
      x: Math.min(start.x, point.x),
      y: Math.min(start.y, point.y),
      width: Math.abs(point.x - start.x),
      height: Math.abs(point.y - start.y),
    }
    setSelection(box)
    drawSelection(box)
  }

  function endDrag() {
    dragRef.current = null
  }

  async function captureFrame() {
    setStreaming(false)
    setMessage('Stopping stream and capturing one frame from Webcam 0...')

    const response = await fetch(`${api}/lesson18/frame`)
    if (!response.ok) {
      setMessage('Could not capture a frame from Webcam 0.')
      return
    }

    const blob = await response.blob()
    setFrameUrl(URL.createObjectURL(blob))
    setSelection(null)
    setMessage('Drag a green box around an object on the frame below, then click Save Reference.')
  }

  async function saveSelection(andStartMatch = false) {
    if (!selection || selection.width < 8 || selection.height < 8) {
      setMessage('Drag a box around the object first.')
      return
    }

    const query = new URLSearchParams({
      x: Math.round(selection.x),
      y: Math.round(selection.y),
      width: Math.round(selection.width),
      height: Math.round(selection.height),
    })
    const response = await fetch(`${api}/lesson18/reference?${query}`)
    if (!response.ok) {
      setMessage('Could not save the reference image.')
      return
    }

    if (andStartMatch) {
      setMessage('Reference saved. Starting camera in ORB Match mode...')
      await startCamera('match')
      return
    }

    setMessage('Saved lesson18-reference.png! Click "ORB Match Reference (L14)" and Start Camera.')
  }

  return (
    <main className="dashboard">
      <header className="header">
        <div>
          <h1>Python OpenCV — All-in-One Lab (Lessons 1–18)</h1>
          <p>
            One live Webcam 0 stream connected to FastAPI. Switch any mode, motion detector, or HSV colour live without restarting the camera.
          </p>
        </div>
        <div className="status-badges">
          <span className={streaming ? 'badge live' : 'badge idle'}>
            {streaming ? 'Camera Live' : 'Camera Off'}
          </span>
          <span className="badge">Mode: {view}</span>
          <span className="badge">Motion: {method}</span>
          <span className="badge">Colour: {colorName}</span>
        </div>
      </header>

      <div className="workspace">
        <section className="stage-card">
          <div className="toolbar">
            <div className="toolbar-group">
              <button
                type="button"
                className="primary-btn"
                onClick={() => startCamera(view)}
              >
                Start Camera
              </button>
              <button
                type="button"
                className="stop-btn"
                onClick={stopCamera}
              >
                Stop Camera
              </button>
            </div>

            <div className="toolbar-group">
              <label>
                Motion Method (L17)
                <select
                  value={method}
                  onChange={(event) => chooseMethod(event.target.value)}
                >
                  {motionMethods.map(([value, label]) => (
                    <option key={value} value={value}>{label}</option>
                  ))}
                </select>
              </label>

              <label>
                HSV Colour (L12)
                <select
                  value={colorName}
                  onChange={(event) => chooseColor(event.target.value)}
                >
                  {colors.map((color) => (
                    <option key={color} value={color}>{color}</option>
                  ))}
                </select>
              </label>
            </div>
          </div>

          <div className="video-frame">
            {streaming ? (
              <img
                key={streamKey}
                className="camera"
                src={streamUrl}
                alt="Lesson 18 live OpenCV stream"
              />
            ) : (
              <div className="camera off">
                <p>Webcam 0 is off.</p>
                <p className="hint">Pick any lesson mode on the right, then click Start Camera.</p>
              </div>
            )}
          </div>

          <div className="reference-card">
            <h2>6. Interactive Reference Selector (Lessons 5, 6 & 14)</h2>
            <p>
              Capture a single webcam frame, drag a box with your mouse to crop a reference object, and save <code>lesson18-reference.png</code> for live ORB matching.
            </p>
            <div className="toolbar-group">
              <button type="button" onClick={captureFrame}>
                1. Capture Frame
              </button>
              <button type="button" onClick={() => saveSelection(false)}>
                2. Save Reference
              </button>
              <button
                type="button"
                className="primary-btn"
                onClick={() => saveSelection(true)}
              >
                3. Save &amp; Start Match
              </button>
            </div>

            {selection ? (
              <p className="coords">
                Selected box: x={Math.round(selection.x)}, y={Math.round(selection.y)}, width={Math.round(selection.width)}, height={Math.round(selection.height)}
              </p>
            ) : null}

            {message ? <p className="message">{message}</p> : null}

            {frameUrl ? (
              <canvas
                ref={canvasRef}
                className="camera selector-canvas"
                onPointerDown={startDrag}
                onPointerMove={moveDrag}
                onPointerUp={endDrag}
                onPointerLeave={endDrag}
              />
            ) : null}
          </div>
        </section>

        <aside className="controls-column">
          {modeGroups.map((group) => (
            <section key={group.title} className="mode-card">
              <h2>{group.title}</h2>
              <p>{group.description}</p>
              <div className="modes">
                {group.modes.map(([value, label]) => (
                  <button
                    key={value}
                    type="button"
                    className={view === value ? 'selected' : ''}
                    onClick={() => chooseView(value)}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </section>
          ))}
        </aside>
      </div>
    </main>
  )
}

export default App
