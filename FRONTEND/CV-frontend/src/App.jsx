import { useEffect, useRef, useState } from 'react'
import './App.css'

const api = 'http://127.0.0.1:4020'

const processingModes = [
  ['gray', 'Grayscale'],
  ['blur', 'Blur'],
  ['brightness', 'Brighter'],
  ['binary', 'Binary'],
  ['adaptive', 'Adaptive'],
  ['range', 'Range'],
  ['color', 'HSV colour'],
  ['erode', 'Erode'],
  ['dilate', 'Dilate'],
  ['opening', 'Opening'],
  ['closing', 'Closing'],
]

const geometryModes = [
  ['crop', 'Crop'],
  ['flip', 'Flip'],
  ['rotate', 'Rotate'],
  ['affine', 'Affine'],
  ['perspective', 'Perspective'],
  ['contours', 'Contours'],
]

const featureModes = [
  ['features', 'Features'],
  ['match', 'Match'],
]

const colors = ['red', 'green', 'blue', 'yellow']

function ModeButtons({ title, text, modes, mode, onChoose }) {
  return (
    <section>
      <h2>{title}</h2>
      <p>{text}</p>
      <div className="modes">
        {modes.map(([value, label]) => (
          <button
            key={value}
            type="button"
            className={mode === value ? 'selected' : ''}
            onClick={() => onChoose(value)}
          >
            {label}
          </button>
        ))}
      </div>
    </section>
  )
}

function App() {
  const [mode, setMode] = useState('original')
  const [colorName, setColorName] = useState('green')
  const [streaming, setStreaming] = useState(false)
  const [frameUrl, setFrameUrl] = useState('')
  const [selection, setSelection] = useState(null)
  const [message, setMessage] = useState('')
  const canvasRef = useRef(null)
  const imageRef = useRef(null)
  const dragRef = useRef(null)

  const streamUrl = `${api}/lesson16/stream?mode=${mode}&colorName=${colorName}`

  useEffect(() => {
    if (!frameUrl) {
      return undefined
    }

    const image = new Image()
    image.onload = () => {
      imageRef.current = image
      const canvas = canvasRef.current
      canvas.width = image.width
      canvas.height = image.height
      canvas.getContext('2d').drawImage(image, 0, 0)
    }
    image.src = frameUrl

    return () => {
      image.onload = null
    }
  }, [frameUrl])

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
    setMessage('Capturing one frame...')
    await new Promise((resolve) => setTimeout(resolve, 400))

    const response = await fetch(`${api}/lesson16/frame`)
    if (!response.ok) {
      setMessage('Could not capture a frame.')
      return
    }

    const blob = await response.blob()
    setFrameUrl(URL.createObjectURL(blob))
    setSelection(null)
    setMessage('Drag a box around the object, then save it.')
  }

  async function saveSelection() {
    if (!selection || selection.width < 2 || selection.height < 2) {
      setMessage('Drag a box around the object first.')
      return
    }

    const query = new URLSearchParams({
      headX: Math.round(selection.x),
      headY: Math.round(selection.y),
      headWidth: Math.round(selection.width),
      headHeight: Math.round(selection.height),
    })
    const response = await fetch(`${api}/lesson16/reference?${query}`)
    if (!response.ok) {
      setMessage('Could not save the selection.')
      return
    }

    setMessage('Reference saved. Choose Match, then start the camera.')
  }

  return (
    <main className="lesson">
      <h1>Lesson 16</h1>

      <section>
        <h2>Camera engine</h2>
        <p>Opens Webcam 0, captures frames, calculates FPS and streams JPEGs.</p>
        <div className="controls">
          <button type="button" onClick={() => setMode('original')}>Original</button>
          <button type="button" onClick={() => setStreaming(true)}>Start</button>
          <button type="button" onClick={() => setStreaming(false)}>Stop</button>
        </div>
        {streaming
          ? <img className="camera" src={streamUrl} alt="Webcam stream" />
          : <p className="camera off">Camera is off.</p>}
      </section>

      <ModeButtons
        title="Image-processing functions"
        text="Reuse grayscale, blur, brightness, thresholding, HSV and morphology."
        modes={processingModes}
        mode={mode}
        onChoose={setMode}
      />

      <div className="controls">
        <label>
          HSV colour
          <select value={colorName} onChange={(event) => setColorName(event.target.value)}>
            {colors.map((color) => (
              <option key={color} value={color}>{color}</option>
            ))}
          </select>
        </label>
      </div>

      <ModeButtons
        title="Geometry and drawing"
        text="Resize, crop, rotate, flip, perspective, contours, circles and bounding boxes."
        modes={geometryModes}
        mode={mode}
        onChoose={setMode}
      />

      <ModeButtons
        title="ORB feature engine"
        text="Detects and compares descriptors, estimates homography and draws matching lines."
        modes={featureModes}
        mode={mode}
        onChoose={setMode}
      />

      <section>
        <h2>Reference selector</h2>
        <p>Uses your mouse to select an object from a real webcam frame, then saves it for matching.</p>
        <div className="controls">
          <button type="button" onClick={captureFrame}>Capture frame</button>
          <button type="button" onClick={saveSelection}>Save selection</button>
        </div>
        {frameUrl
          ? (
            <canvas
              ref={canvasRef}
              className="camera"
              onPointerDown={startDrag}
              onPointerMove={moveDrag}
              onPointerUp={endDrag}
              onPointerLeave={endDrag}
            />
          )
          : null}
        {message ? <p>{message}</p> : null}
      </section>
    </main>
  )
}

export default App
