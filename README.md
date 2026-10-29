# GestureCAD – Hand Gesture Controlled CAD System

GestureCAD is a computer vision-based application that enables users to interact with and manipulate 2D shapes using real-time hand gestures. It uses hand tracking and gesture recognition to provide an intuitive, touch-free drawing and object manipulation experience.

The project combines computer vision, hand landmark detection, and interactive rendering to simulate CAD-style operations using natural hand movements.

## ✨ Features

### 1. Hover Pre-Highlighting
- Detects when the hand cursor moves within **70 pixels** of a shape.
- Displays a dotted cyan rectangle around the object.
- Shows a **"PINCH TO SELECT"** prompt before selection.
- Provides immediate visual feedback to improve selection accuracy.

### 2. CAD-Style Selection and Rotation
- Displays a rotation knob above the selected shape.
- Uses a rotation stem to provide a CAD-like interaction interface.
- Supports rotating objects through pinch-based dragging and hand movements relative to the object's center.

### 3. High-Precision Cursor Crosshairs
- Displays a bright cyan crosshair at the current pinch point.
- Updates the cursor position in real time as the hand moves.
- Helps users position their cursor accurately when interacting with shapes.

### 4. Air Drawing
- Enables drawing directly in the air using the index finger.
- Renders live neon-green strokes as the finger moves.
- Provides an intuitive, touch-free drawing experience.

### 5. Real-Time Hand Tracking
- Tracks hand movements through a camera feed.
- Converts detected hand landmarks into cursor movements and interaction commands.
- Enables gesture-based drawing, selection, and manipulation.

## 🛠️ Technologies Used

- **Python** – Core application logic
- **OpenCV** – Camera input and image processing
- **MediaPipe** – Hand landmark detection and tracking
- **NumPy** – Numerical operations and coordinate processing
- **Pygame / Rendering utilities** – Interactive graphics and visual feedback, depending on the implementation

## 📂 Project Structure

```text
GestureCAD/
└── polygon/
    └── src/
        ├── main.py
        ├── geometry.py
        ├── gesture_detector.py
        ├── hand_tracker.py
        ├── renderer.py
        ├── state_machine.py
        ├── transform3d.py
        └── ui.py
```

**Main components:**

- `main.py` – Application entry point and main execution loop.
- `geometry.py` – Shape geometry and related calculations.
- `gesture_detector.py` – Detects hand gestures and interaction states.
- `hand_tracker.py` – Tracks hand landmarks from the camera.
- `renderer.py` – Renders shapes, highlights, crosshairs, and drawing strokes.
- `state_machine.py` – Manages interaction states and transitions.
- `transform3d.py` – Handles transformation-related calculations.
- `ui.py` – Manages interface elements and visual prompts.

*Note: The descriptions above reflect the intended responsibilities suggested by the filenames. Adjust them if your implementation differs.*

## ⚙️ Installation

### Prerequisites

- Python 3.10 or a compatible version supported by the project's dependencies
- A webcam
- Windows, Linux, or macOS

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd GestureCAD
```

Replace `<YOUR_GITHUB_REPOSITORY_URL>` with your repository URL.

### 2. Install Dependencies

Install the required libraries:

```bash
pip install opencv-python mediapipe numpy pygame
```

If the project includes a `requirements.txt` file, install the dependencies using:

```bash
pip install -r requirements.txt
```

### 3. Run the Application

From the project root directory, run:

```bash
python polygon/src/main.py
```

Allow webcam access when prompted.

## 🖐️ How to Use

1. Launch the application and allow access to your webcam.
2. Position your hand within the camera's field of view.
3. Move your hand cursor over a shape to display the cyan hover highlight.
4. Follow the **"PINCH TO SELECT"** prompt and pinch your fingers to select the shape.
5. Use the rotation control to rotate the selected object.
6. Extend your index finger to draw live neon-green strokes in the air.
7. Watch the crosshair to position your cursor precisely.

*The exact gestures and controls depend on the implemented gesture-detection logic.*

## 🔬 Computer Vision Workflow

1. **Camera acquisition:** Capture frames from the webcam.
2. **Hand landmark detection:** Detect hand landmarks and track their positions.
3. **Gesture interpretation:** Determine whether the user is hovering, pinching, drawing, or manipulating an object.
4. **Coordinate mapping:** Convert hand positions into application coordinates.
5. **Interaction processing:** Update selection, drawing, and rotation states.
6. **Rendering:** Display shapes, hover highlights, rotation controls, cursor crosshairs, and drawing strokes in real time.

## 🎯 Applications

- Touch-free CAD interfaces
- Gesture-controlled drawing applications
- Human-computer interaction research
- Interactive design and prototyping
- Computer vision and hand gesture recognition experiments

## 🚀 Future Improvements

- Add support for more CAD operations, including scaling and translation.
- Introduce undo and redo functionality.
- Support additional shapes and geometric editing tools.
- Improve tracking stability under different lighting conditions.
- Add gesture customization and calibration.
- Enable saving and exporting drawings.
- Explore 3D object manipulation and model export.

## 👨‍💻 Project Objective

The objective of GestureCAD is to explore how real-time computer vision and hand gesture recognition can make graphical design tools more intuitive and accessible by reducing dependence on traditional mouse-and-keyboard interactions.

