# attractor

**attractor** is a Python module for animating and analyzing the **Simon fractal** using efficient rendering techniques. It provides a clean API to generate frames, apply mathematical transformations, and export visualizations as videos.

---

## ✨ Features

- Animate the Simon fractal with customizable parameters
- Efficient video rendering using OpenCV
- Vectorized math functions powered by NumPy and Numba
- Extensible and modular API design
- Multiprocessing support for faster generation

---

## 📦 Installation

Clone the repo and install in editable mode for development:

```bash
git clone https://github.com/yourusername/attractor.git
cd attractor
pip install -e .
```


```python
from attractor import sinspace, Performance_Renderer, ColorMap

def main():
    # Create an array of values following a sinewave (period = 1)
    a = sinspace(0, 1, 100)

    # Initialize the main renderer
    renderer = Performance_Renderer(
        a=a,
        b=1.5,
        colormap=ColorMap("viridis"),
        frames=len(a),
    )

    # Important: mark 'a' as non-static (varies per frame)
    renderer.set_static("a", False)

    # Start rendering to a video file using 4 threads
    renderer.start_render_process("./your_file_path/your_filename.mp4", threads=4, chunksize=4)

if __name__ == "__main__":
    main()
```
