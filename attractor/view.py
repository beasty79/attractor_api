from typing import Optional
from attractor.modMenu import SideWindow
from .colormap import ColorMap
import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np

if 0 != 0: from attractor.frame import SimonFrame

def play_video(video_path, fps=30):
    """
    Plays an .mp4 video in a loop using OpenCV, until 'q', 'Esc', or window close (X) is pressed.

    Args:
        video_path (str): Path to the video file.
        fps (float): Target frames per second for display.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return

    delay = 1 / fps  # Time per frame

    window_name = "Video"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()

        # Restart when video ends
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        cv2.imshow(window_name, frame)

        # Check for quit key or window close
        key = cv2.waitKey(int(delay * 1000)) & 0xFF
        if key in [ord('q'), 27]:  # 'q' or 'Esc'
            break

        # Detect if window is closed via 'X' button
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


def show_frame(frame: "SimonFrame"):
    img = frame.img
    resolution = (frame.resolution, frame.resolution)

    if not isinstance(img, np.ndarray):
        raise TypeError("Input muss ein NumPy ndarray sein.")

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Bild auf gewünschte Auflösung skalieren
    if resolution is not None:
        if len(resolution) != 2:
            raise ValueError("resolution muss ein Tupel (width, height) sein.")
        width, height = resolution
        if width <= 0 or height <= 0:
            raise ValueError("resolution-Werte müssen > 0 sein.")
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    # Erstelle das Fenster einmal
    window_name = "window"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1000, 1000)

    colormaps: list[str] = ColorMap.colormaps()
    current_index = colormaps.index(frame.colors.name)
    inverted = frame.colors.inverted

    mod_menu_open = False

    def modMenuUpdate():
        if mod_menu_open:
            window.updateUi()

    delta = 0.01
    while True:
        key = cv2.waitKey(1) & 0xFF

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        # Update Window Title dynamically
        title = f"a: {frame.a:.3f} b: {frame.b:.3f}        Colormap: {frame.colors.name} ({frame.colors.inverted})          collapsed={frame.collapsed}     delta={delta:6f}"
        cv2.setWindowTitle(window_name, title)

        # Anzeige
        img = cv2.cvtColor(frame.img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        cv2.imshow(window_name, img)


        # Prüfen, ob Fenster geschlossen wurde
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        # 'q' oder 'Esc' beendet die Anzeige
        if key in (ord('q'), 27):
            break

        elif key == ord('m'):
            if not mod_menu_open:
                window = SideWindow(frame)
                window.show()
                window.updateUi()
                mod_menu_open = True
            else:
                window.close()
                mod_menu_open = False

    cv2.destroyAllWindows()


def show_image(
    img: np.ndarray,
    resolution: Optional[tuple[int, int]] = None,  # None for dynamic
    a=None,
    b=None,
    colormap_name=None,
    inverted=None
):
    if not isinstance(img, np.ndarray):
        raise TypeError("Input muss ein NumPy ndarray sein.")

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    window_name = "window"
    if a is not None and b is not None:
        str_a = round(float(a), 2)
        str_b = round(float(b), 2)
        window_name = f"a: {str_a} b: {str_b}        Colormap: {colormap_name} ({inverted})"

    # Enable resizable window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # If fixed resolution, resize once
    if resolution is not None:
        if len(resolution) != 2:
            raise ValueError("resolution muss ein Tupel (width, height) sein.")
        width, height = resolution
        if width <= 0 or height <= 0:
            raise ValueError("resolution-Werte müssen > 0 sein.")
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    while True:
        # Get current window size for dynamic resizing
        if resolution is None:
            _, _, w, h = cv2.getWindowImageRect(window_name)  # returns x, y, w, h
            if w > 0 and h > 0:
                resized_img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            else:
                resized_img = img
        else:
            resized_img = img

        cv2.imshow(window_name, resized_img)

        key = cv2.waitKey(50) & 0xFF
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key in (ord('q'), 27):
            break

    cv2.destroyAllWindows()


def show_image_matplotlib(img: np.ndarray, A: list[float], B: list[float]):
    """
    Display an RGB image using matplotlib with hover coordinates, 
    and show the SimonFrame-rendered image in a second panel on click.

    Args:
        img (np.ndarray): RGB image as a NumPy array.
        A (list[float]): Values corresponding to x-axis.
        B (list[float]): Values corresponding to y-axis.
    """
    from .frame import SimonFrame
    if not isinstance(img, np.ndarray):
        raise TypeError("Input must be a NumPy ndarray.")
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError("Input image must be RGB or RGBA (HxWx3 or HxWx4).")

    # Create two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # First image
    im1 = ax1.imshow(img, origin='lower')
    ax1.set_title("Original Image")

    # Text box for coordinates on hover
    coord_text = ax1.text(0.02, 0.98, '', color='white',
                          transform=ax1.transAxes, verticalalignment='top',
                          bbox=dict(facecolor='black', alpha=0.5, pad=2))

    # Placeholder for second image
    im2 = ax2.imshow(np.zeros_like(img), origin='lower')
    ax2.set_title("Rendered Frame")

    # Mouse hover event for first image
    def on_mouse_move(event):
        if event.inaxes == ax1:
            x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
            try:
                x_val = A[x]
                y_val = B[y]
                coord_text.set_text(f"a: {x_val:.4f}, b: {y_val:.4f}")
                fig.canvas.draw_idle()
            except IndexError:
                return

            # Render SimonFrame
            frame = SimonFrame(
                a=x_val,
                b=y_val,
                n=500_000,
                resolution=500,
                colors=ColorMap("viridis", True)
            )
            frame.render()
            im2.set_data(frame.img)
            im2.set_extent([0, frame.img.shape[1], 0, frame.img.shape[0]])
            ax2.set_aspect('auto')
            ax2.set_title(f"Rendered Frame (a={x_val:.4f}, b={y_val:.4f})")
            ax2.set_xlim(0, frame.img.shape[1])
            ax2.set_ylim(0, frame.img.shape[0])
            fig.canvas.draw_idle()


    # Click event to render SimonFrame and update second panel
    def on_click(event):
        if event.inaxes == ax1:
            x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
            try:
                x_val = A[x]
                y_val = B[y]
            except IndexError:
                return

            # Render SimonFrame
            frame = SimonFrame(
                a=x_val,
                b=y_val,
                n=3_000_000,
                resolution=1000,
                colors=ColorMap("viridis")
            )
            frame.render()
            frame.show()

            # im2.set_data(frame.img)
            # im2.set_extent([0, frame.img.shape[1], 0, frame.img.shape[0]])
            # ax2.set_aspect('auto')
            # ax2.set_title(f"Rendered Frame (a={x_val:.4f}, b={y_val:.4f})")
            # ax2.set_xlim(0, frame.img.shape[1])
            # ax2.set_ylim(0, frame.img.shape[0])
            # fig.canvas.draw_idle()

    # Connect events
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
    fig.canvas.mpl_connect('button_press_event', on_click)

    # Hide axes
    ax1.axis('off')
    ax2.axis('off')

    plt.show()
