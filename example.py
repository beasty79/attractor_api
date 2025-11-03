from attractor import sinspace, Performance_Renderer, ColorMap


def main():
    # array with values from lower to upper using a sinewave (p=1)
    # a, b are the initial values of the system used in the attractor
    # To animate this effectively, at least one of these parameters should change each frame
    a = sinspace(0.32, 0.4, 300)

    # Main rendering class
    # Use this when rendering a video with multiple frames.
    # For single-frame rendering, this class is overkill — use 'render_frame(...)' instead.
    renderer = Performance_Renderer(
        a=a,
        b=1.5,
        colormap=ColorMap("viridis"),
        frames=len(a),
        fps=10
    )

    # Important: 'a' is an array of values, one per frame (a[i] used for frame i)
    # So we need to mark it as non-static to allow per-frame variation
    renderer.set_static("a", False)

    # Show a lower resolution and frames preview
    # opens the demo via opencv and loops until u press Esc or close the window
    renderer.show_demo()

    # Set how many processes/threads to use (via multiprocessing.Pool)
    # Use None for unlimited; here we use 4 threads with a chunk size of 4
    # renderer.start_render_process("./your_filename.mp4", threads=4, chunksize=4)


if __name__ == "__main__":
    # see all colormaps available
    print(ColorMap.colormaps())
    main()
