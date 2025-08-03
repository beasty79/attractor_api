from attractor import sinspace, Performance_Renderer, ColorMap

def main():
    # array with values from lower to upper using a sinewave (p=1)
    a = sinspace(0, 1, 100)

    # Main Render class
    renderer = Performance_Renderer(
        a=a,
        b=1.5,
        colormap=ColorMap("viridis"),
        frames=len(a),
    )

    # this is important (bc a is given as array with arr[i] = arg for frame i)
    renderer.set_static("a", False)

    # allocate how many processes / threads (uses multiprocessing pool) you want (none for infinity)
    renderer.start_render_process("./your_file_path/your_filename.mp4", threads=4, chunksize=4)

if __name__ == "__main__":
    main()