from attractor import (
    sinspace, Performance_Renderer, ColorMap, Option, KeyframeInterpolator, Point, color_generic, CollapseMap
)

def main():
    # Here You define the main properties of a video render
    opts = Option(
        fps=10,
        frames=100,
        resolution=1000,
        colormap=ColorMap("viridis")
    )

    # array with values from lower to upper using a sinewave (p=1)
    # a, b are the initial values of the system used in the attractor
    # To animate this, at least one of these parameters should change each frame
    a = sinspace(0.32, 0.4, 300)



    # Main rendering class
    # Use this when rendering a video with many frames.
    renderer = Performance_Renderer(
        opts=opts,
        a=a,
        b=1.5,
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


def keyframe_example():
    # generic animation option (20 frames)
    opts = Option.from_time(
        seconds=4,
        fps=5
    )

    # ComplexPath is similar to linspace, sinspace, sqaurespace, ...
    # The difference you can define a set of point: point(value, frame)
    # between this points linspace is used to interpolate between those
    interpolation = KeyframeInterpolator(opts, 0.32, 0.32)
    interpolation.add_keyframe(Point(frame=5, value=0.4))
    interpolation.add_keyframe(Point(frame=10, value=0.4))
    interpolation.add_keyframe(Point(frame=15, value=0.32))

    a = interpolation.to_array()

    renderer = Performance_Renderer(
        opts=opts,
        a=a,
        b=1.5,
        iterations=3_000_000
    )
    renderer.set_static("a", False)
    renderer.show_demo(nth_frame=1)

def generic_example():
    opts = Option.from_time(
        seconds=4,
        fps=5
    )

    renderer = Performance_Renderer(
        opts=opts,
        a=0.35,
        b=1.5,
        iterations=3_000_000
    )
    # Set generic flag to save image in grayscale
    renderer.start_render_process("demo.mp4", save_as_generic=True)

    # after rendering you can apply a colormap (still takes some time)
    # make sure to use the same fps and frames as the rendering here
    color_generic("demo.mp4", ColorMap("viridis"), opts.fps, frames=opts.frames)


def mapping_all_stable_points():

    # Renders a map that shows were the attractor is stabel (not empty)
    # This attractor loops every 6.5 units
    map = CollapseMap(
        a_bounds=(0, 6.5),
        b_bounds=(0, 6.5),
        delta=0.01  # the smaller the delta the more pixels rendered
    )
    map.render()
    map.save("test")
    map.show()  # show interactive map with matplotlib

    map.load("test.pkl")  # load a save



if __name__ == "__main__":
    # see all colormaps available
    print(ColorMap.colormaps())
    # main()
    #keyframe_example()
