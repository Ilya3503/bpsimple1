import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

for i in range(5):
    frames = pipeline.wait_for_frames()
    depth = frames.get_depth_frame()
    color = frames.get_color_frame()
    if not depth or not color:
        print("No frames")
    else:
        print(f"Depth shape: {depth.get_width()}x{depth.get_height()}, Color shape: {color.get_width()}x{color.get_height()}")

pipeline.stop()