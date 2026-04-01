import pyrealsense2 as rs
import numpy as np

class RealSenseCamera:
    def __init__(self, width=640, height=480, fps=30):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipeline.start(config)

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        if not depth or not color:
            return None, None
        depth_image = np.asanyarray(depth.get_data())
        color_image = np.asanyarray(color.get_data())
        return color_image, depth_image

    def get_pointcloud(self):
        color, depth = self.get_frame()
        if color is None:
            return None, None
        pc = rs.pointcloud()
        points = pc.calculate(rs.frame())
        v = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1,3)
        return v, color