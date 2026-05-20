import pyrealsense2 as rs
import numpy as np
import cv2

def main():
    # 创建 pipeline 和配置对象
    pipeline = rs.pipeline()
    config = rs.config()

    # 启用彩色流（分辨率 640x480，30fps，RGB 格式）
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # 启用深度流（分辨率 640x480，30fps，Z16 格式）
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # 启动相机
    try:
        pipeline.start(config)
        print("相机启动成功，按 'q' 或 ESC 键退出")
    except Exception as e:
        print(f"相机启动失败: {e}")
        return

    try:
        while True:
            # 等待一帧数据（超时 5 秒）
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # 将彩色帧转为 numpy 数组（已经是 BGR 格式）
            color_image = np.asanyarray(color_frame.get_data())

            # 将深度帧转为 numpy 数组，并映射为伪彩色图（方便观察）
            depth_image = np.asanyarray(depth_frame.get_data())
            # 将深度值（毫米级）归一化到 0-255 并应用颜色映射
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # 可选：在深度图像中心显示深度值（与之前示例类似，但不是必须）
            h, w = depth_image.shape
            center_x, center_y = w // 2, h // 2
            depth_mm = depth_frame.get_distance(center_x, center_y) * 1000.0
            cv2.circle(depth_colormap, (center_x, center_y), 4, (0, 0, 255), -1)
            cv2.putText(depth_colormap, f"{depth_mm:.1f}mm", (center_x+10, center_y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            # 显示两个图像窗口
            cv2.imshow('Color Image', color_image)
            cv2.imshow('Depth Image (Colormap)', depth_colormap)

            # 按键退出
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:   # 27 是 ESC
                break

    except KeyboardInterrupt:
        print("用户中断")
    except Exception as e:
        print(f"运行时错误: {e}")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("相机已关闭")

if __name__ == "__main__":
    main()