import pyrealsense2 as rs
import numpy as np
import cv2

def main():
    # 配置 RealSense 数据流
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 启用深度流（640x480 分辨率，30fps）
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    # 启动相机
    try:
        pipeline.start(config)
        print("相机启动成功")
    except Exception as e:
        print(f"相机启动失败: {e}")
        return

    # 获取深度图像的尺寸，计算中心点坐标
    depth_profile = pipeline.get_active_profile().get_stream(rs.stream.depth)
    width = depth_profile.as_video_stream_profile().width()
    height = depth_profile.as_video_stream_profile().height()
    center_x = width // 2
    center_y = height // 2
    print(f"图像尺寸: {width}x{height}, 中心点坐标: ({center_x}, {center_y})")

    try:
        while True:
            # 等待一帧数据（超时时间 5 秒）
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue

            # 获取中心点的深度值（单位：毫米）
            depth_value_mm = depth_frame.get_distance(center_x, center_y) * 1000.0
            # 将深度帧转换为 numpy 数组以便可视化
            depth_image = np.asanyarray(depth_frame.get_data())
            # 归一化并转为 8 位灰度图用于显示
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # 在深度图像的中心点画一个红色圆点
            cv2.circle(depth_colormap, (center_x, center_y), 5, (0, 0, 255), -1)
            # 显示深度值文本
            text = f"Depth: {depth_value_mm:.1f} mm"
            cv2.putText(depth_colormap, text, (center_x + 10, center_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 打印中心点的坐标和深度值
            print(f"中心点坐标: ({center_x}, {center_y}), 深度值: {depth_value_mm:.2f} mm")

            # 显示图像窗口
            cv2.imshow('RealSense D435i - Depth', depth_colormap)

            # 按 'q' 或 'ESC' 键退出
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    except KeyboardInterrupt:
        print("用户中断程序")
    except Exception as e:
        print(f"运行时错误: {e}")
    finally:
        # 停止相机并关闭所有窗口
        pipeline.stop()
        cv2.destroyAllWindows()
        print("相机已关闭")

if __name__ == "__main__":
    main()
