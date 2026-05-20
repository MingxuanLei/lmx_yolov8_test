import pyrealsense2 as rs
import cv2
import numpy as np
import time
from datetime import datetime

def main():
    # 配置 RealSense 流
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 启用彩色流（分辨率 640x480，30fps，BGR 格式）
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # 启用深度流（分辨率 640x480，30fps，Z16 格式）—— 仅用于对齐，不保存
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    # 启动相机
    try:
        pipeline.start(config)
        print("相机启动成功")
    except Exception as e:
        print(f"相机启动失败: {e}")
        return
    
    # 创建对齐对象（深度对齐到彩色，确保彩色图像中心点深度有效）
    align = rs.align(rs.stream.color)
    
    # 获取彩色图像尺寸
    profile = pipeline.get_active_profile()
    color_profile = profile.get_stream(rs.stream.color)
    color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
    width, height = color_intrinsics.width, color_intrinsics.height
    fps = 30  # 与配置一致
    
    # 准备保存路径（以时间戳命名）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"realsense_video_{timestamp}.mp4"
    video_path = video_filename  # 保存在当前目录，可修改路径
    
    # 初始化 MP4 视频写入器（使用 'mp4v' 编码）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print("无法创建 MP4 文件，请检查编码器或路径权限")
        pipeline.stop()
        return
    
    print(f"录制开始，视频将保存为: {video_path}")
    print("按 'q' 停止录制...")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # 等待一帧并对齐
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            aligned_frames = align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            if not color_frame:
                continue
            
            # 转为 numpy 数组
            color_image = np.asanyarray(color_frame.get_data())
            
            # 实时显示彩色图像（带录制标记）
            display = color_image.copy()
            cv2.putText(display, f"Recording... Frame {frame_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("RealSense Color Recording", display)
            
            # 写入视频帧
            video_writer.write(color_image)
            frame_count += 1
            
            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("用户中断")
    except Exception as e:
        print(f"运行时错误: {e}")
    finally:
        video_writer.release()
        pipeline.stop()
        cv2.destroyAllWindows()
        
        elapsed = time.time() - start_time
        print(f"录制完成，共保存 {frame_count} 帧")
        print(f"视频文件: {video_path}")
        print(f"实际帧率: {frame_count/elapsed:.2f} fps")

if __name__ == "__main__":
    main()