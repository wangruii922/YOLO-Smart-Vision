import subprocess
import sys

BASE_CMD = sys.executable


def run(script):
    print("\n==============================")
    print(f"启动模块: {script}")
    print("==============================\n")

    subprocess.run([BASE_CMD, script])


def menu():

    while True:

        print("\n====================================")
        print("      YOLO Smart Vision System")
        print("====================================")

        print("\n【Detection 检测模块】")
        print("1. 图片检测")
        print("2. 视频检测")
        print("3. 摄像头检测")

        print("\n【Tracking 跟踪模块】")
        print("4. 目标跟踪")
        print("5. 轨迹可视化")

        print("\n【Analytics 分析模块】")
        print("6. 车辆计数分析")

        print("\n【System GUI 系统界面】")
        print("7. GUI系统（主界面）")

        print("\n0. 退出系统")

        choice = input("\n请选择功能: ")

        # ======================
        # Detection
        # ======================
        if choice == "1":
            run("detection/detect_image.py")

        elif choice == "2":
            run("detection/video_detect.py")

        elif choice == "3":
            run("detection/camera_detect.py")

        # ======================
        # Tracking
        # ======================
        elif choice == "4":
            run("tracking/track_video.py")

        elif choice == "5":
            run("tracking/draw_track.py")

        # ======================
        # Analytics
        # ======================
        elif choice == "6":
            run("analytics/count_vehicle.py")

        # ======================
        # GUI
        # ======================
        elif choice == "7":
            run("gui/main_window.py")

        # ======================
        # Exit
        # ======================
        elif choice == "0":
            print("\n系统退出")
            break

        else:
            print("无效输入，请重新选择")


if __name__ == "__main__":
    menu()