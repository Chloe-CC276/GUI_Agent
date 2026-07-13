"""
Test Screen Capture

"""

from pathlib import Path

from src.perception.screen_capture import ScreenCapture

def main():

    capture = ScreenCapture()

    # Screen size
    width, height = capture.get_screen_size()

    print("=" * 50)
    print("Screen Size")
    print(width, height)


    # 捕获整个屏幕
    image = capture.capture_screen()

    save_path = Path("D:/GUIAgent_project/screenshots/full_screen.png")

    capture.save_image(image, save_path)

    print("=" * 50)
    print("Full screen saved.")


    # 展示
    capture.show_image(image)


    # 捕获指定区域
    region = capture.capture_region(
        left=0,
        top=0,
        width=1000,
        height=500,
    )

    region_path = Path("D:/GUIAgent_project/screenshots/region_capture.png")

    capture.save_image(region, region_path)

    print("=" * 50)
    print("Region saved.")

    capture.show_image(region)


if __name__ == "__main__":
    main()