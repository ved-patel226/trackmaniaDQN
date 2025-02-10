import mss
import pygetwindow as gw
import numpy as np
import cv2
import time


def capture_window(window_title, sct=None):
    # Reuse mss instance instead of creating new one each time
    if sct is None:
        sct = mss.mss()

    try:
        # Cache window reference
        window = gw.getWindowsWithTitle(window_title)[0]

        # Pre-define monitor dict
        monitor = {
            "top": window.top,
            "left": window.left,
            "width": window.width,
            "height": window.height,
        }

        # Direct numpy array conversion
        screenshot = np.array(sct.grab(monitor))

        # Combine operations to reduce memory allocations
        screenshot = cv2.resize(
            cv2.cvtColor(screenshot, cv2.COLOR_BGRA2GRAY), (128, 128)
        )

        return np.expand_dims(screenshot, axis=-1)

    except IndexError:
        print("Window not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def main() -> None:
    # Create single mss instance
    sct = mss.mss()

    time.sleep(1)
    start_time = time.perf_counter()

    # Pass sct instance to reuse
    capture_window("AI: 1", sct)

    print(f"Time taken: {time.perf_counter() - start_time:.5f}s")


if __name__ == "__main__":
    main()
