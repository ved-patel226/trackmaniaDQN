import os
import subprocess
import time
import win32gui
import win32process
import win32con
import pyautogui

window_counter = 1


def enum_handler(hwnd, target_pid):
    """Window enumeration handler that finds and renames a window by its process ID"""
    global window_counter
    try:
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            thread_id, pid = win32process.GetWindowThreadProcessId(hwnd)
            if pid == target_pid or "TrackMania" in title:
                new_title = f"AI: {window_counter}"
                win32gui.SetWindowText(hwnd, new_title)
                print(f"Found window with PID {pid}, renamed to {new_title}")
                window_counter += 1
    except Exception as e:
        print(f"Error in enum_handler: {e}")
    return True


def make_n_instances(n):
    prev_path = os.getcwd()
    os.chdir(r"D:\Steam\steamapps\common\TrackMania Nations Forever")
    """Launch n instances of TrackMania Nations Forever"""
    for _ in range(n):
        proc = subprocess.Popen(["TMinterface.exe"])
        time.sleep(1)
        win32gui.EnumWindows(enum_handler, proc.pid)

        time.sleep(1)
        pyautogui.hotkey("ctrl", "win", "t")
    os.chdir(prev_path)


def delete_instances():
    """Kill all windows whose title starts with 'AI:'."""

    def enum_handler(hwnd, lParam):
        try:
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title.startswith("AI:"):
                    pid = win32process.GetWindowThreadProcessId(hwnd)[1]
                    print(f"Closing window with PID {pid} and title '{title}'")
                    win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
        except Exception as e:
            print(f"Error in enum_handler: {e}")
        return True

    win32gui.EnumWindows(enum_handler, None)


def main():
    make_n_instances(4)
    # delete_instances()


if __name__ == "__main__":
    main()
