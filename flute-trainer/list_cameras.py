"""Find which camera number is which.   python list_cameras.py
Opens each camera number 0..9, shows a preview for 2 seconds, and prints what it found.
Look at the preview: the one showing your phone's picture is the number to use as --camera2."""
import cv2

from camera import open_camera


def main():
    found = []
    for i in range(10):
        cap = open_camera(i)
        if cap is None:
            continue
        ok, frame = cap.read()
        h, w = frame.shape[:2] if ok else (0, 0)
        print(f"camera {i}: works, {w}x{h}")
        found.append(i)
        for _ in range(40):
            ok, frame = cap.read()
            if not ok:
                break
            cv2.putText(frame, f"camera {i}  (press any key for next)", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.imshow("Camera finder", frame)
            if cv2.waitKey(50) != -1:
                break
        cap.release()
    cv2.destroyAllWindows()
    if not found:
        print("No camera could be opened. Close other apps using the camera (Zoom, Meet, Teams).")
    elif len(found) == 1:
        print("\nOnly one camera was found. The phone is NOT visible to the laptop yet:")
        print("  - open the phone app (DroidCam / Iriun / Camo) on BOTH phone and laptop")
        print("  - press Start / Connect in the laptop app, and wait for it to say connected")
        print("  - then run this script again")
    else:
        print(f"\nUse for example:  python app.py --sa C5 --camera {found[0]} --camera2 {found[-1]}")


if __name__ == "__main__":
    main()
