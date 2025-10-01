import cv2
import numpy as np
import mediapipe as mp

# FACEMESH_LIPS moved around across mediapipe versions; try both.
try:
    from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LIPS  # type: ignore[reportMissingImports]
except Exception:
    from mediapipe.solutions.face_mesh_connections import FACEMESH_LIPS  # type: ignore[reportMissingImports]

mp_face_mesh = mp.solutions.face_mesh


# ===== Camera config (adjust index if your phone opens instead of the Mac camera) =====
CAM_INDEX = 0
USE_AVFOUNDATION = True

# ===== MediaPipe setup =====
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,         # enables iris + detailed lips
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

def to_xy(landmark, w, h):
    """Convert a normalized landmark to integer pixel coords."""
    return int(landmark.x * w), int(landmark.y * h)

def draw_lips_wireframe(image_bgr, landmarks, color=(0, 255, 0), thickness=2):
    """Draw just the lip connections as lines."""
    h, w = image_bgr.shape[:2]
    for i, j in FACEMESH_LIPS:
        p1 = to_xy(landmarks[i], w, h)
        p2 = to_xy(landmarks[j], w, h)
        cv2.line(image_bgr, p1, p2, color, thickness, cv2.LINE_AA)

def draw_lips_mask(image_bgr, landmarks, alpha=0.5):
    """
    Render a filled, semi-transparent lip mask.
    We approximate the fill by collecting all unique lip points and
    running a convex hull. (Looks good in practice.)
    """
    h, w = image_bgr.shape[:2]
    pts = np.array([to_xy(landmarks[k], w, h) for k in sorted({i for e in FACEMESH_LIPS for i in e})], dtype=np.int32)
    if len(pts) < 3:
        return
    hull = cv2.convexHull(pts)

    overlay = image_bgr.copy()
    # Fill the hull
    cv2.fillConvexPoly(overlay, hull, (255, 255, 255))
    # Blend overlay onto original (white tint)
    cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0, dst=image_bgr)

def main():
    backend = cv2.CAP_AVFOUNDATION if USE_AVFOUNDATION else 0
    cap = cv2.VideoCapture(CAM_INDEX, backend) if USE_AVFOUNDATION else cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"Could not open camera index {CAM_INDEX}. Try a different index (0/1/2) or disable Continuity Camera.")
        return

    show_mask = False
    print("Press 'm' to toggle filled mask; 'q' or ESC to quit.")

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        # MediaPipe expects RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(frame_rgb)

        if result.multi_face_landmarks:
            # Only the first face (you can loop if you want multiple)
            face_landmarks = result.multi_face_landmarks[0].landmark

            # Optional: draw a semi-transparent filled mask
            if show_mask:
                draw_lips_mask(frame_bgr, face_landmarks, alpha=0.35)

            # Always: draw the lip wireframe on top
            draw_lips_wireframe(frame_bgr, face_landmarks, color=(0, 255, 0), thickness=2)

        cv2.imshow("Lip Tracker (MediaPipe)", frame_bgr)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('m'):
            show_mask = not show_mask
        elif key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
