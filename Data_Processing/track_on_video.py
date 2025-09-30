import cv2
import math
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

LIPS_OUTER = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308,
    324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39,
    37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82,
    81, 80, 191, 78
]
LIPS_INNER = [
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 291,
    375, 321, 405, 314, 17, 84, 181, 91, 146, 61
]

LIPS_IDX = sorted(set(LIPS_OUTER + LIPS_INNER))

def get_lip_landmarks(video_path):
    """
    Yield per-frame lip landmark coordinates from a video.
    Returns an iterator of dicts:
      {
        "frame": int,
        "time": float (seconds),
        "points": List[(x_px, y_px)] in the order of LIPS_IDX
      }
    If a frame has no face, 'points' is an empty list.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,     # adds iris + refined mouth contours
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as mesh:

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = mesh.process(rgb)

            points = []
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                for i in LIPS_IDX:
                    x = int(lm[i].x * w)
                    y = int(lm[i].y * h)
                    points.append((x, y))

            yield {
                "frame": frame_idx,
                "time": frame_idx / fps,
                "points": points
            }
            frame_idx += 1

    cap.release()

def annotate_lips_on_video(in_path, out_path=None, show=True):
    cap0 = cv2.VideoCapture(in_path)
    if not cap0.isOpened():
        raise RuntimeError(f"Cannot open {in_path}")
    fps = cap0.get(cv2.CAP_PROP_FPS) or 30.0
    w   = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap0.release()

    # Optional writer
    writer = None
    if out_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    # map from landmark index -> position in rec["points"] list
    idx_pos = {i: k for k, i in enumerate(LIPS_IDX)}

    # iterate landmarks
    for rec in get_lip_landmarks(in_path):
        frame_idx, t = rec["frame"], rec["time"]

        # if you also need the raw frame, reopen capture and seek (slower):
        # For speed, prefer to read frames once and run mediapipe in the same loop.
        # Here we re-open just for simplicity with your existing generator:
        if frame_idx == 0:
            cap = cv2.VideoCapture(in_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break

        pts = rec["points"]
        if pts:
            # rebuild ordered outer/inner polygons using our index mapping
            outer_pts = [pts[idx_pos[i]] for i in LIPS_OUTER if i in idx_pos]
            inner_pts = [pts[idx_pos[i]] for i in LIPS_INNER if i in idx_pos]

            # draw polylines
            if outer_pts:
                cv2.polylines(frame, [np_int(outer_pts)], isClosed=True, color=(0,255,0), thickness=2)
            if inner_pts:
                cv2.polylines(frame, [np_int(inner_pts)], isClosed=True, color=(0,200,255), thickness=2)

            # tiny circles on each point (optional)
            for x, y in outer_pts:
                cv2.circle(frame, (x, y), 1, (0,255,0), -1)
            for x, y in inner_pts:
                cv2.circle(frame, (x, y), 1, (0,200,255), -1)

            # compute and show aperture (inner vertical gap normalized by mouth width)
            def euclid(a,b): return math.hypot(a[0]-b[0], a[1]-b[1])
            have = all(j in idx_pos for j in (61,291,13,14))
            if have:
                p61 = pts[idx_pos[61]]  # left mouth corner
                p291= pts[idx_pos[291]] # right mouth corner
                p13 = pts[idx_pos[13]]  # upper inner lip
                p14 = pts[idx_pos[14]]  # lower inner lip
                width = euclid(p61, p291)
                if width > 1e-6:
                    aperture = euclid(p13, p14) / width
                    cv2.putText(frame, f"t={t:.2f}s  aperture={aperture:.3f}",
                                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,255,50), 2)

        else:
            cv2.putText(frame, f"t={t:.2f}s  (no face detected)",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        if show:
            cv2.imshow("Lip tracking", frame)
            if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
                break

        if writer:
            writer.write(frame)

    if show:
        cv2.destroyAllWindows()
    if writer:
        writer.release()
    if 'cap' in locals():
        cap.release()
    return out_path

def np_int(points_list):
    """Convert list[(x,y)] -> int32 Nx1x2 array for cv2.polylines."""
    import numpy as np
    arr = np.array(points_list, dtype=np.int32).reshape(-1,1,2)
    return arr

def track_lip_point_speeds(video_path, normalize_by_width=False):
    # ChatGPT wrote this
    results = []
    prev_pts = None

    for rec in get_lip_landmarks(video_path): 
        pts = rec["points"] 
        n = len(pts)
        dx = [None]*n
        dy = [None]*n
        sp = [None]*n

        norm = 1.0
        if normalize_by_width and n > 0:
            idx_to_pt = {i: p for i, p in zip(LIPS_IDX, pts)}
            if 61 in idx_to_pt and 291 in idx_to_pt:
                x1,y1 = idx_to_pt[61]
                x2,y2 = idx_to_pt[291]
                width = math.hypot(x1-x2, y1-y2)
                if width > 1e-6:
                    norm = width

        if prev_pts is not None and len(prev_pts) == n and n > 0:
            for k in range(n):
                x, y = pts[k]
                px, py = prev_pts[k]
                ddx = (x - px) / norm
                ddy = (y - py) / norm
                dx[k] = ddx
                dy[k] = ddy
                sp[k] = math.hypot(ddx, ddy)
            valid = [v for v in sp if v is not None]
            mean_speed = sum(valid)/len(valid) if valid else None
            max_speed  = max(valid) if valid else None
        else:
            mean_speed = None
            max_speed  = None
        results.append({
            "frame": rec["frame"],
            "time": rec["time"],
            "points": pts,
            "dx": dx,
            "dy": dy,
            "speed": sp,
            "mean_speed": mean_speed,
            "max_speed": max_speed
        })
        prev_pts = pts if pts else None 
    return results