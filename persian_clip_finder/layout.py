"""Face detection and speaker clustering for smart cropping.

Uses OpenCV Haar cascades (bundled with opencv) to find faces across a few
sampled frames of a clip, then clusters them by horizontal position into
distinct "people" regions. Podcast/livestream speakers are roughly stationary,
so a stable per-person region computed over the clip works well and is cheap.
"""

from __future__ import annotations


def _detect_faces_in_frame(gray, cascade, min_size: int):
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(min_size, min_size),
    )
    return [tuple(int(v) for v in f) for f in faces]


def detect_people(
    source: str,
    start: float,
    end: float,
    samples: int = 12,
) -> list[dict]:
    """Return a list of person regions, each in source pixel coordinates:

        {"cx", "cy", "w", "h", "count"}  sorted left -> right by ``cx``.

    Returns an empty list if OpenCV/detection is unavailable or no faces found.
    """
    try:
        import cv2  # local import so the app loads even without opencv
    except Exception:
        return []

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return []

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    min_size = max(40, int(width * 0.04))

    boxes: list[tuple[int, int, int, int]] = []
    duration = max(0.1, end - start)
    for i in range(samples):
        t = start + (i + 0.5) * duration / samples
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes.extend(_detect_faces_in_frame(gray, cascade, min_size))
    cap.release()

    if not boxes:
        return []

    return _cluster_people(boxes, width, height, samples)


def _cluster_people(
    boxes: list[tuple[int, int, int, int]],
    width: int,
    height: int,
    samples: int,
) -> list[dict]:
    """Group face boxes by horizontal position into person clusters."""
    centers = sorted(
        ((x + w / 2, y + h / 2, w, h) for (x, y, w, h) in boxes),
        key=lambda c: c[0],
    )
    gap = width * 0.18  # new cluster when faces are far apart horizontally

    clusters: list[list[tuple[float, float, float, float]]] = [[centers[0]]]
    for c in centers[1:]:
        if c[0] - clusters[-1][-1][0] > gap:
            clusters.append([c])
        else:
            clusters[-1].append(c)

    min_hits = max(2, int(samples * 0.25))
    people: list[dict] = []
    for cl in clusters:
        if len(cl) < min_hits:
            continue
        n = len(cl)
        cx = sum(c[0] for c in cl) / n
        cy = sum(c[1] for c in cl) / n
        w = sum(c[2] for c in cl) / n
        h = sum(c[3] for c in cl) / n
        people.append(
            {"cx": cx, "cy": cy, "w": w, "h": h, "count": n}
        )

    people.sort(key=lambda p: p["cx"])
    return people
