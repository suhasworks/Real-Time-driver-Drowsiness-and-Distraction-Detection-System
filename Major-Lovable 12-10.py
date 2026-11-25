import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import time
import os
import urllib.request
import threading
import queue

# Offline text-to-speech for the "check your mirror" voice reminders.
try:
    import pyttsx3
    _HAS_TTS = True
except ImportError:
    _HAS_TTS = False
    print("[WARN] pyttsx3 not installed -> voice reminders disabled. "
          "Install with: pip install pyttsx3")

# Audio alert backend: winsound is Windows-only and built into the standard
# library (no extra install). On other platforms we fall back to the
# terminal bell character, which is audible on most systems/terminals.
try:
    import winsound
    _HAS_WINSOUND = True
except ImportError:
    _HAS_WINSOUND = False

# ─── Configuration ────────────────────────────────────────────────────────────
EAR_THRESHOLD        = 0.22
EAR_CLOSED_SECONDS   = 1.5   # eyes must stay below threshold for this many seconds

# Head-turn (yaw) detection for "signal your turn" warning.
# Camera is NOT flipped, so movement on the screen is the opposite of the
# driver's actual side (screen-right head movement = driver's actual LEFT turn).
YAW_THRESHOLD     = 0.12   # how far the nose must skew off-center (normalized)
YAW_CONSEC_FRAMES = 10     # frames the skew must be sustained before it counts

# Set to True to pop up a small window showing the hand-ROI edge map used for
# phone detection, and print candidate contour stats to the console.
PHONE_DEBUG = False

# Object detection (ML-based, via MediaPipe Object Detector).
# The detector now reports ALL COCO object classes it recognizes (not just
# "cell phone") so that a bottle, remote, cup, etc. shows up labeled as what
# it actually is instead of being forced into a "PHONE DETECTED" bucket.
#
# Two DIFFERENT thresholds are used on purpose:
#   OBJECT_DETECTOR_SCORE_THRESHOLD -> passed to MediaPipe itself. Anything
#     below this is discarded by the model BEFORE it ever reaches our Python
#     code, so it must be kept LOW, or a weak-but-real candidate (a held
#     phone at low contrast, say) never shows up at all for Path 1/2 below
#     to even consider -- this was the exact failure where nothing but
#     "PERSON" appeared even though a phone was clearly in frame.
#   OBJECT_SCORE_THRESHOLD -> our own Python-side cutoff for what's drawn as
#     a plain "other object" grey box, kept higher so the screen isn't
#     cluttered with near-zero-confidence junk guesses.
OBJECT_DETECTOR_SCORE_THRESHOLD = 0.15   # MediaPipe-side cutoff (keep LOW)
OBJECT_SCORE_THRESHOLD           = 0.35  # our display cutoff for "other" boxes

# Alert on ANY "cell phone" detection above this confidence — flat threshold,
# no face-position distinction (this WILL also catch finger/hand gestures
# near the face that momentarily read as "cell phone" — see note below).
PHONE_SCORE_AWAY_FACE  = 0.25
PHONE_SCORE_NEAR_FACE  = 0.25   # same as above -> face-overlap logic is now a no-op
FACE_OVERLAP_FRACTION  = 0.30   # kept for reference; has no effect while the two scores above are equal

PHONE_HAND_MARGIN      = 20     # px of slack between phone box and hand box
                                  # (kept tight so an empty hand's own padded
                                  #  region doesn't "swallow" unrelated boxes)
PHONE_CONSEC_FRAMES    = 5      # phone-in-hand must be seen this many
                                  # consecutive frames before it counts —
                                  # filters out single-frame misclassifications

# ── Path 2: shape/context fallback for a held phone the model mislabels ───────
# EfficientDet gives exactly ONE label per box (it's a single-label detector,
# not top-k per box), so if its best guess for a held phone is "backpack" or
# "remote", "cell phone" never appears anywhere in that detection -- raising
# PHONE_SCORE_* thresholds can't fix that, because the label itself is wrong.
# This fallback instead asks: "is this small, hand-held, phone-shaped object
# ALSO one of a short list of classes real phones commonly get confused
# with?" If so, and it passes simple size/aspect-ratio sanity checks for a
# held phone (not an actual backpack, which is much bigger), it still counts.
CONFUSABLE_PHONE_LABELS = {"backpack", "handbag", "remote", "book", "mouse", "clock"}
PHONE_SHAPE_SCORE_MIN     = 0.15   # min confidence for a fallback (confusable-label) match
PHONE_SHAPE_MIN_ASPECT    = 0.20   # min width/height ratio of the box (tall & narrow)
PHONE_SHAPE_MAX_ASPECT    = 0.85   # max width/height ratio of the box
PHONE_SHAPE_MAX_WIDTH_FRAC  = 0.28 # box width must be <= this fraction of frame width
PHONE_SHAPE_MAX_HEIGHT_FRAC = 0.55 # box height must be <= this fraction of frame height

# ── Path 3: pure edge/contour fallback, scoped to each hand's own region ──────
# Covers the case where the ML model proposes NO box at all near the hand
# (e.g. a dark phone against a bright wall can push every class's score
# below OBJECT_DETECTOR_SCORE_THRESHOLD, so nothing reaches Python at all --
# not even a wrong label). This is the same Canny-edge/contour idea from an
# earlier version of this script, but deliberately demoted to a last-resort
# signal: it only runs inside a detected hand's own padded box (never the
# full frame), and still requires PHONE_CONSEC_FRAMES of sustained detection
# before it counts, same as Paths 1 and 2. Set PATH3_ENABLED = False to turn
# this off if it ever causes false positives in your environment.
PATH3_ENABLED        = True
PATH3_MIN_AREA        = 700   # min filled contour area, in pixels
PATH3_MIN_W           = 18    # min bounding-box width, in pixels
PATH3_MIN_H           = 28    # min bounding-box height, in pixels
PATH3_MIN_ASPECT      = 0.18  # min width/height ratio (tall phone silhouettes)
PATH3_MAX_ASPECT      = 0.95  # max width/height ratio
PATH3_MIN_EXTENT      = 0.28  # min (contour area / bounding-box area) -> rejects
                                # thin scattered edge noise, keeps solid rectangles

# Audio alerts
AUDIO_ENABLED  = True
AUDIO_COOLDOWN = 1.2    # min seconds between repeated beeps of the SAME alert
                         # type, so it beeps steadily instead of spamming every frame

# Model file paths (auto-downloaded if missing)
FACE_MODEL_PATH   = "face_landmarker.task"
HAND_MODEL_PATH   = "hand_landmarker.task"
OBJECT_MODEL_PATH = "efficientdet_lite2.tflite"

FACE_MODEL_URL   = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
HAND_MODEL_URL   = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
# EfficientDet-Lite2, COCO-trained — includes a real "cell phone" class.
# Upgraded from Lite0: Lite0 is the smallest/fastest variant and was
# frequently misclassifying a held phone as "backpack", "remote", etc.
# (low confidence, wrong top-1 label). Lite2 is a noticeably more accurate
# detector at a modest speed cost -- still real-time on a normal laptop CPU.
OBJECT_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/float32/latest/efficientdet_lite2.tflite"

# Eye landmark indices for MediaPipe 478-point model
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# Landmarks used to estimate head yaw (left/right head turn)
NOSE_TIP   = 1
FACE_LEFT  = 234   # image-left cheek landmark
FACE_RIGHT = 454   # image-right cheek landmark


# ─── Auto-download models ─────────────────────────────────────────────────────

def download_model(url, path):
    if os.path.exists(path):
        print(f"[OK] Model already exists: {path}")
        return
    print(f"[DOWNLOAD] Downloading {path} ...")
    try:
        urllib.request.urlretrieve(url, path)
        print(f"[OK] Downloaded: {path}")
    except Exception as e:
        print(f"[ERROR] Failed to download {path}: {e}")
        print(f"  Please manually download from:\n  {url}")
        exit(1)


# ─── Eye Aspect Ratio ─────────────────────────────────────────────────────────

def eye_aspect_ratio(landmarks, eye_indices, img_w, img_h):
    pts = np.array([(landmarks[i].x * img_w, landmarks[i].y * img_h)
                    for i in eye_indices])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C)


def draw_eye_outline(frame, landmarks, indices, img_w, img_h, color):
    pts = np.array([(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h))
                    for i in indices], dtype=np.int32)
    cv2.polylines(frame, [cv2.convexHull(pts)], True, color, 1)


# ─── Head Yaw (turn) Detection ─────────────────────────────────────────────────

def estimate_yaw_ratio(landmarks):
    """
    Returns a normalized value describing how far the nose has skewed off the
    center of the face, using image (screen) coordinates.
      > 0  -> nose has moved toward the RIGHT of the screen
      < 0  -> nose has moved toward the LEFT of the screen
    Since the camera is not mirrored, screen-right movement corresponds to the
    driver physically turning their head to their own LEFT, and screen-left
    movement corresponds to the driver turning to their own RIGHT.
    """
    nose  = landmarks[NOSE_TIP]
    left  = landmarks[FACE_LEFT]
    right = landmarks[FACE_RIGHT]
    face_width = abs(right.x - left.x)
    if face_width < 1e-6:
        return 0.0
    center_x = (left.x + right.x) / 2.0
    return (nose.x - center_x) / face_width


# ─── Phone Detection ──────────────────────────────────────────────────────────

def get_hand_boxes(hand_result, w, h, pad=35):
    """Returns a padded bounding box (x1,y1,x2,y2) per detected hand, in pixels."""
    boxes = []
    if not hand_result or not hand_result.hand_landmarks:
        return boxes
    for hand_lms in hand_result.hand_landmarks:
        xs = [lm.x * w for lm in hand_lms]
        ys = [lm.y * h for lm in hand_lms]
        x1 = max(0,   int(min(xs)) - pad)
        y1 = max(0,   int(min(ys)) - pad)
        x2 = min(w-1, int(max(xs)) + pad)
        y2 = min(h-1, int(max(ys)) + pad)
        boxes.append((x1, y1, x2, y2))
    return boxes


def _boxes_overlap(a, b, margin=0):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return (ax1 - margin < bx2 and ax2 + margin > bx1 and
            ay1 - margin < by2 and ay2 + margin > by1)


def _overlap_fraction(box, other):
    """Fraction of `box`'s area that overlaps `other`. 0 if no overlap or other is None."""
    if other is None:
        return 0.0
    ax1, ay1, ax2, ay2 = box
    bx1, by1, bx2, by2 = other
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter_area = (ix2 - ix1) * (iy2 - iy1)
    box_area = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    return inter_area / box_area


def _is_phone_shaped(box, frame_w, frame_h):
    """True if `box` has the rough size/aspect-ratio of a held phone (tall,
    narrow, and small relative to the frame) — used only as a secondary
    signal for the CONFUSABLE_PHONE_LABELS fallback below, never on its own."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0 or frame_w <= 0 or frame_h <= 0:
        return False
    # [FIX] orientation-independent: short side / long side, so a phone held
    # sideways, tilted or edge-on to the ear is accepted, not just upright.
    aspect = min(bw, bh) / max(bw, bh)
    if not (PHONE_SHAPE_MIN_ASPECT <= aspect <= 1.0):
        return False
    if max(bw, bh) > max(frame_w * PHONE_SHAPE_MAX_WIDTH_FRAC,
                         frame_h * PHONE_SHAPE_MAX_HEIGHT_FRAC):
        return False
    return True


# [FIX] Path 3 was configured above but never implemented. This looks for a
# solid, phone-like (rotated) rectangle inside / right next to each hand.
_PHONE_HOLD = {"boxes": [], "left": 0}
PHONE_HOLD_FRAMES = 3   # keep last phone box for a few frames so a 1-frame
                        # miss at a hard angle doesn't reset the confirm counter

def _path3_contour_phone(rgb, hand_boxes, face_box):
    if not PATH3_ENABLED or rgb is None or not hand_boxes:
        return []
    H, W = rgb.shape[:2]
    gray = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)
    found = []
    for (hx1, hy1, hx2, hy2) in hand_boxes:
        # widen the search area: the phone often sticks out of the hand
        pw, ph = int((hx2 - hx1) * 0.35), int((hy2 - hy1) * 0.35)
        x1, y1 = max(0, hx1 - pw), max(0, hy1 - ph)
        x2, y2 = min(W, hx2 + pw), min(H, hy2 + ph)
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        roi = cv2.GaussianBlur(roi, (5, 5), 0)
        # dark phone body (relative to this hand region's brightness) + edges
        thr = min(90, int(np.percentile(roi, 25)))
        _, dark = cv2.threshold(roi, thr, 255, cv2.THRESH_BINARY_INV)
        edges = cv2.dilate(cv2.Canny(roi, 40, 120), None, iterations=2)
        mask = cv2.bitwise_or(dark, edges)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_area = roi.shape[0] * roi.shape[1]
        for c in cnts:
            area = cv2.contourArea(c)
            if area < PATH3_MIN_AREA or area > roi_area * 0.6:
                continue
            (cx, cy), (rw, rh), _ang = cv2.minAreaRect(c)   # rotation-aware
            if min(rw, rh) < PATH3_MIN_W or max(rw, rh) < PATH3_MIN_H:
                continue
            aspect = min(rw, rh) / max(rw, rh)
            if not (PATH3_MIN_ASPECT <= aspect <= PATH3_MAX_ASPECT):
                continue
            if area / max(1.0, rw * rh) < max(PATH3_MIN_EXTENT, 0.55):
                continue   # must be a solid rectangle, not hair / shadows
            bx, by, bw, bh = cv2.boundingRect(c)
            box = (x1 + bx, y1 + by, x1 + bx + bw, y1 + by + bh)
            if face_box is not None and _overlap_fraction(box, face_box) > 0.6:
                continue   # mostly inside the face -> glasses/beard, not phone
            found.append((box[0], box[1], box[2], box[3], "phone-shape", 0.30))
    return found


def detect_objects(object_detector, mp_image, timestamp, hand_boxes, face_box=None,
                    frame_w=None, frame_h=None):
    """
    Runs the MediaPipe Object Detector (EfficientDet-Lite, all COCO classes)
    and splits the results into two lists:

      phone_boxes  -> confirmed as a held phone, via EITHER of two paths:

        (1) Direct label match: top-1 label is "cell phone", AND confidence
            clears a position-aware bar:
              - if the box overlaps the face heavily (>= FACE_OVERLAP_FRACTION)
                -> needs PHONE_SCORE_NEAR_FACE (high bar; this is the common
                false-positive zone for a finger/hand gesture near the face)
              - otherwise -> needs only PHONE_SCORE_AWAY_FACE (low bar; a
                real phone held to the ear/side often scores modestly but
                sits outside the face outline)

        (2) Shape/context fallback: EfficientDet gives exactly ONE label per
            box, so if its best guess for a held phone is wrong (commonly
            "backpack", "remote", "book", etc. at LOW-ish confidence — the
            exact failure where a held phone shows up as "BACKPACK (38%)"),
            "cell phone" never appears in that detection at all, and no
            amount of raising PHONE_SCORE_* fixes a wrong label. This path
            instead accepts a detection whose label is one of a short list
            of phone-confusable classes (CONFUSABLE_PHONE_LABELS), IF its
            box is phone-sized/phone-shaped (see _is_phone_shaped) and it
            clears a lower confidence floor (PHONE_SHAPE_SCORE_MIN). This
            deliberately does NOT fire for a large box (an actual backpack
            worn on the shoulder, say) since real backpacks fail the
            size/aspect check.

      Both paths additionally require: if a hand is visible, the box must
      overlap it (filters a phone sitting on the dashboard). If NO hand is
      visible at all, the detection is trusted on its own — fingers curling
      around a held phone often hide the hand from HandLandmarker entirely.

      other_boxes  -> everything else that passed OBJECT_SCORE_THRESHOLD,
                      each labeled with its REAL class name (e.g. "bottle",
                      "cup", "remote") instead of being folded into "phone".

    Each item: (x1, y1, x2, y2, category_name, score)
    """
    result = object_detector.detect_for_video(mp_image, timestamp)
    phone_boxes = []
    other_boxes = []

    if result and result.detections:
        for det in result.detections:
            if not det.categories:
                continue
            cat = det.categories[0]
            # [FIX] only drop low scores for non-phone objects; a held phone
            # at an angle often scores 0.15-0.35 and was discarded here.
            if cat.score < OBJECT_SCORE_THRESHOLD and \
                    cat.category_name != "cell phone" and \
                    cat.category_name not in CONFUSABLE_PHONE_LABELS:
                continue
            bbox = det.bounding_box
            x1, y1 = bbox.origin_x, bbox.origin_y
            x2, y2 = x1 + bbox.width, y1 + bbox.height
            box = (x1, y1, x2, y2)

            if hand_boxes:
                # A hand is visible -> require the object to actually overlap
                # it (filters out a phone sitting on the dashboard/seat).
                in_hand = any(_boxes_overlap(box, hb, PHONE_HAND_MARGIN) for hb in hand_boxes)
            else:
                # No hand detected at all -> can't verify overlap. This is
                # common when fingers curl around and hide the phone from
                # HandLandmarker, so don't punish a confident phone read
                # just because the hand itself couldn't be found.
                in_hand = True

            # Path 1: direct "cell phone" label match.
            is_phone_label = cat.category_name == "cell phone"
            near_face = _overlap_fraction(box, face_box) >= FACE_OVERLAP_FRACTION
            required_score = PHONE_SCORE_NEAR_FACE if near_face else PHONE_SCORE_AWAY_FACE
            direct_match = is_phone_label and cat.score >= required_score

            # Path 2: wrong label, but phone-shaped and held -> fallback.
            shape_match = False
            if not direct_match and in_hand and cat.category_name in CONFUSABLE_PHONE_LABELS \
                    and cat.score >= PHONE_SHAPE_SCORE_MIN \
                    and _is_phone_shaped(box, frame_w, frame_h):
                shape_match = True

            confident_phone = direct_match or shape_match

            if confident_phone and in_hand:
                phone_boxes.append((x1, y1, x2, y2, cat.category_name, cat.score))
            elif cat.score >= OBJECT_SCORE_THRESHOLD:
                other_boxes.append((x1, y1, x2, y2, cat.category_name, cat.score))

            if PHONE_DEBUG:
                print(f"[obj-debug] {cat.category_name} score={cat.score:.2f} "
                      f"box={box} near_face={near_face} req={required_score:.2f} "
                      f"in_hand={in_hand} direct={direct_match} shape={shape_match} "
                      f"-> {'PHONE' if confident_phone and in_hand else 'other'}")

    # [FIX] Path 3: model proposed nothing usable -> contour check near hands
    if not phone_boxes:
        try:
            phone_boxes = _path3_contour_phone(mp_image.numpy_view(), hand_boxes, face_box)
        except Exception as e:
            if PHONE_DEBUG:
                print("[path3] error:", e)

    # [FIX] short hold so a single missed frame doesn't reset confirmation
    if phone_boxes:
        _PHONE_HOLD["boxes"], _PHONE_HOLD["left"] = phone_boxes, PHONE_HOLD_FRAMES
    elif _PHONE_HOLD["left"] > 0:
        _PHONE_HOLD["left"] -= 1
        phone_boxes = _PHONE_HOLD["boxes"]

    return phone_boxes, other_boxes


# ─── Drawing Helpers ──────────────────────────────────────────────────────────

def highlight_phone(frame, x1, y1, x2, y2, score=None, raw_label=None):
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return
    frame[y1:y2, x1:x2] = cv2.convertScaleAbs(region, alpha=1.7, beta=70)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), -1)
    cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 3)
    c = 18
    for cx, cy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame, (cx,cy), (cx+dx*c, cy), (0,255,255), 3)
        cv2.line(frame, (cx,cy), (cx, cy+dy*c), (0,255,255), 3)
    # When the match came via the shape/context fallback (model's raw label
    # wasn't literally "cell phone"), show that label too -- makes it obvious
    # during a demo *why* the alert fired, instead of just "trust me".
    if raw_label and raw_label != "cell phone":
        label = f"PHONE DETECTED (as '{raw_label}', {score:.0%})" if score is not None else "PHONE DETECTED"
    else:
        label = "PHONE DETECTED" if score is None else f"PHONE DETECTED ({score:.0%})"
    cv2.putText(frame, label, (x1, max(y1-10,15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0,220,255), 2)


def draw_labeled_object(frame, x1, y1, x2, y2, name, score):
    """Neutral box + label for a detected object that is NOT a confirmed
    in-hand phone — shows its real class name so it's never confused with
    a phone alert (e.g. 'bottle', 'cup', 'remote', or even 'cell phone'
    when it's visible but not actually being held)."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 2)
    label = f"{name.upper()} ({score:.0%})"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, max(y1-th-10,0)), (x1+tw+8, max(y1-2,th+8)), (60,60,60), -1)
    cv2.putText(frame, label, (x1+4, max(y1-6,th+3)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1)


def draw_hand_skeleton(frame, hand_result, h, w):
    if not hand_result or not hand_result.hand_landmarks:
        return
    CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS if hasattr(mp, 'solutions') else [
        (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)
    ]
    for hand_lms in hand_result.hand_landmarks:
        pts = [(int(lm.x*w), int(lm.y*h)) for lm in hand_lms]
        for (a, b) in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0,180,0), 1)
        for pt in pts:
            cv2.circle(frame, pt, 3, (0,220,80), -1)


# ─── Audio Alerts ─────────────────────────────────────────────────────────────

_last_beep_time = {"drowsy": 0.0, "phone": 0.0}

# (frequency_hz, duration_ms) per alert kind — distinct tones so you can tell
# them apart by ear without looking at the screen.
_ALERT_TONES = {
    "drowsy": (900, 450),   # lower, longer tone
    "phone":  (1800, 200),  # higher, shorter tone
}


def _beep_worker(freq, duration_ms):
    if _HAS_WINSOUND:
        winsound.Beep(freq, duration_ms)
    else:
        # Cross-platform fallback: terminal bell (audible on most terminals).
        print("\a", end="", flush=True)


def play_alert(kind):
    """
    Plays a short beep for the given alert kind ('drowsy' or 'phone').
    Non-blocking (runs on a background thread) so the video loop never
    stalls, and rate-limited per kind via AUDIO_COOLDOWN so it beeps
    steadily rather than every single frame while the condition holds.
    """
    if not AUDIO_ENABLED:
        return
    now = time.time()
    if now - _last_beep_time.get(kind, 0.0) < AUDIO_COOLDOWN:
        return
    _last_beep_time[kind] = now

    freq, duration = _ALERT_TONES.get(kind, (1200, 250))
    threading.Thread(target=_beep_worker, args=(freq, duration), daemon=True).start()


# A fresh pyttsx3 engine is created for EVERY utterance (not one reused
# engine). This is deliberate: on Windows/SAPI5, a single pyttsx3 engine
# reliably speaks its FIRST utterance but often goes silent on the SECOND
# call to runAndWait() — which is exactly why "right" spoke once and then
# nothing else did afterwards, regardless of which word came next.
#
# A background queue + single worker thread is still used so presses are
# processed one at a time in order (rather than racing each other), but the
# engine itself is rebuilt fresh for each item, with COM properly
# initialized/uninitialized on that thread each time.
_speech_queue = queue.Queue()
_speech_thread = None


def _speak_one(text):
    com_ready = False
    if os.name == "nt":
        try:
            import pythoncom
            pythoncom.CoInitialize()
            com_ready = True
        except Exception as e:
            print(f"[voice] COM init warning: {e}")
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        try:
            engine.stop()
        except Exception:
            pass
    except Exception as e:
        print(f"[voice] TTS error: {e}")
    finally:
        if com_ready:
            try:
                import pythoncom
                pythoncom.CoUninitialize()
            except Exception:
                pass


def _speech_worker_loop():
    while True:
        text = _speech_queue.get()
        if text is None:   # sentinel to stop the thread (not currently used, but available)
            break
        _speak_one(text)


def _ensure_speech_thread():
    global _speech_thread
    if _speech_thread is None:
        _speech_thread = threading.Thread(target=_speech_worker_loop, daemon=True)
        _speech_thread.start()


def speak_once(text):
    """
    Queues `text` to be spoken by the background speech thread (non-blocking
    — the video loop never waits on it). Called once per event (e.g. every
    time a turn indicator is switched on) — a fresh engine is used for each
    utterance so it reliably speaks every time, not just the first time.
    """
    if not _HAS_TTS:
        print(f"[voice] (TTS unavailable) would say: \"{text}\"")
        return
    _ensure_speech_thread()
    _speech_queue.put(text)



def draw_drowsy_alert(frame):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (w,h), (0,0,140), -1)
    cv2.addWeighted(ov, 0.18, frame, 0.82, 0, frame)
    if int(time.time()*2) % 2 == 0:
        cv2.putText(frame, "!  DROWSINESS ALERT  !",
                    (w//2-240, h//2),
                    cv2.FONT_HERSHEY_DUPLEX, 1.15, (0,0,255), 3)
    cv2.rectangle(frame, (0,0), (w-1,h-1), (0,0,255), 5)


def draw_phone_border(frame):
    h, w = frame.shape[:2]
    if int(time.time()*3) % 2 == 0:
        cv2.rectangle(frame, (0,0), (w-1,h-1), (0,255,255), 6)


def draw_signal_warning(frame, direction):
    """direction: 'LEFT' or 'RIGHT' — the turn the driver appears to be making
    without the matching indicator on."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0,0), (w-1,h-1), (0,140,255), 4)
    if int(time.time()*4) % 2 == 0:
        text = f"!  SIGNAL YOUR {direction} TURN  !"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.85, 3)
        x = max(10, w//2 - tw//2)
        y = h - 30
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, (0,140,255), 3)


def draw_indicator_arrow(frame, cx, cy, size, pointing_left, color):
    """Draw a filled chevron/arrow pointing left or right."""
    if pointing_left:
        pts = np.array([
            [cx + size, cy - size],
            [cx - size, cy],
            [cx + size, cy + size],
        ], dtype=np.int32)
    else:
        pts = np.array([
            [cx - size, cy - size],
            [cx + size, cy],
            [cx - size, cy + size],
        ], dtype=np.int32)
    cv2.fillPoly(frame, [pts], color)
    cv2.polylines(frame, [pts], True, (0,0,0), 1)


def draw_turn_indicators(frame, left_on, right_on):
    """Blinking LEFT / RIGHT turn-signal indicators, clearly visible to the driver.

    NOTE: The webcam feed is NOT flipped, so it shows a true (non-mirrored)
    image of the driver — the opposite of what a mirror/selfie view would show.
    To match the driver's real left/right, the LEFT indicator is drawn on the
    screen's RIGHT side, and the RIGHT indicator is drawn on the screen's LEFT
    side (swapped on purpose). If you later add cv2.flip(frame, 1) to make the
    preview an actual mirror image, swap these back to their natural sides.
    """
    h, w = frame.shape[:2]
    blink_on = int(time.time() * 2.5) % 2 == 0  # ~1.25 Hz blink
    amber = (0, 190, 255)
    dim   = (0, 70, 90)

    # LEFT indicator -> drawn on screen's RIGHT side (mirror-correct)
    if left_on:
        col = amber if blink_on else dim
        y = 100
        for i, x in enumerate([w-130, w-90, w-50]):
            draw_indicator_arrow(frame, x, y, 18, pointing_left=False, color=col)
        if blink_on:
            (tw, th), _ = cv2.getTextSize("LEFT", cv2.FONT_HERSHEY_DUPLEX, 0.85, 2)
            cv2.putText(frame, "LEFT", (w-40-tw, 140),
                        cv2.FONT_HERSHEY_DUPLEX, 0.85, amber, 2)

    # RIGHT indicator -> drawn on screen's LEFT side (mirror-correct)
    if right_on:
        col = amber if blink_on else dim
        y = 100
        for i, x in enumerate([50, 90, 130]):
            draw_indicator_arrow(frame, x, y, 18, pointing_left=True, color=col)
        if blink_on:
            cv2.putText(frame, "RIGHT", (40, 140),
                        cv2.FONT_HERSHEY_DUPLEX, 0.85, amber, 2)


def draw_hud(frame, drowsy, phone, ear, closed_seconds, left_on, right_on):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (w,60), (12,12,12), -1)
    cv2.addWeighted(ov, 0.72, frame, 0.28, 0, frame)

    cv2.putText(frame, f"EAR: {ear:.2f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,200), 2)

    bar_fill = min(int((closed_seconds/EAR_CLOSED_SECONDS)*180), 180)
    cv2.rectangle(frame, (10,40), (190,54), (40,40,40), -1)
    if bar_fill > 0:
        col = (0,60,255) if closed_seconds > EAR_CLOSED_SECONDS*0.6 else (0,180,80)
        cv2.rectangle(frame, (10,40), (10+bar_fill,54), col, -1)

    d_col  = (0,0,255) if drowsy else (0,210,80)
    cv2.putText(frame, f"Driver: {'DROWSY!' if drowsy else 'AWAKE'}",
                (w//2-95, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.78, d_col, 2)

    p_col  = (0,200,255) if phone else (0,210,80)
    cv2.putText(frame, "PHONE IN USE" if phone else "No Phone",
                (w-220,42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, p_col, 2)

    # Small always-on indicator status text, top-center-ish, so driver
    # can see current turn-signal state even between blinks.
    if left_on or right_on:
        label = "IND: LEFT" if left_on else "IND: RIGHT"
        cv2.putText(frame, label, (w//2-95, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,190,255), 1)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Download models if needed
    download_model(FACE_MODEL_URL, FACE_MODEL_PATH)
    download_model(HAND_MODEL_URL, HAND_MODEL_PATH)
    download_model(OBJECT_MODEL_URL, OBJECT_MODEL_PATH)

    # Build FaceLandmarker
    face_opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    face_landmarker = mp_vision.FaceLandmarker.create_from_options(face_opts)

    # Build HandLandmarker
    hand_opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    hand_landmarker = mp_vision.HandLandmarker.create_from_options(hand_opts)

    # Build ObjectDetector — all COCO classes (not just "cell phone"), so
    # bottles/cups/remotes/etc. get their own real label instead of being
    # forced into a phone match. Filtering to "confirmed phone" happens in
    # detect_objects().
    object_opts = mp_vision.ObjectDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=OBJECT_MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        score_threshold=OBJECT_SCORE_THRESHOLD,
        max_results=8
    )
    object_detector = mp_vision.ObjectDetector.create_from_options(object_opts)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    eyes_closed_start = None   # timestamp when eyes first went below EAR_THRESHOLD
    drowsy      = False
    alert_ts    = 0.0

    phone_confirm_counter = 0  # consecutive frames a phone-in-hand was seen

    # Turn-indicator state (toggled by 'a' / 'd' key presses)
    left_indicator  = False
    right_indicator = False

    # Head-turn detection state (mirror-corrected, see estimate_yaw_ratio)
    yaw_counter_left  = 0   # sustained frames of screen-right skew -> driver's actual LEFT turn
    yaw_counter_right = 0   # sustained frames of screen-left skew  -> driver's actual RIGHT turn

    print("=" * 52)
    print("  Driver Monitoring System  –  MediaPipe Tasks API")
    print("=" * 52)
    print("  Press 'a' to toggle LEFT turn indicator")
    print("  Press 'd' to toggle RIGHT turn indicator")
    print("  Press 'q' to quit")
    print("=" * 52)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame     = cv2.resize(frame, (800, 600))
        h, w      = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # ── Face / Drowsiness ─────────────────────────────────────────────────
        face_result = face_landmarker.detect_for_video(mp_image, timestamp)
        ear = 1.0
        face_box = None   # bounding box of the face, used to spot "near face"
                           # false-positive phone detections (finger/hand gestures)

        if face_result.face_landmarks:
            lms = face_result.face_landmarks[0]
            face_xs = [pt.x * w for pt in lms]
            face_ys = [pt.y * h for pt in lms]
            face_box = (min(face_xs), min(face_ys), max(face_xs), max(face_ys))

            left_ear  = eye_aspect_ratio(lms, LEFT_EYE,  w, h)
            right_ear = eye_aspect_ratio(lms, RIGHT_EYE, w, h)
            ear       = (left_ear + right_ear) / 2.0

            eye_col = (0,60,255) if ear < EAR_THRESHOLD else (0,220,80)
            draw_eye_outline(frame, lms, LEFT_EYE,  w, h, eye_col)
            draw_eye_outline(frame, lms, RIGHT_EYE, w, h, eye_col)

            if ear < EAR_THRESHOLD:
                if eyes_closed_start is None:
                    eyes_closed_start = time.time()
                elif time.time() - eyes_closed_start >= EAR_CLOSED_SECONDS:
                    drowsy   = True
                    alert_ts = time.time()
            else:
                eyes_closed_start = None
                if time.time() - alert_ts > 2.0:
                    drowsy = False

            # ── Head yaw / turn detection (mirror-corrected) ───────────────────
            yaw_ratio = estimate_yaw_ratio(lms)
            if yaw_ratio > YAW_THRESHOLD:        # screen-right skew -> actual LEFT turn
                yaw_counter_left  += 1
                yaw_counter_right  = 0
            elif yaw_ratio < -YAW_THRESHOLD:     # screen-left skew  -> actual RIGHT turn
                yaw_counter_right += 1
                yaw_counter_left   = 0
            else:
                yaw_counter_left  = 0
                yaw_counter_right = 0
        else:
            eyes_closed_start = None
            yaw_counter_left  = 0
            yaw_counter_right = 0

        turning_left_now  = yaw_counter_left  >= YAW_CONSEC_FRAMES
        turning_right_now = yaw_counter_right >= YAW_CONSEC_FRAMES
        unsignaled_left    = turning_left_now  and not left_indicator
        unsignaled_right   = turning_right_now and not right_indicator

        # ── Hands / Phone / Other Objects ───────────────────────────────────────
        hand_result  = hand_landmarker.detect_for_video(mp_image, timestamp)
        hand_boxes   = get_hand_boxes(hand_result, w, h)
        phone_boxes, other_boxes = detect_objects(object_detector, mp_image, timestamp, hand_boxes, face_box,
                                                   frame_w=w, frame_h=h)

        # Require the phone-in-hand detection to be sustained across several
        # consecutive frames before counting it — a single-frame "cell phone"
        # read on an empty hand is usually a momentary misclassification.
        if phone_boxes:
            phone_confirm_counter += 1
        else:
            phone_confirm_counter = 0
        phone_detected = phone_confirm_counter >= PHONE_CONSEC_FRAMES

        draw_hand_skeleton(frame, hand_result, h, w)

        for (x1, y1, x2, y2, name, score) in other_boxes:
            draw_labeled_object(frame, x1, y1, x2, y2, name, score)
        if phone_detected:
            for (x1, y1, x2, y2, name, score) in phone_boxes:
                highlight_phone(frame, x1, y1, x2, y2, score, raw_label=name)
        else:
            # Not yet confirmed — show as a plain labeled object, not an alert.
            for (x1, y1, x2, y2, name, score) in phone_boxes:
                draw_labeled_object(frame, x1, y1, x2, y2, name, score)

        # ── Overlays ──────────────────────────────────────────────────────────
        if drowsy:
            draw_drowsy_alert(frame)
            play_alert("drowsy")
        if phone_detected:
            draw_phone_border(frame)
            play_alert("phone")

        draw_turn_indicators(frame, left_indicator, right_indicator)
        closed_seconds = (time.time() - eyes_closed_start) if eyes_closed_start else 0.0
        draw_hud(frame, drowsy, phone_detected, ear, closed_seconds,
                 left_indicator, right_indicator)

        if unsignaled_left:
            draw_signal_warning(frame, "LEFT")
        elif unsignaled_right:
            draw_signal_warning(frame, "RIGHT")

        cv2.imshow("Driver Monitoring System", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            left_indicator  = not left_indicator
            right_indicator = False   # indicators are mutually exclusive
            if left_indicator:        # just turned ON -> remind once
                speak_once("Check your left mirror")
        elif key == ord('d'):
            right_indicator = not right_indicator
            left_indicator  = False
            if right_indicator:       # just turned ON -> remind once
                speak_once("Check your right mirror")

    cap.release()
    cv2.destroyAllWindows()
    face_landmarker.close()
    hand_landmarker.close()
    object_detector.close()
    print("[INFO] Monitoring stopped.")


if __name__ == "__main__":
    main()