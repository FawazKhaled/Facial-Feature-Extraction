import argparse
from collections import Counter, deque
from pathlib import Path
import sqlite3

import cv2

from lib.dataset_reader import DEFAULT_CASIA_DIRECTORY
from lib.face_id import MATCH_DISTANCE, FaceWorker, draw_banner, draw_face, load_face_codes, load_itq, match_face_code, open_camera, remove_duplicate_people


WINDOW_NAME = "Face ID"
DEFAULT_MAX_DISTANCE = MATCH_DISTANCE
LABEL_HISTORY = 5


# each face keeps its last few results, and the label shown is the most common one so a single bad frame can't flip it
def update_tracks(tracks, faces, names, codes, max_distance):
    updated_tracks = []
    for face in faces:
        name, distance = match_face_code(face["code"], names, codes)
        label = name if distance <= max_distance else "Unknown"
        left, top, right, bottom = face["bbox"]
        center = ((left + right) / 2, (top + bottom) / 2)
        unmatched = [track for track in tracks if not any(track is used for used in updated_tracks)]
        closest = min(unmatched, key=lambda track: (track["center"][0] - center[0]) ** 2 + (track["center"][1] - center[1]) ** 2, default=None)
        if closest is None or (closest["center"][0] - center[0]) ** 2 + (closest["center"][1] - center[1]) ** 2 > (right - left) ** 2:
            closest = {"history": deque(maxlen=LABEL_HISTORY)}
        closest["history"].append(label)
        closest.update({"bbox": face["bbox"], "center": center, "distance": distance, "label": Counter(closest["history"]).most_common(1)[0][0]})
        updated_tracks.append(closest)
    return updated_tracks


def run_recognition(camera, worker, names, codes, max_distance):
    people_count = len(set(names))
    tracks = []
    last_result_number = 0
    while True:
        ok, frame = camera.read()
        if not ok:
            raise RuntimeError("Could not read a frame from the camera.")
        frame = cv2.flip(frame, 1)
        worker.submit(frame)
        if worker.error is not None:
            raise RuntimeError(f"Face processing failed: {worker.error}")

        result_number, faces = worker.latest()
        if result_number != last_result_number:
            tracks = update_tracks(tracks, faces, names, codes, max_distance)
            last_result_number = result_number
        for track in tracks:
            color = (0, 0, 255) if track["label"] == "Unknown" else (0, 200, 0)
            draw_face(frame, track["bbox"], color, f"{track['label']} ({track['distance']})")

        draw_banner(frame, [f"{people_count} enrolled, match if distance <= {max_distance}. Press Q to quit."])
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")) or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            return


def main():
    parser = argparse.ArgumentParser(description="Recognize enrolled faces live from the webcam using binary codes.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--max-distance", type=int, default=DEFAULT_MAX_DISTANCE, help=f"Largest Hamming distance that still counts as a match (default: {DEFAULT_MAX_DISTANCE})")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_CASIA_DIRECTORY, help="CASIA-WebFace folder used to find the data folder")
    parser.add_argument("--database", type=Path, help="Face ID database of binary codes (default: data/face_id.sqlite3)")
    parser.add_argument("--itq", type=Path, help="ITQ settings file made during enrollment (default: data/itq_256.npz)")
    args = parser.parse_args()

    dataset_directory = args.dataset.expanduser().resolve()
    database_path = args.database or dataset_directory.parent / "face_id.sqlite3"
    itq_path = args.itq or dataset_directory.parent / "itq_256.npz"
    camera = None
    worker = None

    try:
        if args.max_distance < 0:
            raise ValueError("max-distance cannot be negative.")
        mean, projection, fingerprint = load_itq(itq_path)
        for removed_name, original_name in remove_duplicate_people(database_path, fingerprint):
            print(f"Removed {removed_name}: same person as {original_name}, who registered first.")
        names, codes = load_face_codes(database_path, fingerprint)
        worker = FaceWorker(mean, projection, with_pose=False)
        worker.start()
        camera = open_camera(args.camera)
        run_recognition(camera, worker, names, codes, args.max_distance)
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as error:
        parser.error(str(error))
    finally:
        if camera is not None:
            camera.release()
        if worker is not None:
            worker.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
