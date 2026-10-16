import argparse
from pathlib import Path
import sqlite3
import time

import cv2
import numpy as np

from lib.dataset_reader import DEFAULT_CASIA_DIRECTORY
from lib.face_id import FaceWorker, draw_banner, draw_face, load_or_fit_itq, open_camera, remove_duplicate_people, store_face_codes


WINDOW_NAME = "Face ID setup"
POSES = [
    ("center", "Look straight at the camera"),
    ("left", "Slowly turn your head to the left"),
    ("right", "Slowly turn your head to the right"),
    ("up", "Tilt your head up a little"),
    ("down", "Tilt your head down a little"),
]
MAX_CENTER_YAW = 12
MIN_TURN = 15
MIN_TILT = 10
MIN_FACE_WIDTH = 100
MIN_DETECTION_SCORE = 0.7
CAPTURE_GAP = 0.4


# turns are measured from the straight-ahead pose, and each second turn must go the opposite way of the first
def pose_direction(step, pose, center_pose, turn_signs):
    if step == "center":
        return 1 if abs(pose[1]) <= MAX_CENTER_YAW else 0

    axis, minimum, first_step = (1, MIN_TURN, "left") if step in ("left", "right") else (0, MIN_TILT, "up")
    offset = pose[axis] - center_pose[axis]
    if abs(offset) < minimum:
        return 0
    direction = int(np.sign(offset))
    expected = turn_signs.get(step) or (-turn_signs[first_step] if step != first_step else None)
    return direction if expected is None or direction == expected else 0


def capture_samples(camera, worker, name, samples_per_pose):
    samples = []
    center_poses = []
    turn_signs = {}
    step_index = 0
    last_capture_number = 0
    last_capture_time = 0.0
    total_samples = len(POSES) * samples_per_pose

    while step_index < len(POSES):
        ok, frame = camera.read()
        if not ok:
            raise RuntimeError("Could not read a frame from the camera.")
        frame = cv2.flip(frame, 1)
        worker.submit(frame)
        if worker.error is not None:
            raise RuntimeError(f"Face processing failed: {worker.error}")

        result_number, faces = worker.latest()
        step, instruction = POSES[step_index]
        center_pose = np.mean(center_poses, axis=0) if center_poses else None
        status = ""
        if not faces:
            status = "No face found"
        elif len(faces) > 1:
            status = "Only one face should be in view"
        else:
            face = faces[0]
            left, _, right, _ = face["bbox"]
            direction = pose_direction(step, face["pose"], center_pose, turn_signs)
            if right - left < MIN_FACE_WIDTH:
                status = "Move closer to the camera"
            elif face["det_score"] < MIN_DETECTION_SCORE:
                status = "Face the camera in better light"
            elif direction == 0:
                status = "Keep going" if step != "center" else "Face the camera"
            elif result_number > last_capture_number and time.time() - last_capture_time >= CAPTURE_GAP:
                samples.append((step, face["code"]))
                if step == "center":
                    center_poses.append(face["pose"])
                else:
                    turn_signs.setdefault(step, direction)
                last_capture_number = result_number
                last_capture_time = time.time()
            draw_face(frame, face["bbox"], (0, 200, 0) if direction != 0 and not status else (0, 200, 255))

        step_samples = sum(pose == step for pose, _ in samples)
        draw_banner(frame, [f"Setting up Face ID for {name}", f"Step {step_index + 1}/{len(POSES)}: {instruction}",
                            f"{step_samples}/{samples_per_pose} captured  {status}"])
        bar_width = int(frame.shape[1] * len(samples) / total_samples)
        cv2.rectangle(frame, (0, frame.shape[0] - 10), (bar_width, frame.shape[0]), (0, 200, 0), -1)
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")) or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            return None
        if step_samples >= samples_per_pose:
            step_index += 1

    return samples


def main():
    parser = argparse.ArgumentParser(description="Set up Face ID for one person with the webcam. Only binary codes are saved.")
    parser.add_argument("--name", required=True, help="Name to show when this person is recognized")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--samples-per-pose", type=int, default=3, help="Codes captured for each head pose (default: 3)")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_CASIA_DIRECTORY, help="CASIA-WebFace folder used to find the data folder and embedding cache")
    parser.add_argument("--cache", type=Path, help="ArcFace embedding cache used to fit ITQ (default: data/arcface_embeddings.sqlite3)")
    parser.add_argument("--database", type=Path, help="Face ID database of binary codes (default: data/face_id.sqlite3)")
    parser.add_argument("--itq", type=Path, help="ITQ settings file, fitted on first use (default: data/itq_256.npz)")
    parser.add_argument("--bits", type=int, default=256, help="Code length used when fitting ITQ (default: 256)")
    args = parser.parse_args()

    name = args.name.strip()
    dataset_directory = args.dataset.expanduser().resolve()
    cache_path = args.cache or dataset_directory.parent / "arcface_embeddings.sqlite3"
    database_path = args.database or dataset_directory.parent / "face_id.sqlite3"
    itq_path = args.itq or dataset_directory.parent / f"itq_{args.bits}.npz"
    camera = None
    worker = None

    try:
        if not name:
            raise ValueError("name cannot be empty.")
        if args.samples_per_pose < 1 or args.bits < 1:
            raise ValueError("samples-per-pose and bits must be at least 1.")
        mean, projection, fingerprint = load_or_fit_itq(itq_path, cache_path, dataset_directory, args.bits)
        worker = FaceWorker(mean, projection)
        worker.start()
        camera = open_camera(args.camera)
        samples = capture_samples(camera, worker, name, args.samples_per_pose)
        if samples is None:
            print("Setup cancelled. Nothing was saved.")
            return
        replaced = store_face_codes(database_path, name, samples, fingerprint)
        removed = remove_duplicate_people(database_path, fingerprint)
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as error:
        parser.error(str(error))
    finally:
        if camera is not None:
            camera.release()
        if worker is not None:
            worker.stop()
        cv2.destroyAllWindows()

    duplicate_of = dict(removed).get(name)
    if duplicate_of is not None:
        print(f"This face is already registered as {duplicate_of}, so the new registration for {name} was removed.")
    else:
        print(f"{'Replaced' if replaced else 'Saved'} Face ID for {name}: {len(samples)} codes of {projection.shape[1]} bits in {database_path}.")
    for removed_name, original_name in removed:
        if removed_name != name:
            print(f"Removed {removed_name}: same person as {original_name}, who registered first.")


if __name__ == "__main__":
    main()
