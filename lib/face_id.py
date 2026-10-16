import hashlib
from pathlib import Path
import sqlite3
import sys
import threading
import time

import cv2
import numpy as np

from lib.ArcFace import load_arcface_model
from lib.caching import dataset_signature, load_embeddings
from lib.hashing import encode_itq, train_itq


FACE_THREADS = 3
MATCH_DISTANCE = 90


def open_camera(camera_index):
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}.")
    return camera


# the ITQ settings come from the cached CASIA embeddings, never from enrolled faces
def load_or_fit_itq(itq_path, cache_path, dataset_directory, bit_length=256):
    itq_path = Path(itq_path)
    if not itq_path.is_file():
        records = load_embeddings(cache_path, dataset_signature(dataset_directory))
        if not records:
            raise ValueError("No cached embeddings found to fit ITQ. Run face_comparison.py first.")
        mean, projection = train_itq(np.stack([embedding for _, _, embedding in records]), bit_length)
        itq_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(itq_path, mean=mean, projection=projection)
    return load_itq(itq_path)


def load_itq(itq_path):
    itq_path = Path(itq_path)
    if not itq_path.is_file():
        raise FileNotFoundError(f"ITQ settings not found: {itq_path}. Enroll someone with enroll_face.py first.")
    with np.load(itq_path) as values:
        mean = values["mean"]
        projection = values["projection"]
    fingerprint = hashlib.sha256(itq_path.read_bytes()).hexdigest()
    return mean, projection, fingerprint


def initialize_face_database(database_path):
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS face_codes (
            name TEXT NOT NULL,
            sample INTEGER NOT NULL,
            pose TEXT NOT NULL,
            itq_fingerprint TEXT NOT NULL,
            code BLOB NOT NULL,
            registered_at REAL NOT NULL DEFAULT 0,
            PRIMARY KEY (name, sample)
        )
        """
    )
    # databases made before registered_at existed get 0, and their insertion order (rowid) still gives who came first
    columns = [row[1] for row in connection.execute("PRAGMA table_info(face_codes)")]
    if "registered_at" not in columns:
        connection.execute("ALTER TABLE face_codes ADD COLUMN registered_at REAL NOT NULL DEFAULT 0")
    connection.commit()
    return connection


# only binary codes are saved; enrolling the same name again replaces the old codes but keeps the original registration time
def store_face_codes(database_path, name, samples, fingerprint):
    connection = initialize_face_database(database_path)
    try:
        previous_time = connection.execute("SELECT MIN(registered_at) FROM face_codes WHERE name = ?", (name,)).fetchone()[0]
        replaced = previous_time is not None
        registered_at = previous_time if replaced else time.time()
        connection.execute("DELETE FROM face_codes WHERE name = ?", (name,))
        connection.executemany(
            "INSERT INTO face_codes (name, sample, pose, itq_fingerprint, code, registered_at) VALUES (?, ?, ?, ?, ?, ?)",
            [(name, sample, pose, fingerprint, code, registered_at) for sample, (pose, code) in enumerate(samples, start=1)],
        )
        connection.commit()
    finally:
        connection.close()
    return replaced


def load_face_codes(database_path, fingerprint):
    if not Path(database_path).is_file():
        raise FileNotFoundError(f"Face ID database not found: {database_path}. Enroll someone with enroll_face.py first.")
    connection = sqlite3.connect(database_path)
    try:
        rows = connection.execute("SELECT name, itq_fingerprint, code FROM face_codes ORDER BY name, sample").fetchall()
    finally:
        connection.close()

    if not rows:
        raise ValueError("No one is enrolled yet. Run enroll_face.py first.")
    if any(row_fingerprint != fingerprint for _, row_fingerprint, _ in rows):
        raise ValueError("Some enrolled codes were made with different ITQ settings. Enroll those people again.")
    names = np.asarray([name for name, _, _ in rows])
    codes = np.frombuffer(b"".join(code for _, _, code in rows), dtype=np.uint8).reshape(len(rows), -1)
    return names, codes


# a later registration is the same person when most of its codes match one of an earlier person's codes
def is_same_person(later_codes, earlier_codes, max_distance=MATCH_DISTANCE):
    distances = np.unpackbits(np.bitwise_xor(later_codes[:, None, :], earlier_codes[None, :, :]), axis=2).sum(axis=2)
    return bool(np.mean(distances.min(axis=1) <= max_distance) >= 0.5)


# goes through people in the order they registered and removes anyone who matches someone registered before them
def remove_duplicate_people(database_path, fingerprint, max_distance=MATCH_DISTANCE):
    if not Path(database_path).is_file():
        return []
    connection = initialize_face_database(database_path)
    try:
        rows = connection.execute("SELECT name, registered_at, rowid, code FROM face_codes WHERE itq_fingerprint = ?", (fingerprint,)).fetchall()
        people = {}
        for name, registered_at, row_id, code in rows:
            person = people.setdefault(name, {"order": (registered_at, row_id), "codes": []})
            person["order"] = min(person["order"], (registered_at, row_id))
            person["codes"].append(code)

        kept = []
        removed = []
        for name, person in sorted(people.items(), key=lambda item: item[1]["order"]):
            codes = np.frombuffer(b"".join(person["codes"]), dtype=np.uint8).reshape(len(person["codes"]), -1)
            original = next((kept_name for kept_name, kept_codes in kept if is_same_person(codes, kept_codes, max_distance)), None)
            if original is None:
                kept.append((name, codes))
            else:
                removed.append((name, original))
        connection.executemany("DELETE FROM face_codes WHERE name = ?", [(name,) for name, _ in removed])
        connection.commit()
    finally:
        connection.close()
    return removed


def match_face_code(code, names, codes):
    distances = np.unpackbits(np.bitwise_xor(codes, np.frombuffer(code, dtype=np.uint8)), axis=1).sum(axis=1)
    best = int(np.argmin(distances))
    return str(names[best]), int(distances[best])


def draw_face(frame, bbox, color, label=None):
    left, top, right, bottom = [int(value) for value in bbox]
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    if label:
        cv2.putText(frame, label, (left, max(top - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_banner(frame, lines):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 12 + 26 * len(lines)), (0, 0, 0), -1)
    for line_number, line in enumerate(lines):
        cv2.putText(frame, line, (10, 30 + 26 * line_number), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


# runs face detection and hashing on the newest camera frame in the background so the preview stays smooth
class FaceWorker(threading.Thread):
    def __init__(self, mean, projection, with_pose=True):
        super().__init__(daemon=True)
        modules = ["detection", "recognition", "landmark_3d_68"] if with_pose else ["detection", "recognition"]
        self.model = load_arcface_model(allowed_modules=modules, threads=FACE_THREADS)
        self.mean = mean
        self.projection = projection
        self.condition = threading.Condition()
        self.frame = None
        self.frame_number = 0
        self.results = []
        self.result_number = 0
        self.error = None
        self.running = True

    def submit(self, frame):
        with self.condition:
            self.frame_number += 1
            self.frame = frame.copy()
            self.condition.notify()

    def latest(self):
        with self.condition:
            return self.result_number, self.results

    def stop(self):
        with self.condition:
            self.running = False
            self.condition.notify()

    def run(self):
        try:
            while True:
                with self.condition:
                    while self.running and self.frame is None:
                        self.condition.wait()
                    if not self.running:
                        return
                    frame, frame_number = self.frame, self.frame_number
                    self.frame = None

                results = []
                for face in self.model.get(frame):
                    embedding = np.asarray(face.normed_embedding, dtype=np.float32).reshape(1, -1)
                    code = encode_itq(embedding, self.mean, self.projection)[0]
                    results.append({"bbox": face.bbox, "pose": face.pose, "det_score": float(face.det_score), "code": code})

                with self.condition:
                    self.results = results
                    self.result_number = frame_number
        except Exception as error:
            self.error = error
