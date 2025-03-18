import os
import cv2
import numpy as np
import shutil
import logging
import contextlib
from pathlib import Path
import mediapipe as mp
from tqdm import tqdm

# Configure logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MEDIAPIPE_DISABLE_LOGGING'] = '1'
logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)


@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)


mp_pose = mp.solutions.pose

BODY_PARTS_TO_CHECK = [
    mp_pose.PoseLandmark.NOSE,
    mp_pose.PoseLandmark.LEFT_EYE_INNER,
    mp_pose.PoseLandmark.RIGHT_EYE_INNER,
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP
]

TORSO_PARTS = [
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP
]


class ProgressTracker:
    def __init__(self, total_files):
        self.progress = tqdm(total=total_files, desc="Processing files")

    def update(self, filename, status):
        self.progress.set_description(f"Processing {filename}: {status}")
        self.progress.update(1)

    def close(self):
        self.progress.close()


def get_pose_landmarks(image_path):
    with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
        image = cv2.imread(image_path)
        if image is None:
            return None
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            return None

        landmarks = []
        for body_part in BODY_PARTS_TO_CHECK:
            landmark = results.pose_landmarks.landmark[body_part]
            landmarks.append((landmark.x, landmark.y))
        return np.array(landmarks)


def get_body_vectors_and_weights(landmarks):
    vectors = []
    weights = []

    for idx, (x, y) in enumerate(landmarks):
        if x == 0 and y == 0:
            continue
        vectors.append([x, y])
        weights.append(2.0 if BODY_PARTS_TO_CHECK[idx] in TORSO_PARTS else 1.0)

    return np.array(vectors), np.array(weights)


def are_poses_similar(pose1, pose2, threshold=0.1):
    if pose1 is None or pose2 is None:
        return False

    vectors1, weights1 = get_body_vectors_and_weights(pose1)
    vectors2, weights2 = get_body_vectors_and_weights(pose2)

    if len(vectors1) != len(vectors2) or len(vectors1) == 0:
        return False

    distances = np.linalg.norm(vectors1 - vectors2, axis=1)
    similarity_score = np.mean(distances * (weights1 + weights2) / 2)
    return similarity_score < threshold


def main():
    input_dir = 'sources'
    uniposes_dir = 'uniposes'
    pose_check_dir = 'pose_check'
    trash_dir = 'trash'

    os.makedirs(uniposes_dir, exist_ok=True)
    os.makedirs(pose_check_dir, exist_ok=True)
    os.makedirs(trash_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    progress = ProgressTracker(len(files))
    current_idx = len(os.listdir(uniposes_dir)) + 1
    unique_poses = []

    for filename in files:
        filepath = os.path.join(input_dir, filename)
        try:
            landmarks = get_pose_landmarks(filepath)

            if landmarks is None:
                dest = os.path.join(pose_check_dir, filename)
                shutil.move(filepath, dest)
                progress.update(filename, 'no_pose')
                continue

            is_duplicate = False
            for ref_landmarks, pose_idx in unique_poses:
                if are_poses_similar(landmarks, ref_landmarks):
                    trash_subdir = os.path.join(trash_dir, str(pose_idx))
                    os.makedirs(trash_subdir, exist_ok=True)
                    orig_path = os.path.join(uniposes_dir, f"{pose_idx}.jpg")
                    shutil.copy2(orig_path, os.path.join(trash_subdir, "original.jpg"))

                    dest = os.path.join(trash_subdir, filename)
                    shutil.move(filepath, dest)
                    progress.update(filename, 'duplicate')
                    is_duplicate = True
                    break

            if not is_duplicate:
                ext = Path(filename).suffix
                unipose_path = os.path.join(uniposes_dir, f"{current_idx}{ext}")
                shutil.move(filepath, unipose_path)
                unique_poses.append((landmarks, current_idx))
                progress.update(filename, 'unique')
                current_idx += 1

        except Exception as e:
            progress.update(filename, 'error')

    # Cleanup empty directories
    for root, dirs, _ in os.walk(trash_dir, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)

    progress.close()


if __name__ == "__main__":
    main()