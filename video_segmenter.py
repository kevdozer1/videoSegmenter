import cv2
import numpy as np
from pydub import AudioSegment
import librosa
import sys
from video_player import main as video_player_main

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

DEBUG = False
SHOT_THRESHOLD = 15

# If adjacent frames have histogram similarity below this threshold, a new shot is defined (if using this method to detect shots)
SHOT_COLOR_THRESHOLD = 0.7

# Used when extracting keyframes. If a frame has histogram similarity below this threshold to all current keyframes, it is added as a new keyframe
KEYFRAME_SIMILARITY_THRESHOLD = 0.40

# Used when detecting potential scene boundaries. If the shot coherence between two adjacent shots is below this threshold, a new scene is defined
SCENE_DETECTION_THRESHOLD = 0.5
SCENE_DETECTION_VALLEY_THRESHOLD = 0.3

# How many shots back to look when calculating the shot coherence
# More shots back means greater chances to achieve a higher shot coherence
N_SHOTS_LOOK_BACK = 5

# The artificial BSC value to use for the first shot
FIRST_BSC = 0.75

# If adjacent potential scenes have keyframes that have histogram similarity >= this threshold, those scenes are merged
PSB_FILTERING_COLOR_THRESHOLD = 0.7

MIN_SHOT_LEN_SECS = 0.6
MIN_SCENE_LEN_SECS = 2.0

FPS = None

SHOTS = [0]
SCENES = [0]
SUBSHOTS = [0]
N_SHOTS = None
SHOTS_WITH_END_MARKER = None
SHOTS_SECONDS = [0.0]
SCENES_SECONDS = [0.0]
SUBSHOTS_SECONDS = [0.0]
FRAMES_HISTOGRAMS = []
ACCUMULATED_SHOTS_HISTOGRAMS = (
    []
)  # histogram of each shot, obtained by summing histograms of the frames
MIDPOINT_SHOTS_HISTOGRAMS = []  # histogram of the middle frame of each shot
SHOT_KEYFRAME_INDICES = []  # indices of keyframes for each shot, 2D
SHOT_KEYFRAME_HISTOGRAMS = []  # histograms of keyframes for each shot, 2D
SHOT_AVERAGED_KEYFRAMES = []  # averaged keyframes for each shot, 1D
SHOT_AVERAGED_KEYFRAME_HISTOGRAMS = []  # averaged keyframe histograms for each shot, 1D
SHOT_AVERAGED_FRAMES = []  # averaged frames for each shot, 1D
SHOT_AVERAGED_FRAME_HISTOGRAMS = []  # averaged frame histograms for each shot, 1D


# Histogram params
h_bins = 32
s_bins = 16
v_bins = 16
hist_size = [h_bins, s_bins, v_bins]
# hue varies from 0 to 179, saturation from 0 to 255
h_ranges = [0, 180]
s_ranges = [0, 256]
v_ranges = [0, 256]
ranges = h_ranges + s_ranges + v_ranges  # concat lists
# Use all 3 channels
channels = [0, 1, 2]

GROUPING_THRESH = 60


def similarity_between_two_histograms(hist1, hist2):
    """
    Calculates the similarity between two histograms
    """
    # hist1 /= hist1.sum()
    # hist2 /= hist2.sum()
    # return cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)
    return 1 - cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)


def load_video(video_file):
    """Loads the video and returns the video capture object, width, height, fps, and total frame count"""
    cap = cv2.VideoCapture(video_file)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if DEBUG:
        print("Width:", width)
        print("Height:", height)
        print("FPS:", fps)
        print("Total frame count:", count)

    return cap, width, height, fps, count


def detect_shots_sudden(cap):
    """
    Detects shots in the video and stores the shot boundaries in SHOTS and SHOTS_SECONDS, and the histograms of individual frames in FRAMES_HISTOGRAMS
    """
    i = 1
    n = 1  # length of current shot
    min_scene_len = MIN_SHOT_LEN_SECS * FPS  # in frames

    ret, prev_frame = cap.read()
    prev_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(prev_hsv.astype(np.int32))
    avg_H = np.mean(H)
    avg_S = np.mean(S)
    avg_V = np.mean(V)
    avg_frame = np.array(prev_hsv, dtype=np.float128)

    # Process the first frame
    hist = cv2.calcHist([prev_hsv], channels, None, hist_size, ranges, accumulate=False)
    # cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    # hist /= hist.sum()
    FRAMES_HISTOGRAMS.append(hist)

    while True:
        if i % 1000 == 0 and DEBUG:
            print("Frame " + str(i))

        # Read next frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to HSV space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Calculate and store the histogram
        hist = cv2.calcHist([hsv], channels, None, hist_size, ranges, accumulate=False)
        # cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        # hist /= hist.sum()
        FRAMES_HISTOGRAMS.append(hist)

        # Calculate the pixelwise difference
        diff = np.mean(np.abs(hsv.astype(np.int32) - prev_hsv.astype(np.int32)))

        # Split out H, S, V
        H, S, V = cv2.split(hsv.astype(np.int32))

        if (
            diff >= SHOT_THRESHOLD
            and ((i - SHOTS[-1]) > GROUPING_THRESH)
            and n >= min_scene_len
        ):
            SHOTS.append(i)
            SHOTS_SECONDS.append(i / 30)
            n = 1
            avg_H = 0.0
            avg_S = 0.0
            avg_V = 0.0
            # Store and reset the average frame
            SHOT_AVERAGED_FRAMES.append(avg_frame)
            avg_frame = np.array(hsv, dtype=np.float128)
        else:
            # Update the average frame
            avg_frame += (hsv.astype(np.float128) - avg_frame) / (n + 1)

        avg_H += (np.mean(H) - avg_H) / (n)
        avg_S += (np.mean(S) - avg_S) / (n)
        avg_V += (np.mean(V) - avg_V) / (n)

        prev_hsv = hsv
        i += 1
        n += 1

    SHOT_AVERAGED_FRAMES.append(avg_frame)

    global SHOTS_WITH_END_MARKER
    SHOTS_WITH_END_MARKER = SHOTS + [len(FRAMES_HISTOGRAMS)]
    global N_SHOTS
    N_SHOTS = len(SHOTS)


def accumulate_histograms_within_shots():
    """
    Accumulates the histograms of the frames within each shot and stores them in ACCUMULATED_SHOTS_HISTOGRAMS
    """
    prev_shot = 0
    for shot in SHOTS_WITH_END_MARKER[1:]:
        # print("From " + str(prev_shot) + " to " + str(shot))
        hist = None
        count = 0
        for frame_idx in range(prev_shot, shot):
            if hist is None:
                hist = np.array(FRAMES_HISTOGRAMS[frame_idx])
            else:
                hist += FRAMES_HISTOGRAMS[frame_idx]
            count += 1
        hist /= count
        ACCUMULATED_SHOTS_HISTOGRAMS.append(hist)
        prev_shot = shot


def calc_midpoint_histograms():
    """
    Calculates the histograms of the midpoint frame of each shot and stores them in MIDPOINT_SHOTS_HISTOGRAMS
    """
    for idx in range(N_SHOTS):
        midpoint = SHOTS_WITH_END_MARKER[idx] + (
            (SHOTS_WITH_END_MARKER[idx + 1] - SHOTS_WITH_END_MARKER[idx]) // 2
        )
        MIDPOINT_SHOTS_HISTOGRAMS.append(FRAMES_HISTOGRAMS[midpoint])


def calc_average_frame_histograms():
    """
    Calculates the normalized average frame histogram for each shot and stores them in SHOT_AVERAGED_FRAME_HISTOGRAMS
    Also stores the averaged frames in SHOT_AVERAGED_FRAMES
    """
    for shot_idx in range(N_SHOTS):
        avg_frame = SHOT_AVERAGED_FRAMES[shot_idx]
        hist = cv2.calcHist(
            [avg_frame.astype(np.float32)], channels, None, hist_size, ranges
        )
        # cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        # hist /= hist.sum()
        SHOT_AVERAGED_FRAME_HISTOGRAMS.append(hist)


def calc_keyframe_histograms():
    """
    Extracts keyframes from each shot and calculates their histograms
    Keyframe indices are stored in SHOT_KEYFRAME_INDICES
    Keyframe histograms are stored in SHOT_KEYFRAME_HISTOGRAMS
    """

    for shot_idx in range(N_SHOTS):
        if DEBUG:
            print()
            print("Calculating keyframes for shot " + str(shot_idx))
        # Start with the midpoint frame
        midpoint_frame_idx = SHOTS_WITH_END_MARKER[shot_idx] + (
            (SHOTS_WITH_END_MARKER[shot_idx + 1] - SHOTS_WITH_END_MARKER[shot_idx]) // 2
        )
        keyframes = set()
        if DEBUG:
            print(
                "Shot "
                + str(shot_idx)
                + " keyframes initialized with midpoint frame: "
                + str(midpoint_frame_idx)
            )
        keyframes.add(midpoint_frame_idx)

        # For each other frame in the shot, calculate its maximum intersection with all current keyframes
        for frame_idx in range(
            SHOTS_WITH_END_MARKER[shot_idx], SHOTS_WITH_END_MARKER[shot_idx + 1]
        ):
            max_intersection = float("-inf")
            for keyframe_idx in keyframes:
                intersection = similarity_between_two_histograms(
                    FRAMES_HISTOGRAMS[frame_idx], FRAMES_HISTOGRAMS[keyframe_idx]
                )
                if intersection > max_intersection:
                    max_intersection = intersection
            # If the maximum intersection is below the threshold, add the frame as a keyframe
            if max_intersection < KEYFRAME_SIMILARITY_THRESHOLD:
                if DEBUG:
                    print(
                        "Max intersection for frame "
                        + str(frame_idx)
                        + " is "
                        + str(max_intersection)
                        + ", below threshold "
                        + str(KEYFRAME_SIMILARITY_THRESHOLD)
                        + " adding as keyframe"
                    )
                keyframes.add(frame_idx)

        # Store the keyframe indices
        keyframes = sorted(list(keyframes))
        SHOT_KEYFRAME_INDICES.append(keyframes)

        # Store the histograms of the keyframes
        keyframe_histograms = []
        for keyframe_idx in keyframes:
            keyframe_histograms.append(FRAMES_HISTOGRAMS[keyframe_idx])
        SHOT_KEYFRAME_HISTOGRAMS.append(keyframe_histograms)


def calc_1d_shot_coherence(shot1_feat, shot2_feat):
    """
    Calculates the coherence between two shots
    shot1_feat and shot2_feat are the features of the two shots
    Each feature is a single histogram
    """
    out = similarity_between_two_histograms(shot1_feat, shot2_feat)
    return out


def calc_2d_shot_coherence(shot1_feat, shot2_feat):
    """
    Calculates the coherence between two shots as defined in https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.152.8574&rep=rep1&type=pdf
    shot1_feat and shot2_feat are the features of the two shots. Each feature is a list of histograms
    """
    # Shot coherence is the max coherence between any pair of keyframes
    max_coherence = float("-inf")
    for shot1_keyframe in shot1_feat:
        for shot2_keyframe in shot2_feat:
            coherence = similarity_between_two_histograms(
                shot1_keyframe, shot2_keyframe
            )
            if coherence > max_coherence:
                max_coherence = coherence
    return max_coherence


def calculate_backwards_shot_coherence(
    shot_features, current_shot_idx, start_of_scene_shot_idx
):
    """
    Looks back for the best matching shot and returns the match score
    """
    best_shot_coherence = float("-inf")
    start_idx = max(current_shot_idx - N_SHOTS_LOOK_BACK, start_of_scene_shot_idx)

    if DEBUG:
        print()
        print(
            "Calculating backwards shot coherence from "
            + str(start_idx)
            + " to "
            + str(current_shot_idx)
        )

    # Check the dimensionality of the shot features
    if isinstance(shot_features[0], list):
        dim = 2
        calc_shot_coherence = calc_2d_shot_coherence
    else:
        dim = 1
        calc_shot_coherence = calc_1d_shot_coherence

    for idx in range(start_idx, current_shot_idx):
        shot_coherence = calc_shot_coherence(
            shot_features[idx], shot_features[current_shot_idx]
        )
        if DEBUG:
            print(
                "Shot coherence between "
                + str(idx)
                + " and "
                + str(current_shot_idx)
                + " is "
                + str(shot_coherence)
            )

        if shot_coherence > best_shot_coherence:
            best_shot_coherence = shot_coherence

    if DEBUG:
        print("Best shot coherence is " + str(best_shot_coherence))
    return best_shot_coherence


def detect_potential_scene_boundaries(shot_features, method="valley"):
    """
    shot_features is a list of size equal to the number of shots
    each element is either a single histogram or a list of histograms
    method can be:
        'valley' - marks a new scene if the drop in shot coherence is above a threshold
        'threshold' - marks a new scene if the shot coherence is below a threshold
    """
    print("Detecting scenes...")
    print("First calculating all backwards shot coherences")
    start_of_scene_shot_idx = 0
    backwards_shot_coherences = [FIRST_BSC]
    for shot_idx in range(1, N_SHOTS):
        backward_shot_coherence = calculate_backwards_shot_coherence(
            shot_features, shot_idx, start_of_scene_shot_idx
        )
        backwards_shot_coherences.append(backward_shot_coherence)

    # if DEBUG:
    #     print()
    #     print("Replacing first BSC with the mean of the rest")
    #     backwards_shot_coherences[0] = np.mean(backwards_shot_coherences[1:])

    if DEBUG:
        print()
        print("Backwards shot coherences:")
        print(backwards_shot_coherences)
        print()

    # Calculate the scene boundaries
    if method == "threshold":
        if DEBUG:
            print("Using threshold method to detect PSBs")
        for shot_idx in range(1, N_SHOTS):
            shot = SHOTS[shot_idx]
            if (
                backwards_shot_coherences[shot_idx] < SCENE_DETECTION_THRESHOLD
                and shot - SHOTS[start_of_scene_shot_idx] > MIN_SCENE_LEN_SECS * FPS
            ):
                SCENES.append(shot)
                SCENES_SECONDS.append(shot / 30)
                start_of_scene_shot_idx = shot_idx
                if DEBUG:
                    print("Shot " + str(shot_idx) + " is the start of a scene")
    elif method == "valley":
        if DEBUG:
            print("Using valley method to detect PSBs")
        for shot_idx in range(1, N_SHOTS):
            shot = SHOTS[shot_idx]
            if (
                backwards_shot_coherences[shot_idx - 1]
                - backwards_shot_coherences[shot_idx]
                > SCENE_DETECTION_VALLEY_THRESHOLD
                and shot - SHOTS[start_of_scene_shot_idx] > MIN_SCENE_LEN_SECS * FPS
            ):
                SCENES.append(shot)
                SCENES_SECONDS.append(shot / 30)
                start_of_scene_shot_idx = shot_idx
                if DEBUG:
                    print("Shot " + str(shot_idx) + " is the start of a scene")

    if DEBUG:
        print("Potential scene boundaries:")
        print(SCENES)


def filter_scenes_on_keyframe_similarity():
    """
    Filters out potential scene boundaries where any two keyframes of adjacent scenes have color similarity
    greater than PSB_FILTERING_COLOR_THRESHOLD
    """
    # TODO: implement this


def detect_potential_scenes_with_clustering(shot_features, eps=0.5, min_samples=2):
    """
    Groups shots into scenes using the DBSCAN clustering algorithm
    """
    hists = []
    for hist in shot_features:
        cv2.normalize(hist, hist)
        hists.append(hist.flatten())

    # Normalize features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(hists)

    # Cluster using DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(normalized_features)

    import ipdb

    ipdb.set_trace()

    return labels


def detect_subshots(
    shot_change_indices,
    video_file,
    motion_threshold=1.0,
    hist_threshold=0.01,
    window_size=2,
    ignore_frames=10,
    proximity=20,
):
    cap = cv2.VideoCapture(video_file)
    ret, frame1 = cap.read()
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame1_gray = cv2.resize(frame1_gray, (0, 0), fx=0.5, fy=0.5)

    subshots = []
    subshots_seconds = []
    frame_count = 0
    high_motion_count = 0
    motion_scores = []

    def is_near_shot_boundary(frame_count, shot_change_indices, proximity):
        for index in shot_change_indices:
            if abs(frame_count - index) <= proximity:
                return True
        return False

    while True:
        ret, frame2 = cap.read()
        if not ret:
            break

        if frame_count % 1000 == 0:
            print(f"Frame {frame_count}")

        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        frame2_gray = cv2.resize(frame2_gray, (0, 0), fx=0.5, fy=0.5)

        if is_near_shot_boundary(frame_count, shot_change_indices, proximity):
            frame_count += 1
            frame1_gray = frame2_gray
            continue

        flow = cv2.calcOpticalFlowFarneback(
            frame1_gray, frame2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_score = np.mean(magnitude)
        motion_scores.append(motion_score)

        if len(motion_scores) > window_size:
            motion_scores.pop(0)

        hist1 = cv2.calcHist([frame1_gray], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([frame2_gray], [0], None, [256], [0, 256])
        hist_diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

        if (
            np.mean(motion_scores) > motion_threshold
            and abs(hist_diff) > hist_threshold
        ):
            high_motion_count += 1
            if high_motion_count > ignore_frames:
                subshots.append(frame_count)
                subshots_seconds.append(frame_count / FPS)
                high_motion_count = 0
        else:
            high_motion_count = 0

        frame_count += 1
        frame1_gray = frame2_gray

    cap.release()

    return subshots, subshots_seconds


def detect_subshots_new(
    shot_change_indices,
    video_file,
    motion_threshold=1.0,
    hist_threshold=0.01,
    window_size=2,
    ignore_frames=10,
    proximity=20,
    proximity_to_subshot=30,
):
    cap = cv2.VideoCapture(video_file)
    ret, frame1 = cap.read()
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame1_gray = cv2.resize(frame1_gray, (0, 0), fx=0.5, fy=0.5)

    subshots = []
    subshots_seconds = []
    frame_count = 0
    high_motion_count = 0
    motion_scores = []

    def is_near_shot_boundary(frame_count, shot_change_indices, proximity):
        for index in shot_change_indices:
            if abs(frame_count - index) <= proximity:
                return True
        return False

    def is_near_subshot(frame_count, subshots, proximity_to_subshot):
        for index in subshots:
            if abs(frame_count - index) <= proximity_to_subshot:
                return True
        return False

    while True:
        ret, frame2 = cap.read()
        if not ret:
            break

        if frame_count % 1000 == 0:
            print(f"Frame {frame_count}")

        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        frame2_gray = cv2.resize(frame2_gray, (0, 0), fx=0.5, fy=0.5)

        if is_near_shot_boundary(
            frame_count, shot_change_indices, proximity
        ) or is_near_subshot(frame_count, subshots, proximity_to_subshot):
            frame_count += 1
            frame1_gray = frame2_gray
            continue

        flow = cv2.calcOpticalFlowFarneback(
            frame1_gray, frame2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_score = np.mean(magnitude)
        motion_scores.append(motion_score)

        if len(motion_scores) > window_size:
            motion_scores.pop(0)

        hist1 = cv2.calcHist([frame1_gray], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([frame2_gray], [0], None, [256], [0, 256])
        hist_diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

        if (
            np.mean(motion_scores) > motion_threshold
            and abs(hist_diff) > hist_threshold
        ):
            high_motion_count += 1
            if high_motion_count > ignore_frames:
                subshots.append(frame_count)
                subshots_seconds.append(frame_count / FPS)
                high_motion_count = 0
        else:
            high_motion_count = 0

        frame_count += 1
        frame1_gray = frame2_gray

    cap.release()

    return subshots, subshots_seconds


def main():
    # video_file = "./data/The_Long_Dark_rgb/InputVideo.mp4"
    # audio_file = "./data/The_Long_Dark_rgb/InputAudio.wav"
    # video_file_rgb = "./data/The_Long_Dark_rgb/InputVideo.rgb"

    video_file = "./data/The_Great_Gatsby_rgb/InputVideo.mp4"
    audio_file = "./data/The_Great_Gatsby_rgb/InputAudio.wav"
    video_file_rgb = "./data/The_Great_Gatsby_rgb/InputVideo.rgb"

    cap, width, height, fps, count = load_video(video_file)
    global FPS
    FPS = fps
    # audio = load_audio(audio_file)

    print("Detecting shots...")
    detect_shots_sudden(cap)
    print()
    print("Shots timestamps:", SHOTS_SECONDS)
    print()
    print("Accumulating histograms within shots...")
    accumulate_histograms_within_shots()

    # Calculate the auxiliary histograms
    calc_midpoint_histograms()
    calc_keyframe_histograms()
    calc_average_frame_histograms()

    # Go looking for scenes
    # detect_potential_scene_boundaries(
    #     shot_features=SHOT_KEYFRAME_HISTOGRAMS, method="valley"
    # )

    # detect_potential_scene_boundaries(
    #     shot_features=ACCUMULATED_SHOTS_HISTOGRAMS, method="valley"
    # )

    detect_potential_scene_boundaries(
        shot_features=SHOT_AVERAGED_FRAME_HISTOGRAMS, method="valley"
    )

    # detect_potential_scenes_with_clustering(
    #     shot_features=SHOT_AVERAGED_FRAME_HISTOGRAMS
    # )

    print()
    print("Scenes timestamps:", SCENES_SECONDS)
    print()

    # Look for subshots in between shots
    print("Detecting subshots...")
    SUBSHOTS, SUBSHOTS_SECONDS = detect_subshots(SHOTS, video_file)
    print()
    print("Subshots timestamps:", SUBSHOTS_SECONDS)
    print()

    # Call video player
    video_player_main(video_file, SCENES, SHOTS, SUBSHOTS, fps, width, height)


if __name__ == "__main__":
    main()
