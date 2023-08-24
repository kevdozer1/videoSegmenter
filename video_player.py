import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QListWidget,
    QVBoxLayout,
    QWidget,
    QLabel,
    QPushButton,
    QListWidgetItem,
)
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QTimer, Qt, QEventLoop
from PyQt5.QtGui import QImage, QPixmap
import sounddevice as sd
import soundfile as sf

from pydub import AudioSegment
from pydub.playback import (
    play,
    _play_with_simpleaudio,
    _play_with_pyaudio,
    _play_with_ffplay,
)

from ffpyplayer.player import MediaPlayer
import time
from PIL import Image
import threading


class VideoPlayer:
    change_pixmap_signal = pyqtSignal(QImage)

    def __init__(
        self,
        hierarchy_list,
        mp4_file,
        scene_idxs,
        frames_per_second,
        width,
        height,
    ):
        super().__init__()
        self.hierarchy_list = hierarchy_list
        self.fps = frames_per_second
        self.width = width
        self.height = height

        self.scene_idxs = scene_idxs
        self.current_scene = 0
        self.scene_timestamps = [idx / self.fps for idx in self.scene_idxs]

        self.video_label = None

        self.playing = False
        self.paused = False
        self.audio_playback = None

        self.frame_duration = 1 / self.fps

        self.player = MediaPlayer(
            mp4_file, ff_opts={"paused": True, "x": width, "y": height}
        )

        time.sleep(1)  # so that video_label is added - clumsy
        self.ignore_timestamps_for = 0  # set this to 10 when seeking, so that the first few timestamps - which are likely to be wrong - are ignored

        player_thread = threading.Thread(target=self.play, args=(), daemon=True)
        player_thread.start()

    def play(self):
        while True:
            frame, val = self.player.get_frame()
            # if val == "eof":
            #     break

            if isinstance(val, str) or val == 0.0:
                # print("wait default 32")
                waitkey = 32
            else:
                waitkey = int(val * 100)
                # print("wait", waitkey)

            if waitkey != 0:  # prevents some niche bug
                pressed_key = cv2.waitKey(waitkey) & 0xFF
            else:
                pressed_key = cv2.waitKey(32) & 0xFF

            if frame is None:
                continue

            img, timestamp = frame
            # print("Timestamp:", timestamp)
            data = img.to_bytearray()[0]
            # image = Image.frombytes("RGB", (self.width, self.height), bytes(data))
            # image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # h, w, ch = image.shape
            # qimage = QImage(image.data, w, h, ch * w, QImage.Format_RGB888)
            qimage = QImage(data, self.width, self.height, QImage.Format_RGB888)
            self.pixmap = QPixmap.fromImage(qimage)
            self.video_label.setPixmap(self.pixmap)
            # self.video_label.setFixedWidth(self.video_label.pixmap().width())
            # self.video_label.setFixedHeight(self.video_label.pixmap().height())
            # self.video_label.parent().update()

            # Use timestamp to update the current scene
            if self.ignore_timestamps_for > 0:
                self.ignore_timestamps_for -= 1
            else:
                while (
                    self.current_scene < len(self.scene_timestamps) - 1
                    and timestamp >= self.scene_timestamps[self.current_scene + 1]
                ):
                    self.current_scene += 1
                    print("In play, current scene incremented to:", self.current_scene)
                    self.hierarchy_list.setCurrentRow(self.current_scene)

    def handle_play(self):
        self.player.set_pause(False)

    def handle_pause(self):
        self.player.set_pause(True)

    def handle_stop(self):
        self.player.set_pause(True)
        seek_to = self.scene_idxs[self.current_scene] / self.fps
        self.player.seek(seek_to, relative=False)

    def jump_to_frame(self, frame_idx):
        seek_to = frame_idx / self.fps
        self.current_scene = self.scene_idxs.index(
            frame_idx
        )  # assume we'll always be jumping to a bookmark
        print("In jump, current scene is", self.current_scene)
        self.ignore_timestamps_for = 10
        self.player.seek(seek_to, relative=False)


class MainWindow(QMainWindow):
    def __init__(self, mp4_file, scenes, shots, subshots, fps, width, height):
        super().__init__()
        self.setWindowTitle("Video Player")

        # Initialize hierarchy list so we can create the video player
        self.hierarchy_list = QListWidget(self)

        # Combine the lists of scenes, shots and subshots into a single list
        # Duplicates are allowed
        all_idxs = scenes + shots + subshots
        all_idxs.sort()

        self.all_idxs = all_idxs
        self.scenes = scenes
        self.shots = shots
        self.subshots = subshots

        # Initialize video player
        self.video_player = VideoPlayer(
            self.hierarchy_list,
            mp4_file,
            all_idxs,
            frames_per_second=fps,
            width=width,
            height=height,
        )

        # Initialize UI components
        self.init_ui()

        self.video_player.video_label = self.video_label

        # self.video_player.change_pixmap_signal.connect(self.update_video_frame)

    def init_ui(self):
        # Implement UI initialization here
        # Central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Layout
        layout = QVBoxLayout()

        # Video label
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(self.video_player.width, self.video_player.height)
        layout.addWidget(self.video_label)

        # Play, pause and stop buttons
        self.play_button = QPushButton("Play", self)
        self.play_button.clicked.connect(self.play_video)
        layout.addWidget(self.play_button)

        self.pause_button = QPushButton("Pause", self)
        self.pause_button.clicked.connect(self.pause_video)
        layout.addWidget(self.pause_button)

        self.stop_button = QPushButton("Stop", self)
        self.stop_button.clicked.connect(self.stop_video)
        layout.addWidget(self.stop_button)

        # Build the Hierarchy List
        self.hierarchy_list.itemClicked.connect(self.scene_selected)

        scene_label = 1
        shot_label = 1
        subshot_label = 1

        combined_bookmarks = sorted(list(set(self.scenes + self.shots + self.subshots)))

        for bookmark in combined_bookmarks:
            if bookmark in self.scenes:
                scene_item = QListWidgetItem(f"Scene {scene_label} at " + str(bookmark))
                scene_item.setData(Qt.UserRole, bookmark)
                self.hierarchy_list.addItem(scene_item)
                scene_label += 1
                shot_label = 1
                subshot_label = 1
            if bookmark in self.shots:
                shot_item = QListWidgetItem(f"   Shot {shot_label} at " + str(bookmark))
                shot_item.setData(Qt.UserRole, bookmark)
                self.hierarchy_list.addItem(shot_item)
                shot_label += 1
                subshot_label = 1
            if bookmark in self.subshots:
                subshot_item = QListWidgetItem(
                    f"      Subshot {subshot_label} at " + str(bookmark)
                )
                subshot_item.setData(Qt.UserRole, bookmark)
                self.hierarchy_list.addItem(subshot_item)
                subshot_label += 1

        layout.addWidget(self.hierarchy_list)

        # Set layout
        central_widget.setLayout(layout)

    def play_video(self):
        # Handle play button click
        self.video_player.handle_play()

    def pause_video(self):
        # Handle pause button click
        self.video_player.handle_pause()

    def stop_video(self):
        # Handle stop button click
        self.video_player.handle_stop()

    def scene_selected(self):
        # Handle scene selection
        selected_item = self.hierarchy_list.currentItem()
        if selected_item is not None:
            print("DEBUG: Jumping to scene" + str(selected_item.data(Qt.UserRole)))
            self.video_player.jump_to_frame(selected_item.data(Qt.UserRole))

    def closeEvent(self, event):
        self.video_player.stop()
        self.video_player.video_thread.quit()
        self.video_player.audio_thread.quit()
        # self.video_player.video_thread.wait()
        # self.video_player.audio_thread.wait()
        event.accept()


def main(mp4_file, scenes, shots, subshots, fps, width, height):
    app = QApplication(sys.argv)
    main_window = MainWindow(mp4_file, scenes, shots, subshots, fps, width, height)
    main_window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    # Define your input files and scenes here
    # rgb_file = "data/The_Long_Dark_rgb/InputVideo.rgb"
    # wav_file = "data/The_Long_Dark_rgb/InputAudio.wav"

    wav_file = "./data/The_Great_Gatsby_rgb/InputAudio.wav"
    rgb_file = "./data/The_Great_Gatsby_rgb/InputVideo.rgb"
    video_file = "./data/The_Great_Gatsby_rgb/InputVideo.mp4"

    # video_file = "./data/The_Great_Gatsby_rgb/trimmed.mp4"

    width = 480
    height = 270
    fps = 30

    scenes = [0, 300, 600, 900, 1200]
    shots = [0, 60, 120, 300, 600, 900, 1050, 1200]
    subshots = [90, 150]

    main(video_file, scenes, shots, subshots, fps, width, height)
