# Video Segmenter

This python program automatically detects changes in scenes, shots, and subshots(camera movements) using OpenCV image processing data and scikit-learn clustering.

The generated scene, shot, and subshot indices are fed into a GUI application made with PyQT5 to play the source video alongside a hierarchical structure for visualizing the boundaries.

The user is able to navigate the video and skip to various scenes, shots, or subshots through this hierarchical structure.

## Setup
`pip install -r requirements.txt`

## Run
`python3 video_player.py`

![](https://github.com/kevdozer1/videoSegmenter/blob/main/playerPicture.png)
