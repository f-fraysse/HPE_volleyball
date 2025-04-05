# HPE_volleyball Project Brief

## Project Overview

HPE_volleyball is a computer vision project that combines object detection, multi-object tracking, and pose estimation to analyze beach volleyball training sessions. The system processes video recordings to automatically track players and analyze their movements, with future plans to recognize specific actions like spikes.

## Primary Goals

1. **Automate Training Analysis**: Develop a robust pipeline that helps volleyball coaches save time by automating the analysis of player actions and behaviors from training videos.

2. **Player Detection & Tracking**: Implement reliable detection and tracking of players throughout training sessions, maintaining consistent player IDs.

3. **Pose Estimation**: Extract accurate skeletal pose data for each tracked player to enable detailed movement analysis.

4. **Performance Optimization**: Improve inference speed to enable near real-time analysis without sacrificing accuracy.

5. **Future: Action Recognition**: Develop capabilities to automatically identify and classify volleyball-specific actions (e.g., spikes, serves, blocks) based on pose data and movement patterns.

## Current Focus

The immediate focus is on optimizing the performance of the detection-tracking-pose pipeline. While the system currently works with acceptable accuracy, inference speed needs improvement to make the tool more practical for coaches.

## Technical Requirements

1. Process 1920x1080, 50fps video data efficiently
2. Maintain consistent player tracking IDs throughout a session
3. Extract accurate pose keypoints for movement analysis
4. Generate output with visual overlays for verification
5. Store structured data (bounding boxes, IDs, keypoints) for further analysis

## Constraints

1. Target hardware: RTX 4060 (lab PC)
2. Must work with standard volleyball training video setups (camera positioned ~5m behind back line, ~4m high)
3. Focus on beach volleyball specifically (2v2 format)

## Success Criteria

1. Reliable player detection and tracking throughout training sessions
2. Accurate pose estimation for tracked players
3. Inference speed of at least **50 FPS (20ms per frame)** on target hardware for potential real-time applications (Initial target of 15-20 FPS met).
4. Structured data output suitable for further analysis
5. Eventually: Accurate recognition of volleyball-specific actions
