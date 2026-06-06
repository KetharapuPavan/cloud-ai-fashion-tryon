import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def detect_pose(image):
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return results.pose_landmarks
