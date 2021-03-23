import numpy as np
import cv2
import argparse
from stabilize import stabilizeVideo
from localize import localizeVideo

parser = argparse.ArgumentParser(description="Video Query tool")
parser.add_argument("query", help="query video")
args = parser.parse_args()

stabilizeVideo(args.query, "stabilized.avi")
localizeVideo("stabilized.avi", "cropped.avi")
