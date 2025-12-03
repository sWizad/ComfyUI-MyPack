import json
import numpy as np
from typing import List, Tuple
import torch

class CalculatorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "number1": ("FLOAT", {"default": 0.0}),
                "number2": ("FLOAT", {"default": 0.0}),
                "operation": ("STRING", ),
                #"operation": (["+", "-", "*", "/", "%"], ),
            }
        }

    RETURN_TYPES = ("STRING",)
    #RETURN_TYPES = ("*",)
    FUNCTION = "calculate"
    CATEGORY = "MyPack"

    def calculate(self, number1, number2, operation):
        try:
            if operation == "+":
                result = number1 + number2
            elif operation == "-":
                result = number1 - number2
            elif operation == "*":
                result = number1 * number2
            elif operation == "/":
                if number2 == 0:
                    return ("Error: Division by zero",)
                result = number1 / number2
            elif operation == "%":
                result = number1 % number2
            else:
                return ("Error: Unknown operation",)

            return (f"Result: {result}",)
        except Exception as e:
            return (f"Error: {str(e)}",)


FIXED_LENGTH = 121

def pad_pts(tr):
    """Convert list of {x,y} to (FIXED_LENGTH,1,3) array, padding/truncating."""
    pts = np.array([[p['x'], p['y'], 1] for p in tr], dtype=np.float32)
    n = pts.shape[0]
    if n < FIXED_LENGTH:
        pad = np.zeros((FIXED_LENGTH - n, 3), dtype=np.float32)
        pts = np.vstack((pts, pad))
    else:
        pts = pts[:FIXED_LENGTH]
    return pts.reshape(FIXED_LENGTH, 1, 3)

def parse_json_tracks(tracks):
    tracks_data = []
    try:
        # If tracks is a string, try to parse it as JSON
        if isinstance(tracks, str):
            parsed = json.loads(tracks.replace("'", '"'))
            tracks_data.extend(parsed)
        else:
            # If tracks is a list of strings, parse each one
            for track_str in tracks:
                parsed = json.loads(track_str.replace("'", '"'))
                tracks_data.append(parsed)
        
        # Check if we have a single track (dict with x,y) or a list of tracks
        if tracks_data and isinstance(tracks_data[0], dict) and 'x' in tracks_data[0]:
            # Single track detected, wrap it in a list
            tracks_data = [tracks_data]
        elif tracks_data and isinstance(tracks_data[0], list) and tracks_data[0] and isinstance(tracks_data[0][0], dict) and 'x' in tracks_data[0][0]:
            # Already a list of tracks, nothing to do
            pass
        else:
            # Unexpected format
            print(f"Warning: Unexpected track format: {type(tracks_data[0])}")
            
    except json.JSONDecodeError as e:
        print(f"Error parsing tracks JSON: {e}")
        tracks_data = []

    return tracks_data

def paint_point_track(
    frames: np.ndarray,
    ref: np.ndarray,
    point_tracks: np.ndarray,
    visibles: np.ndarray,
    min_radius: int = 1,
    max_radius: int = 6,
    max_retain: int = 50
) -> np.ndarray:
    import cv2
    num_points, num_frames = point_tracks.shape[:2]
    H, W = frames.shape[1:3]

    first_frame = ref[0]
    if first_frame.dtype != np.uint8:
        first_frame = (first_frame * 255).astype(np.uint8)

    #video = np.zeros((num_frames, H, W, 3), dtype=np.uint8)
    video = (frames * 255).astype(np.uint8)
    
    track_colors = []
    for i in range(num_points):
        x, y = point_tracks[i, 0] + 0.5
        xi = int(np.clip(x, 0, W - 1))
        yi = int(np.clip(y, 0, H - 1))
        color_bgr = tuple(int(c) for c in first_frame[yi, xi, [0, 1, 2]])
        track_colors.append(color_bgr)

    for t in range(num_frames):
        frame = video[t]
        for i in range(num_points):
            if not visibles[i, t]:
                continue
            x, y = point_tracks[i, t] + 0.5
            xi = int(np.clip(x, 0, W - 1))
            yi = int(np.clip(y, 0, H - 1))
            cv2.circle(frame, (xi, yi), max_radius, track_colors[i], -1)
        video[t] = frame

    return video / 255.0
    
class CondVideoPointTracksVisualize:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "background": ("IMAGE",),
            "ref_images": ("IMAGE",),
            "tracks": ("STRING",),
            "min_radius": ("INT", {"default": 1, "min": 0, "max": 100, "step": 1, "tooltip": "radius for the very first point (oldest)"}),
            "max_radius": ("INT", {"default": 6, "min": 0, "max": 100, "step": 1, "tooltip": "radius for the current point (newest)"}),
            "max_retain": ("INT", {"default": 50, "min": 0, "max": 100, "step": 1, "tooltip": "Maximum number of points to retain"}),
        },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "patchmodel"
    CATEGORY = "MyPack"

    def patchmodel(self, background, ref_images, tracks, min_radius, max_radius, max_retain):
        tracks_data = parse_json_tracks(tracks)
        arrs = []
        for track in tracks_data:
            pts = pad_pts(track)
            arrs.append(pts)

        tracks_np = np.stack(arrs, axis=0)
        track = np.repeat(tracks_np, 2, axis=1)[:, ::3]
        points = track[:, :, 0, :2].astype(np.float32)
        visibles = track[:, :, 0, 2].astype(np.float32)

        if background.shape[0] < points.shape[1]:
            repeat_count = (points.shape[1] + background.shape[0] - 1) // background.shape[0]
            background = background.repeat(repeat_count, 1, 1, 1)
            background = background[:points.shape[1]]
        elif background.shape[0] > points.shape[1]:
            background = background[:points.shape[1]]

        video_viz = paint_point_track(background.cpu().numpy(), ref_images.cpu().numpy(), points, visibles, min_radius, max_radius, max_retain)
        video_viz = torch.from_numpy(video_viz).float()
        
        return (video_viz,)

NODE_CLASS_MAPPINGS = {
    "CalculatorNode": CalculatorNode,
    "CondVideoPointTracksVisualize": CondVideoPointTracksVisualize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CalculatorNode": "Basic Calculator",
    "CondVideoPointTracksVisualize": "Point Tracks to Video"
}