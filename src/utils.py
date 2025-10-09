import sys
import numpy as np
import cv2 as cv
import albumentations as A

import torch
from model import FluidSegmentationModel


def moving_average(x, window_len):
    """
    Computes a moving average of the input signal using a uniform window.
    
    Args:
        x: Input signal to be smoothed
        window_len: Length of the moving average window
        
    Returns:
        np.ndarray: Smoothed signal with reduced noise hopefully.
    """
    weights = np.ones(window_len) / window_len
    zs_smooth = np.convolve(x, weights, mode='valid')
    return zs_smooth

def check_end_of_expansion(horizontal_distances: list,
                     smooth_distances: list,
                     distance_window_len: int,
                     velocities: list,
                     velocity_window_len: int,
                     frame_number: int) -> int:
    """
    Checks if the expansion of the rheometer is already over.
    
    Args:
        horizontal_distances: list of horizontal distances over frames
        smooth_distances: list of filtered horizontal distances
        distance_window_len: window length for distance vector smoothing
        velocities: list of calculated velocities
        velocity_window_len: window length for velocity smoothing
        frame_number: current frame number

    Returns:
        end_of_expansion: if the expansion ended, returns the frame number, else None
    """
    if len(horizontal_distances) > distance_window_len:
        smooth_distances.append(moving_average(horizontal_distances[-distance_window_len-1:-1],
                                  distance_window_len))
        
    if len(smooth_distances) > 2:
        velocity = smooth_distances[-1] - smooth_distances[-2]
        velocities.append(velocity[0])

    if len(velocities) > velocity_window_len:
        smooth_velocity = moving_average(velocities[-velocity_window_len-1:-1],
                                              velocity_window_len)[0]

    if horizontal_distances[-1] > horizontal_distances[0] * 1.5 and np.abs(smooth_velocity) < 0.1:
            end_of_expansion = frame_number
            print(f'End of expansion detected at frame: {end_of_expansion}')
            return end_of_expansion
    return None

def check_break_moment(contours: list, frame_number: int) -> int:
    """
    Checks if the fluid already broke.
    
    Args:
        contours: list of detected contours in the current frame
        frame_number: current frame number

    Returns:
        break_frame: if the fluid broke, returns the frame number, else None
    """
    if len(contours) > 1:
        areas = [cv.contourArea(cnt) for cnt in contours]
        largest_area = max(areas)
        areas.remove(largest_area)
        second_largest_area = max(areas) if areas else 0
        if second_largest_area > largest_area * 0.5:
            break_frame = frame_number
            print(f'Break detected at frame: {frame_number}')
            return break_frame
    return None

def get_fluid_bounds(contours: list,
                     fluid_broke: bool,) -> tuple:
    """
    Determines the bounding box of the fluid based on contours detected in
    the binary mask.
    
    Args:
        contours: list of detected contours in the current frame
        fluid_broke: boolean indicating if the fluid has broken
    
    Returns:
        left_bound: left x-coordinate of the fluid region
        right_bound: guess bro
        top_bound: top y-coordinate of the fluid region
        bottom_bound: guess bro
    """

    largest_contour = max(contours, key=cv.contourArea) 

    if not fluid_broke:
        left_bound = largest_contour[:, 0, 0].min()
        right_bound = largest_contour[:, 0, 0].max()
        top_bound = largest_contour[:, 0, 1].min()
        bottom_bound = largest_contour[:, 0, 1].max()
    else:
        two_largest_contours = sorted(contours, key=cv.contourArea, reverse=True)[:2]
        left_bound = min(two_largest_contours[0][:, 0, 0].min(), two_largest_contours[1][:, 0, 0].min())
        right_bound = max(two_largest_contours[0][:, 0, 0].max(), two_largest_contours[1][:, 0, 0].max())
        top_bound = min(two_largest_contours[0][:, 0, 1].min(), two_largest_contours[1][:, 0, 1].min())
        bottom_bound = max(two_largest_contours[0][:, 0, 1].max(), two_largest_contours[1][:, 0, 1].max())

    return (left_bound, right_bound, top_bound, bottom_bound)


def get_binary_mask(frame: np.ndarray,
                    model: FluidSegmentationModel,
                    device: str,
                    transform) -> np.ndarray:
    """
    Uses a segmentation CNN to get a binary mask for the input frame.

    Args:
        frame: single video frame in BGR format 
        model: trained segmentation model
        device: 'cpu' or 'cuda', must match with model's device
        transform: preprocessing transformations for cnn

    Returns:
        pred_mask: binary mask of the segmented object as a numpy array, single channel 
    """
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame = transform(image=frame)['image'].unsqueeze(0)  # Add batch dimension
    frame = frame.to(device)

    with torch.inference_mode():
        pred_mask = torch.sigmoid(model(frame))
    pred_mask = (pred_mask > 0.5).float()
    pred_mask = pred_mask.squeeze().cpu().numpy()  # Remove batch dimension and convert to numpy
    pred_mask = (pred_mask * 255).astype(np.uint8)

    return pred_mask

def find_narrowest_point(binary_mask: np.ndarray,
                         left_bound: int,
                         right_bound: int) -> tuple:
    """
    Finds the narrowest point in the segmented fluid region.
    
    Args:
        binary_mask: binary mask of the segmented object as a numpy array, single channel 
        left_bound: left x-coordinate of the fluid region
        right_bound: right x-coordinate of the fluid region
    
    Returns:
        narrowest_h: height of the narrowest point in pixels
        narrowest_x: x-coordinate of the narrowest point in pixels"""

    maks_region_sums = binary_mask[:, left_bound:right_bound].sum(axis=0)
    narrowest_relative_x = np.argmin(maks_region_sums)
    narrowest_x = left_bound + narrowest_relative_x 
    narrowest_h = maks_region_sums[narrowest_relative_x] // 255 # convert from pixel intensity to pixel count

    return narrowest_h, narrowest_x

def find_gap_center(binary_mask: np.ndarray,
                    left_bound: int,
                    right_bound:int) -> int:
    """
    Finds the center of the gap in the segmented fluid region. Should be called
    only after the fluid has broken.
    
    Args:
        binary_mask: binary mask of the segmented object as a numpy array, single channel 
        left_bound: left x-coordinate of the fluid region
        right_bound: right x-coordinate of the fluid region
    
    Returns:
        gap_center: x-coordinate of the gap center in pixels
    """
    mask_region_sums = binary_mask[:, left_bound:right_bound].sum(axis=0)
    
    gap_columns = np.where(mask_region_sums == 0)[0]

    gap_center_relative = gap_columns[0] + (gap_columns[-1] - gap_columns[0]) // 2
    gap_center = left_bound + gap_center_relative

    print(gap_center)

    return gap_center

def crop_fluid_region(frame: np.ndarray,
                      bounds: tuple,
                      w_ratio: int,
                      h_ratio: int,
                      model_input_size: int) -> tuple:
    """
    """

    original_bounds = (int(bounds[0] * w_ratio),
                       int(bounds[1] * w_ratio),
                       int(bounds[2] * h_ratio),
                       int(bounds[3] * h_ratio)) # (l, r, t, b)
    
    orig_h = original_bounds[3] - original_bounds[2]
    orig_w = original_bounds[1] - original_bounds[0]

    if orig_h < model_input_size:
        diff = model_input_size - orig_h
        pad = diff // 2
        new_top = max(0, original_bounds[2] - pad)
        new_bottom = min(frame.shape[0], original_bounds[3] + pad)
    else: 
        new_top = max(0, original_bounds[2] - int(0.1 * orig_h))
        new_bottom = min(frame.shape[0], original_bounds[3] + int(0.1 * orig_h))


    return new_top, new_bottom


def get_start_end_frames(cap: cv.VideoCapture,
                         model: FluidSegmentationModel,
                         device: str,
                         transform: A.Compose,
                         display: bool) -> tuple:

    distance_window_len = 7
    velocity_window_len = 5
    horizontal_distances = []
    smooth_distances = []
    velocities = []

    end_of_expansion_frame = None

    fluid_broke = False

    break_moment_frame = None

    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    while True:
        frame_number = int(cap.get(cv.CAP_PROP_POS_FRAMES))
        ret, frame = cap.read()
        if not ret:
            break
        
        orig_frame = frame.copy()

        pred_mask = get_binary_mask(frame, model, device, transform)
        
        contours = cv.findContours(pred_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]

        if not end_of_expansion_frame:
            # fluid broke here before the check is stupid but whatever,
            # expansion ends before break so it's fine :D
            bounds = get_fluid_bounds(contours, fluid_broke) # left, right, top, bottom       
            horizontal_distance = bounds[1] - bounds[0]
            horizontal_distances.append(horizontal_distance)
            end_of_expansion_frame = check_end_of_expansion(horizontal_distances,
                                                    smooth_distances,
                                                    distance_window_len,
                                                    velocities,
                                                    velocity_window_len,
                                                    frame_number)
            
        if end_of_expansion_frame and not break_moment_frame:
            break_moment_frame = (check_break_moment(contours, frame_number))
            fluid_broke = bool(break_moment_frame)

        if fluid_broke:
            break

        if frame_number == total_frames - 1:
            print('Reached end of video without detecting fluid break.')
            

        # here fluid broke is important so it's after the check
        
        # display frames stuff
        if display:
            pred_mask = cv.cvtColor(pred_mask, cv.COLOR_GRAY2BGR)
            small_frame = cv.resize(orig_frame, (512, 512))
            combined = cv.hconcat([small_frame, pred_mask])
            
            cv.imshow('Segmentation Inference', combined)

            key_code = cv.waitKey(1)  
            if key_code == 27:  
                cv.destroyAllWindows()
                break

    cv.destroyAllWindows()

    return end_of_expansion_frame, break_moment_frame

def get_expected_narrowest_point_v1(cap: cv.VideoCapture,
                            model: FluidSegmentationModel,
                            device: str,
                            transform: A.Compose,
                            end_of_expansion_frame: int,
                            break_frame: int,
                            crop_indexes: tuple, # (new_top, new_bottom)
                            display: bool) -> int:
    
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    if break_frame:
        cap.set(cv.CAP_PROP_POS_FRAMES, max(end_of_expansion_frame,break_frame - 100))
    else:
        cap.set(cv.CAP_PROP_POS_FRAMES, max(total_frames - 100, end_of_expansion_frame))
    
    narrowest_points = []

    while True:
        current_frame = int(cap.get(cv.CAP_PROP_POS_FRAMES))
        if break_frame is not None:
            if current_frame >= break_frame:
                fluid_broke = True
            else:
                fluid_broke = False
        else:
            fluid_broke = False
        
        ret, frame = cap.read()

        cropped_frame = frame[crop_indexes[0]:crop_indexes[1], :]
        cropped_mask = get_binary_mask(cropped_frame, model, device, transform)

        if not fluid_broke and current_frame >= end_of_expansion_frame:
            contours = cv.findContours(cropped_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]
            bounds = get_fluid_bounds(contours, fluid_broke)
            narrowest_h, narrowest_x = find_narrowest_point(cropped_mask, bounds[0] + 10, bounds[1] - 10)
            narrowest_points.append(narrowest_x)

        if fluid_broke or current_frame == total_frames - 1:
            break

        if display:
            cropped_frame_resized = cv.resize(cropped_frame, (512, 512))
            cropped_mask_bgr = cv.cvtColor(cropped_mask, cv.COLOR_GRAY2BGR)
            combined_cropped = np.hstack((cropped_frame_resized, cropped_mask_bgr))

            cv.imshow('Cropped Frame | Cropped Mask', combined_cropped)

            key_code = cv.waitKey(1)
            if key_code == 27:  # ESC key to exit
                break

    cv.destroyAllWindows()
    x_expected = int(np.mean(narrowest_points))
    return x_expected