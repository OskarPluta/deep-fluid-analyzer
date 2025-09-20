import sys
import cv2 as cv
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


sys.path.append(str(Path(__file__).parent / 'src'))
from dataset import get_validation_transforms
from model import create_model, FluidSegmentationModel


def find_narrowest_point_and_break(mask: np.ndarray) -> tuple:
    """
    Find the narrowest point in mask and detect if/where the mask breaks
    Returns: (narrowest_x, narrowest_height, break_detected, break_x, break_y)
    """
    height, width, _ = mask.shape
    mask = mask[:, :, 0]  
    
    # Find the leftmost and rightmost columns where the mask exists
    column_sums = mask.sum(axis=0)
    mask_columns = np.where(column_sums > 0)[0]
    
    if len(mask_columns) == 0:
        return width // 2, 0, False, -1, -1  # Return center if no mask found
    
    leftmost_col = mask_columns[0]
    rightmost_col = mask_columns[-1]
    mask_width = rightmost_col - leftmost_col
    
    # Calculate 10% bounds based on actual mask extent
    left_bound = leftmost_col + int(mask_width * 0.1)
    right_bound = rightmost_col - int(mask_width * 0.1)
    
    # Ensure bounds are valid
    left_bound = max(left_bound, leftmost_col)
    right_bound = min(right_bound, rightmost_col)
    
    if left_bound >= right_bound:
        return (leftmost_col + rightmost_col) // 2, 0, False, -1, -1
    
    # Find narrowest point in the valid range
    search_column_sums = mask[:, left_bound:right_bound + 1].sum(axis=0)
    narrowest_relative_index = np.argmin(search_column_sums)
    narrowest_index = narrowest_relative_index + left_bound
    narrowest_height = search_column_sums[narrowest_relative_index] // 255  # Convert from pixel intensity sum
    
    # Check for break detection
    break_detected = False
    break_x = -1
    break_y = -1
    
    # Method 1: Check if narrowest point has zero height (complete break)
    if narrowest_height == 0:
        break_detected = True
        break_x = narrowest_index
        # Find the y-coordinate where the break occurs (middle of the gap)
        break_y = height // 2
    
    # Method 2: Check for significant gaps in the fluid at the narrowest point
    if not break_detected:
        column = mask[:, narrowest_index]
        white_pixels = np.where(column > 127)[0]
        
        if len(white_pixels) > 0:
            # Check for gaps in the vertical direction
            gaps = []
            gap_positions = []
            
            for i in range(1, len(white_pixels)):
                gap_size = white_pixels[i] - white_pixels[i-1] - 1
                if gap_size > 3:  # Minimum gap size to consider as break
                    gaps.append(gap_size)
                    gap_positions.append((white_pixels[i-1], white_pixels[i]))
            
            # If we find a significant gap, this is a break
            if gaps and max(gaps) > 10:  # Threshold for break detection
                break_detected = True
                break_x = narrowest_index
                # Find the largest gap position
                largest_gap_idx = gaps.index(max(gaps))
                gap_start, gap_end = gap_positions[largest_gap_idx]
                break_y = (gap_start + gap_end) // 2
    
    # Method 3: Check multiple columns around narrowest point for consistency
    if not break_detected:
        check_range = min(5, (right_bound - left_bound) // 4)  # Check 5 pixels around or 25% of range
        
        # Collect all break positions to find the center
        break_positions = []
        
        for offset in range(-check_range, check_range + 1):
            check_x = narrowest_index + offset
            if left_bound <= check_x <= right_bound:
                column = mask[:, check_x]
                white_pixels = np.where(column > 127)[0]
                
                if len(white_pixels) == 0:
                    break_positions.append(check_x)
                else:
                    # Check for gaps
                    for i in range(1, len(white_pixels)):
                        gap_size = white_pixels[i] - white_pixels[i-1] - 1
                        if gap_size > 15:  # Larger threshold for nearby columns
                            break_positions.append(check_x)
                            break
        
        # If we found break positions, use the center one
        if break_positions:
            break_detected = True
            # Use the middle position of all detected breaks
            center_idx = len(break_positions) // 2
            break_x = break_positions[center_idx]
            
            # Find break_y for the center position
            column = mask[:, break_x]
            white_pixels = np.where(column > 127)[0]
            
            if len(white_pixels) == 0:
                break_y = height // 2
            else:
                # Find the largest gap in the center column
                max_gap_size = 0
                gap_center_y = height // 2
                for i in range(1, len(white_pixels)):
                    gap_size = white_pixels[i] - white_pixels[i-1] - 1
                    if gap_size > max_gap_size:
                        max_gap_size = gap_size
                        gap_center_y = (white_pixels[i-1] + white_pixels[i]) // 2
                break_y = gap_center_y
            
            print(f"Break detected across {len(break_positions)} pixels, using center at x={break_x}")
    
    return narrowest_index, narrowest_height, break_detected, break_x, break_y

def find_narrowest_point(mask: np.ndarray) -> int:
    """Backward compatibility - returns only the narrowest point x-coordinate"""
    narrowest_x, narrowest_height, break_detected, break_x, break_y = find_narrowest_point_and_break(mask) 
    print(break_detected, narrowest_x)
    return narrowest_x

def moving_average(zs, window_len):
    # zs: measurements (distance)
    weights = np.ones(window_len) / window_len
    zs_smooth = np.convolve(zs, weights, mode='valid')
    return zs_smooth

def check_end_of_expansion(horizontal_distances: list,
                     smooth_distances: list,
                     distance_window_len: int,
                     velocities: list,
                     velocity_window_len: int,
                     cap: cv.VideoCapture) -> int:
    if len(horizontal_distances) > distance_window_len:
        smooth_distances.append(moving_average(horizontal_distances[-distance_window_len-1:-1],
                                  distance_window_len))
        
    if len(smooth_distances) > 2:
        velocity = smooth_distances[-1] - smooth_distances[-2]
        velocities.append(velocity[0])

    if len(velocities) > velocity_window_len:
        smooth_velocity = moving_average(velocities[-velocity_window_len-1:-1],
                                              velocity_window_len)[0]

    if horizontal_distances[-1] > horizontal_distances[0] * 1.5 and np.abs(smooth_velocity) < 0.5:
            end_of_expansion_frame = cap.get(cv.CAP_PROP_POS_FRAMES) - (velocity_window_len // 2 + distance_window_len //2)
            print(f'End of expansion detected at frame: {end_of_expansion_frame}')
            return end_of_expansion_frame
    return None

def check_break_moment(contours: list, cap: cv.VideoCapture) -> int:
    if len(contours) > 1:
        areas = [cv.contourArea(cnt) for cnt in contours]
        largest_area = max(areas)
        areas.remove(largest_area)
        second_largest_area = max(areas) if areas else 0
        if second_largest_area > largest_area * 0.5:
            break_frame = cap.get(cv.CAP_PROP_POS_FRAMES)
            print(f'Break detected at frame: {break_frame}')
            return break_frame
    return None

# spacją se możesz przewinąć do przodu o klatkę
def get_breakpoint(break_moment_frame: int, cap: cv.VideoCapture,
                   model: FluidSegmentationModel=None,
                   device: str='cpu',
                   transform=None):
    cap.set(cv.CAP_PROP_POS_FRAMES, break_moment_frame-10)
    disp_number = -11
    while True:
        ret, frame = cap.read()
        
        if ret:
            orig_frame = frame.copy()
            orig_frame = cv.resize(orig_frame, (512, 512))
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frame = transform(image=frame)['image'].unsqueeze(0)  # Add batch dimension
            frame = frame.to(device)

            with torch.inference_mode():
                pred_mask = torch.sigmoid(model(frame))
            pred_mask = (pred_mask > 0.5).float()
            pred_mask = pred_mask.squeeze().cpu().numpy()  
            pred_mask = (pred_mask * 255).astype(np.uint8)
            pred_mask = cv.cvtColor(pred_mask, cv.COLOR_GRAY2BGR)
            
            disp_number += 1 # ten numer to chyba nie do końca jest prawidłowy

            cv.putText(orig_frame,
                       f'Frame: {disp_number} relative to break',
                       (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX,
                       1,
                       (255, 0, 255),
                       2)
            combined = cv.hconcat([orig_frame, pred_mask])

            narrowest_point = find_narrowest_point(pred_mask)
            cv.line(combined, (narrowest_point, 0), (narrowest_point, combined.shape[0]), (0, 0, 255), 2)


            cv.imshow('Last frame', combined)
            if cap.get(cv.CAP_PROP_POS_FRAMES) >= break_moment_frame + 10:
                cap.set(cv.CAP_PROP_POS_FRAMES, break_moment_frame-10)
                disp_number = -11 
            key_code = cv.waitKey(0)
            if key_code == 27:  # ESC key to exit
                cv.destroyAllWindows()
                break
            elif key_code == ord(' '):
                continue

def measure_height_at_break_progression(end_frame: int, break_frame: int, cap: cv.VideoCapture, 
                                       model, transform, device) -> dict:
    """Measure height progression at the break point from end_of_expansion to break detection"""
    results = {
        'frames': [],
        'heights': [],
        'break_x': None,
        'break_frame': None
    }
    
    # First, scan through frames to find where the break actually occurs
    # Only search in the 10 frames before the contour-based break detection
    search_start_frame = max(end_frame, break_frame - 10)
    cap.set(cv.CAP_PROP_POS_FRAMES, search_start_frame)
    break_x_coordinate = None
    
    print(f"Scanning frames {search_start_frame} to {break_frame} to find the precise break point...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        current_frame = int(cap.get(cv.CAP_PROP_POS_FRAMES) - 1)
        
        # Stop when we reach the original break moment detection
        if current_frame >= break_frame:
            break
            
        # Process frame
        orig_frame = frame.copy()
        orig_frame = cv.resize(orig_frame, (512, 512))
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = transform(image=frame)['image'].unsqueeze(0)
        frame = frame.to(device)

        with torch.inference_mode():
            pred_mask = torch.sigmoid(model(frame))
        pred_mask = (pred_mask > 0.5).float()
        pred_mask = pred_mask.squeeze().cpu().numpy()
        pred_mask = (pred_mask * 255).astype(np.uint8)
        pred_mask_3ch = cv.cvtColor(pred_mask, cv.COLOR_GRAY2BGR)
        
        # Check for break using enhanced detection
        narrowest_x, narrowest_height, break_detected, break_x, break_y = find_narrowest_point_and_break(pred_mask_3ch)
        
        # If break detected by enhanced function, record this as the break point
        if break_detected and break_x_coordinate is None:
            break_x_coordinate = break_x
            results['break_x'] = break_x_coordinate
            results['break_frame'] = current_frame
            print(f"PRECISE BREAK DETECTED at frame {current_frame}, x-coordinate: {break_x_coordinate}")
            break
    
    # If no break was found during the limited scan, use the narrowest point from a frame near the break
    if break_x_coordinate is None:
        print("No precise break detected in last 10 frames, using narrowest point from frame near break")
        cap.set(cv.CAP_PROP_POS_FRAMES, break_frame - 5)  # Use frame 5 frames before contour break
        ret, frame = cap.read()
        if ret:
            orig_frame = frame.copy()
            orig_frame = cv.resize(orig_frame, (512, 512))
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frame = transform(image=frame)['image'].unsqueeze(0)
            frame = frame.to(device)

            with torch.inference_mode():
                pred_mask = torch.sigmoid(model(frame))
            pred_mask = (pred_mask > 0.5).float()
            pred_mask = pred_mask.squeeze().cpu().numpy()
            pred_mask = (pred_mask * 255).astype(np.uint8)
            pred_mask_3ch = cv.cvtColor(pred_mask, cv.COLOR_GRAY2BGR)
            
            narrowest_x, _, _, _, _ = find_narrowest_point_and_break(pred_mask_3ch)
            break_x_coordinate = narrowest_x
            results['break_x'] = break_x_coordinate
            print(f"Using narrowest point x={break_x_coordinate} from frame {break_frame - 5}")
    
    # Now measure height at the fixed break x-coordinate for all frames
    cap.set(cv.CAP_PROP_POS_FRAMES, end_frame)
    
    print(f"Measuring height progression at break x-coordinate: {break_x_coordinate}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        current_frame = int(cap.get(cv.CAP_PROP_POS_FRAMES) - 1)
        
        # Stop when we reach the original break moment detection or actual break frame
        stop_frame = results['break_frame'] if results['break_frame'] else break_frame
        if current_frame >= stop_frame:
            break
            
        # Process frame
        orig_frame = frame.copy()
        orig_frame = cv.resize(orig_frame, (512, 512))
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = transform(image=frame)['image'].unsqueeze(0)
        frame = frame.to(device)

        with torch.inference_mode():
            pred_mask = torch.sigmoid(model(frame))
        pred_mask = (pred_mask > 0.5).float()
        pred_mask = pred_mask.squeeze().cpu().numpy()
        pred_mask = (pred_mask * 255).astype(np.uint8)
        
        # Measure height at the fixed break x-coordinate
        column = pred_mask[:, break_x_coordinate]
        white_pixels = np.where(column > 127)[0]
        height = len(white_pixels) if len(white_pixels) > 0 else 0
        
        # Find top and bottom of the fluid at break x-coordinate for visualization
        top_y = white_pixels[0] if len(white_pixels) > 0 else -1
        bottom_y = white_pixels[-1] if len(white_pixels) > 0 else -1
        
        results['frames'].append(current_frame)
        results['heights'].append(height)
        
        # VISUALIZATION: Draw the measurement on the frame
        # Convert mask to 3-channel for visualization
        pred_mask_vis = cv.cvtColor(pred_mask, cv.COLOR_GRAY2BGR)
        
        # Draw vertical line at break x-coordinate
        cv.line(orig_frame, (break_x_coordinate, 0), (break_x_coordinate, orig_frame.shape[0]), (0, 0, 255), 2)
        cv.line(pred_mask_vis, (break_x_coordinate, 0), (break_x_coordinate, pred_mask_vis.shape[0]), (0, 0, 255), 2)
        
        # Draw measurement line if fluid exists
        if top_y >= 0 and bottom_y >= 0:
            # Draw thick measurement line on both images
            cv.line(orig_frame, (break_x_coordinate, top_y), (break_x_coordinate, bottom_y), (0, 255, 0), 4)
            cv.line(pred_mask_vis, (break_x_coordinate, top_y), (break_x_coordinate, bottom_y), (0, 255, 0), 4)
            
            # Draw markers at top and bottom
            cv.circle(orig_frame, (break_x_coordinate, top_y), 3, (255, 255, 0), -1)
            cv.circle(orig_frame, (break_x_coordinate, bottom_y), 3, (255, 255, 0), -1)
            cv.circle(pred_mask_vis, (break_x_coordinate, top_y), 3, (255, 255, 0), -1)
            cv.circle(pred_mask_vis, (break_x_coordinate, bottom_y), 3, (255, 255, 0), -1)
        
        # Add text information
        cv.putText(orig_frame, f'Frame: {current_frame}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv.putText(orig_frame, f'Break X: {break_x_coordinate}', (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv.putText(orig_frame, f'Height: {height}px', (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv.putText(orig_frame, f'Progress: {len(results["frames"])}/{stop_frame-end_frame}', (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # Combine original and mask
        combined = cv.hconcat([orig_frame, pred_mask_vis])
        cv.imshow('Height Measurement at Break Point', combined)
        
        print(f"Frame {current_frame}: Height at break x={break_x_coordinate} = {height}px")
        
        # Allow user to control playback speed
        key = cv.waitKey(1) & 0xFF
        if key == 27:  # ESC to exit
            break
        elif key == ord(' '):  # SPACE to pause
            cv.waitKey(0)
    
    cv.destroyWindow('Height Measurement at Break Point')
    return results

def measure_height_at_mask_center(end_frame: int, cap: cv.VideoCapture, 
                                 model, transform, device) -> dict:
    """Measure height progression at the center of the mask when no break is detected"""
    results = {
        'frames': [],
        'heights': [],
        'center_x': None,
        'analysis_type': 'center'
    }
    
    # First, find the center x-coordinate of the mask from a representative frame
    cap.set(cv.CAP_PROP_POS_FRAMES, end_frame + 5)  # Use a frame shortly after expansion ends
    ret, frame = cap.read()
    
    if ret:
        orig_frame = frame.copy()
        orig_frame = cv.resize(orig_frame, (512, 512))
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = transform(image=frame)['image'].unsqueeze(0)
        frame = frame.to(device)

        with torch.inference_mode():
            pred_mask = torch.sigmoid(model(frame))
        pred_mask = (pred_mask > 0.5).float()
        pred_mask = pred_mask.squeeze().cpu().numpy()
        pred_mask = (pred_mask * 255).astype(np.uint8)
        
        # Find the center x-coordinate of the mask
        column_sums = pred_mask.sum(axis=0)
        mask_columns = np.where(column_sums > 0)[0]
        
        if len(mask_columns) > 0:
            center_x_coordinate = (mask_columns[0] + mask_columns[-1]) // 2
            results['center_x'] = center_x_coordinate
            print(f"Using mask center x-coordinate: {center_x_coordinate}")
        else:
            print("No mask found! Using image center")
            center_x_coordinate = pred_mask.shape[1] // 2
            results['center_x'] = center_x_coordinate
    else:
        print("Could not read frame! Using image center")
        center_x_coordinate = 256  # Default for 512x512 image
        results['center_x'] = center_x_coordinate
    
    # Get total frame count to measure until end of video
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    
    # Now measure height at the center x-coordinate for all frames from end_frame to end of video
    cap.set(cv.CAP_PROP_POS_FRAMES, end_frame)
    
    print(f"Measuring height progression at mask center x-coordinate: {center_x_coordinate}")
    print(f"From frame {end_frame} to end of video (frame {total_frames})")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        current_frame = int(cap.get(cv.CAP_PROP_POS_FRAMES) - 1)
        
        # Process frame
        orig_frame = frame.copy()
        orig_frame = cv.resize(orig_frame, (512, 512))
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = transform(image=frame)['image'].unsqueeze(0)
        frame = frame.to(device)

        with torch.inference_mode():
            pred_mask = torch.sigmoid(model(frame))
        pred_mask = (pred_mask > 0.5).float()
        pred_mask = pred_mask.squeeze().cpu().numpy()
        pred_mask = (pred_mask * 255).astype(np.uint8)
        
        # Measure height at the center x-coordinate
        column = pred_mask[:, center_x_coordinate]
        white_pixels = np.where(column > 127)[0]
        height = len(white_pixels) if len(white_pixels) > 0 else 0
        
        # Find top and bottom of the fluid at center x-coordinate for visualization
        top_y = white_pixels[0] if len(white_pixels) > 0 else -1
        bottom_y = white_pixels[-1] if len(white_pixels) > 0 else -1
        
        results['frames'].append(current_frame)
        results['heights'].append(height)
        
        # VISUALIZATION: Draw the measurement on the frame
        # Convert mask to 3-channel for visualization
        pred_mask_vis = cv.cvtColor(pred_mask, cv.COLOR_GRAY2BGR)
        
        # Draw vertical line at center x-coordinate
        cv.line(orig_frame, (center_x_coordinate, 0), (center_x_coordinate, orig_frame.shape[0]), (255, 0, 0), 2)
        cv.line(pred_mask_vis, (center_x_coordinate, 0), (center_x_coordinate, pred_mask_vis.shape[0]), (255, 0, 0), 2)
        
        # Draw measurement line if fluid exists
        if top_y >= 0 and bottom_y >= 0:
            # Draw thick measurement line on both images
            cv.line(orig_frame, (center_x_coordinate, top_y), (center_x_coordinate, bottom_y), (0, 255, 0), 4)
            cv.line(pred_mask_vis, (center_x_coordinate, top_y), (center_x_coordinate, bottom_y), (0, 255, 0), 4)
            
            # Draw markers at top and bottom
            cv.circle(orig_frame, (center_x_coordinate, top_y), 3, (255, 255, 0), -1)
            cv.circle(orig_frame, (center_x_coordinate, bottom_y), 3, (255, 255, 0), -1)
            cv.circle(pred_mask_vis, (center_x_coordinate, top_y), 3, (255, 255, 0), -1)
            cv.circle(pred_mask_vis, (center_x_coordinate, bottom_y), 3, (255, 255, 0), -1)
        
        # Add text information
        cv.putText(orig_frame, f'Frame: {current_frame}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv.putText(orig_frame, f'Center X: {center_x_coordinate}', (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv.putText(orig_frame, f'Height: {height}px', (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv.putText(orig_frame, f'Progress: {len(results["frames"])}/{total_frames-end_frame}', (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv.putText(orig_frame, f'NO BREAK - Center Analysis', (10, 150), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Combine original and mask
        combined = cv.hconcat([orig_frame, pred_mask_vis])
        cv.imshow('Height Measurement at Mask Center', combined)
        
        print(f"Frame {current_frame}: Height at center x={center_x_coordinate} = {height}px")
        
        # Allow user to control playback speed
        key = cv.waitKey(1) & 0xFF
        if key == 27:  # ESC to exit
            break
        elif key == ord(' '):  # SPACE to pause
            cv.waitKey(0)
    
    cv.destroyWindow('Height Measurement at Mask Center')
    return results

def main():
    video_path = 'video/problematyczny1.MP4'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load('best_model/best_model.pth', map_location=device)
    encoder_name = checkpoint.get('encoder_name', 'resnet101')
    print(f"Using encoder: {encoder_name}")
    model = create_model(encoder_name=encoder_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    cap = cv.VideoCapture(video_path)
    transform = get_validation_transforms(image_size=(512, 512))
    
    horizontal_distances = []
    smooth_distances = []
    distance_window_len = 7
    velocity_window_len = 5
    velocities = []

    end_of_expansion_frame = None
    break_moment_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        orig_frame = frame.copy()
        orig_frame = cv.resize(orig_frame, (512, 512))
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = transform(image=frame)['image'].unsqueeze(0)  # Add batch dimension
        frame = frame.to(device)

        with torch.inference_mode():
            pred_mask = torch.sigmoid(model(frame))
        pred_mask = (pred_mask > 0.5).float()
        pred_mask = pred_mask.squeeze().cpu().numpy()  # Remove batch dimension and convert to numpy
        pred_mask = (pred_mask * 255).astype(np.uint8)
        
        contours = cv.findContours(pred_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
        largest_contour = max(contours, key=cv.contourArea) if contours else None

        max_left = largest_contour[:, 0, 0].min()
        max_right = largest_contour[:, 0, 0].max()

        horizontal_distance = max_right - max_left
        horizontal_distances.append(horizontal_distance)

        if not end_of_expansion_frame:
            end_of_expansion_frame = check_end_of_expansion(horizontal_distances,
                                                    smooth_distances,
                                                    distance_window_len,
                                                    velocities,
                                                    velocity_window_len,
                                                    cap)
            
        if end_of_expansion_frame:
            break_moment_frame = check_break_moment(contours, cap)
        
        if break_moment_frame:
            cv.destroyAllWindows()
            cv.imshow('Break Moment Frame', orig_frame)
            cv.waitKey(0)
            break

        cv.drawContours(orig_frame, [largest_contour], -1, (0, 255, 0), 2) if largest_contour is not None else None
        pred_mask = cv.cvtColor(pred_mask, cv.COLOR_GRAY2BGR)
        combined = cv.hconcat([orig_frame, pred_mask])
        cv.imshow('Segmentation Inference', combined)

        key_code = cv.waitKey(1)  
        if key_code == 27:  
            cv.destroyAllWindows()
            break

    # NEW: Analyze break progression with enhanced detection
    if end_of_expansion_frame and break_moment_frame:
        print(f"\n=== ANALYZING BREAK PROGRESSION ===")
        print(f"From frame {end_of_expansion_frame} to {break_moment_frame}")
        
        height_analysis = measure_height_at_break_progression(end_of_expansion_frame, 
                                                             break_moment_frame, 
                                                             cap, model, transform, device)
        
        if height_analysis['frames']:
            # Create real-time visualization during measurement
            print("Creating final analysis plot...")
            
            # Create visualization
            plt.figure(figsize=(12, 6))
            
            # Mark the actual break detection point
            break_frame = height_analysis['break_frame']

            plt.scatter(height_analysis['frames'], height_analysis['heights'], c='b', s=2, label='Height at break point')

            if break_frame:
                break_idx = height_analysis['frames'].index(break_frame) if break_frame in height_analysis['frames'] else -1
                if break_idx >= 0:
                    plt.axvline(x=break_frame, color='red', linestyle='--', linewidth=2, label=f'Break detected (frame {break_frame})')
                    plt.plot(break_frame, height_analysis['heights'][break_idx], 'ro', markersize=10, label='Break point')
            
            plt.title(f'Fluid Height Evolution at x={height_analysis["break_x"]} (Break Point)')
            plt.xlabel('Frame Number')
            plt.ylabel('Height (pixels)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # # Add statistics
            avg_height = np.mean(height_analysis['heights'])
            min_height = min(height_analysis['heights'])
            max_height = max(height_analysis['heights'])
            
            # stats_text = f'Stats: Avg={avg_height:.1f}px, Min={min_height}px, Max={max_height}px'
            # plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
            #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            #         verticalalignment='top')
            
            plt.tight_layout()
            plt.savefig('height_progression_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Save data to CSV
            np.savetxt('height_progression_data.csv', 
                      np.column_stack([height_analysis['frames'], 
                                     height_analysis['heights']]),
                      delimiter=',',
                      header='Frame,Height_at_break_point',
                      comments='')
            print(f"Data saved to height_progression_data.csv")
            
            # Print summary
            print(f"\n=== SUMMARY ===")
            print(f"Break x-coordinate: x = {height_analysis['break_x']}")
            print(f"Frames analyzed: {len(height_analysis['frames'])}")
            print(f"Break detected at frame: {break_frame}")
            print(f"Height range: {min_height} - {max_height} pixels")
            print(f"Average height: {avg_height:.1f} pixels")
    
    # Handle case when no break is detected but end of expansion is found
    elif end_of_expansion_frame and not break_moment_frame:
        print(f"\n=== NO BREAK DETECTED - ANALYZING MASK CENTER ===")
        print(f"From frame {end_of_expansion_frame} to end of video")
        
        height_analysis = measure_height_at_mask_center(end_of_expansion_frame, 
                                                       cap, model, transform, device)
        
        if height_analysis['frames']:
            # Create visualization for center analysis
            print("Creating center analysis plot...")
            
            plt.figure(figsize=(12, 6))
            
            plt.scatter(height_analysis['frames'], height_analysis['heights'], c='g', s=2, label='Height at mask center')
            
            plt.title(f'Fluid Height Evolution at x={height_analysis["center_x"]} (Mask Center - No Break)')
            plt.xlabel('Frame Number')
            plt.ylabel('Height (pixels)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Add statistics
            avg_height = np.mean(height_analysis['heights'])
            min_height = min(height_analysis['heights'])
            max_height = max(height_analysis['heights'])
            
            plt.tight_layout()
            plt.savefig('height_progression_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Save data to CSV
            np.savetxt('height_progression_data.csv', 
                      np.column_stack([height_analysis['frames'], 
                                     height_analysis['heights']]),
                      delimiter=',',
                      header='Frame,Height_at_mask_center',
                      comments='')
            print(f"Data saved to height_progression_data.csv")
            
            # Print summary
            print(f"\n=== SUMMARY (NO BREAK DETECTED) ===")
            print(f"Mask center x-coordinate: x = {height_analysis['center_x']}")
            print(f"Frames analyzed: {len(height_analysis['frames'])}")
            print(f"Analysis type: Center of mask (no break detected)")
            print(f"Height range: {min_height} - {max_height} pixels")
            print(f"Average height: {avg_height:.1f} pixels")
    
    # Handle case when no end of expansion is detected at all
    elif not end_of_expansion_frame:
        print(f"\n=== NO END OF EXPANSION DETECTED ===")
        print("The fluid may still be expanding or the analysis parameters need adjustment.")
        print("No height analysis will be performed.")
            
    if end_of_expansion_frame:
        get_breakpoint(break_moment_frame, cap, model, device, transform)

        cap.set(cv.CAP_PROP_POS_FRAMES, end_of_expansion_frame)
        while True:
            ret, frame = cap.read()
            key_code = cv.waitKey(1)
            if not ret or key_code == 27:
                break

            if cap.get(cv.CAP_PROP_POS_FRAMES) == break_moment_frame:
                cap.set(cv.CAP_PROP_POS_FRAMES, end_of_expansion_frame)


            orig_frame = frame.copy()
            orig_frame = cv.resize(orig_frame, (512, 512))
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frame = transform(image=frame)['image'].unsqueeze(0)  # Add batch dimension
            frame = frame.to(device)

            with torch.inference_mode():
                pred_mask = torch.sigmoid(model(frame))
            pred_mask = (pred_mask > 0.5).float()
            pred_mask = pred_mask.squeeze().cpu().numpy()  # Remove batch dimension and convert to numpy
            pred_mask = (pred_mask * 255).astype(np.uint8)
        
            contours = cv.findContours(pred_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]
            cv.drawContours(orig_frame, contours, -1, (0, 255, 0), 2) if contours else None

            pred_mask = cv.cvtColor(pred_mask, cv.COLOR_GRAY2BGR)
            
            # Enhanced break analysis for post-expansion
            narrowest_x, narrowest_height, break_detected, break_x, break_y = find_narrowest_point_and_break(pred_mask)
            
            # Draw narrowest point line
            cv.line(pred_mask, (narrowest_x, 0), (narrowest_x, pred_mask.shape[0]), (0, 0, 255), 2)
            
            # If break detected, highlight it
            if break_detected:
                cv.circle(orig_frame, (break_x, break_y), 8, (0, 255, 255), 2)
                cv.line(orig_frame, (0, break_y), (orig_frame.shape[1], break_y), (0, 255, 255), 1)
                
                current_frame_num = int(cap.get(cv.CAP_PROP_POS_FRAMES) - 1)
                cv.putText(orig_frame, f'Frame: {current_frame_num}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv.putText(orig_frame, f'BREAK DETECTED!', (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv.putText(orig_frame, f'Height: {narrowest_height}px', (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                current_frame_num = int(cap.get(cv.CAP_PROP_POS_FRAMES) - 1)
                cv.putText(orig_frame, f'Frame: {current_frame_num}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv.putText(orig_frame, f'Height: {narrowest_height}px', (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            combined = cv.hconcat([orig_frame, pred_mask])
            cv.imshow('Post Expansion', combined)


    cap.release()
    cv.destroyAllWindows()
    
if __name__ == "__main__":
    main()