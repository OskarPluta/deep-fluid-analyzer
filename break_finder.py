import sys
import cv2 as cv
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


sys.path.append(str(Path(__file__).parent / 'src'))
from dataset import get_validation_transforms
from model import create_model

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
                     horizontal_distance: float,
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

    if horizontal_distance > horizontal_distances[0] * 1.5 and np.abs(smooth_velocity) < 0.5:
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

def find_break_point(mask: np.ndarray) -> int:
    """Find the y-coordinate where the fluid mask breaks/separates"""
    height, width = mask.shape
    
    # Scan from top to bottom to find where the continuous fluid stream breaks
    for y in range(height):
        row = mask[y, :]
        # Find white pixels (fluid) in this row
        white_pixels = np.where(row > 127)[0]
        
        if len(white_pixels) > 0:
            # Check for significant gaps in the fluid (indicating a break)
            gaps = []
            if len(white_pixels) > 1:
                for i in range(1, len(white_pixels)):
                    gap_size = white_pixels[i] - white_pixels[i-1] - 1
                    if gap_size > 5:  # Minimum gap size to consider as break
                        gaps.append(gap_size)
                
                # If we find significant gaps, this is the break point
                if gaps and max(gaps) > 10:  # Threshold for break detection
                    return y
    
    return -1  # No break found

def measure_vertical_height_at_break(mask: np.ndarray, break_y: int) -> tuple:
    """Measure the vertical height of fluid at the breaking point"""
    if break_y < 0 or break_y >= mask.shape[0]:
        return 0, -1, -1
    
    height, width = mask.shape
    
    # Find the x-coordinate of the break point (center of the gap)
    row = mask[break_y, :]
    white_pixels = np.where(row > 127)[0]
    
    if len(white_pixels) == 0:
        return 0, -1, -1
    
    # Find the center x-coordinate where the break occurs
    break_x = width // 2  # Default to center
    if len(white_pixels) > 1:
        # Find the largest gap
        max_gap = 0
        gap_center = width // 2
        for i in range(1, len(white_pixels)):
            gap_size = white_pixels[i] - white_pixels[i-1] - 1
            if gap_size > max_gap:
                max_gap = gap_size
                gap_center = (white_pixels[i-1] + white_pixels[i]) // 2
        break_x = gap_center
    
    # Measure vertical height at the break x-coordinate
    column = mask[:, break_x]
    white_y_pixels = np.where(column > 127)[0]
    
    if len(white_y_pixels) == 0:
        return 0, -1, -1
    
    top_y = white_y_pixels[0]
    bottom_y = white_y_pixels[-1]
    vertical_height = bottom_y - top_y + 1
    
    return vertical_height, top_y, bottom_y

def analyze_break_progression(end_frame: int, break_frame: int, cap: cv.VideoCapture, 
                            model, transform, device) -> dict:
    """Analyze break progression and measure vertical height at break point"""
    results = {
        'frames': [],
        'vertical_heights': []
    }
    
    print(f"Analyzing vertical height from frame {end_frame} to {break_frame}")
    
    # Go to end of expansion frame
    cap.set(cv.CAP_PROP_POS_FRAMES, end_frame)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        current_frame = cap.get(cv.CAP_PROP_POS_FRAMES) - 1
        
        # Stop when we reach break moment
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
        
        # Find break point
        break_y = find_break_point(pred_mask)
        
        # Measure vertical height at break point
        if break_y >= 0:
            vertical_height, top_y, bottom_y = measure_vertical_height_at_break(pred_mask, break_y)
            
            results['frames'].append(current_frame)
            results['vertical_heights'].append(vertical_height)
            
            print(f"Frame {current_frame}: Vertical height = {vertical_height} pixels")
        else:
            print(f"Frame {current_frame}: No break detected")
    
    return results


def main():
    video_path = 'video/n2.MP4'
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
                                                    horizontal_distance,
                                                    cap)
            
        if end_of_expansion_frame:
            break_moment_frame = check_break_moment(contours, cap)
        
        if break_moment_frame:
            break

        # cv.drawContours(orig_frame, [largest_contour], -1, (0, 255, 0), 2) if largest_contour is not None else None
        # pred_mask = cv.cvtColor(pred_mask, cv.COLOR_GRAY2BGR)
        # combined = cv.hconcat([orig_frame, pred_mask])
        # cv.imshow('Segmentation Inference', combined)

        key_code = cv.waitKey(1)  
        if key_code == 27:  
            break

    # Analyze break progression if both frames are detected
    if end_of_expansion_frame and break_moment_frame:
        print(f"\nAnalyzing break from frame {end_of_expansion_frame} to {break_moment_frame}")
        break_analysis = analyze_break_progression(end_of_expansion_frame, break_moment_frame, 
                                                 cap, model, transform, device)
        
        # Create plots to visualize the results
        if break_analysis['frames']:
            plt.figure(figsize=(10, 6))
            
            plt.plot(break_analysis['frames'], break_analysis['vertical_heights'], 'r-o', linewidth=2, markersize=6)
            plt.title('Vertical Height at Break Point vs Frame', fontsize=14)
            plt.xlabel('Frame Number', fontsize=12)
            plt.ylabel('Vertical Height (pixels)', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Add summary statistics as text
            avg_height = np.mean(break_analysis['vertical_heights'])
            min_height = np.min(break_analysis['vertical_heights'])
            max_height = np.max(break_analysis['vertical_heights'])
            
            plt.text(0.02, 0.98, f"Frames analyzed: {len(break_analysis['frames'])}", 
                    transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            plt.text(0.02, 0.88, f"Average height: {avg_height:.1f} px", 
                    transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            plt.text(0.02, 0.78, f"Range: {min_height:.1f} - {max_height:.1f} px", 
                    transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig('vertical_height_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Save results to file
            np.savetxt('break_analysis_results.csv', 
                      np.column_stack([break_analysis['frames'], 
                                     break_analysis['vertical_heights']]),
                      delimiter=',',
                      header='Frame,Vertical_Height',
                      comments='')
            print("Results saved to 'break_analysis_results.csv'")

    if end_of_expansion_frame:
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
        
            contours = cv.findContours(pred_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
            cv.drawContours(orig_frame, contours, -1, (0, 255, 0), 2) if contours else None
            
            # Find and visualize break point
            break_y = find_break_point(pred_mask)
            if break_y >= 0:
                vertical_height, top_y, bottom_y = measure_vertical_height_at_break(pred_mask, break_y)
                
                # Find break x-coordinate
                row = pred_mask[break_y, :]
                white_pixels = np.where(row > 127)[0]
                break_x = pred_mask.shape[1] // 2
                if len(white_pixels) > 1:
                    max_gap = 0
                    for i in range(1, len(white_pixels)):
                        gap_size = white_pixels[i] - white_pixels[i-1] - 1
                        if gap_size > max_gap:
                            max_gap = gap_size
                            break_x = (white_pixels[i-1] + white_pixels[i]) // 2
                
                # Draw only the vertical height line
                cv.line(orig_frame, (break_x, top_y), (break_x, bottom_y), (0, 0, 255), 3)  # Vertical height line
                
                # Add only height information
                current_frame_num = int(cap.get(cv.CAP_PROP_POS_FRAMES) - 1)
                cv.putText(orig_frame, f'Frame: {current_frame_num}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv.putText(orig_frame, f'Height: {vertical_height}px', (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            pred_mask = cv.cvtColor(pred_mask, cv.COLOR_GRAY2BGR)
            combined = cv.hconcat([orig_frame, pred_mask])
            cv.imshow('Post Expansion', combined)


    cap.release()
    cv.destroyAllWindows()
    
if __name__ == "__main__":
    main()