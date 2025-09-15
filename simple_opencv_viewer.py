#!/usr/bin/env python3
import sys
import os
import argparse
import numpy as np
import cv2

# Import local modules
from video_processing import VideoLoader

def simple_mask_viewer(data_path, load_original_frames=False):
    """
    Simple function to load and view fluid mask data
    
    Args:
        data_path: Path to .npz file with mask data
        load_original_frames: Whether to load original frames
    """
    # Load the data
    print(f"Loading data from {data_path}")
    video_data = VideoLoader.load_video_data(data_path, load_original_frames)
    
    # Get basic info
    total_frames = len(video_data['masks'])
    height, width = video_data['masks'][0].shape
    has_frames = 'frames' in video_data
    
    # Print mask info
    print(f"Loaded {total_frames} frames, size: {width}x{height}")
    print(f"Original frames available: {has_frames}")
    
    # Setup display window
    window_name = "Fluid Mask Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    
    # Set up variables for the main loop
    current_frame = 0
    playing = False
    fps = video_data.get('original_fps', 30)
    colormap = cv2.COLORMAP_JET  # Default colormap
    
    print("\nControls:")
    print("Space - Play/Pause")
    print("Left/Right - Previous/Next frame")
    print("C - Toggle colormap")
    print("O - Toggle overlay mode (if original frames available)")
    print("ESC or Q - Quit")
    
    # Main display loop
    overlay_mode = has_frames
    use_color = True
    last_time = cv2.getTickCount()
    
    while True:
        # Get the current frame data
        frame_data = VideoLoader.get_frame_data(video_data, current_frame)
        mask = frame_data['mask']
        # mask = 
        
        
        # Create display frame
        if overlay_mode and has_frames:
            # Show mask overlaid on original frame
            display = frame_data['frame'].copy()
            
            # Apply colored mask overlay
            if use_color:
                mask_colored = cv2.applyColorMap((mask * 255).astype(np.uint8), colormap)
                mask_area = mask > 0
                for c in range(3):
                    display[:, :, c] = np.where(
                        mask_area,
                        display[:, :, c] * 0.3 + mask_colored[:, :, c] * 0.7,
                        display[:, :, c]
                    )
            else:
                # Simple red overlay
                mask_area = mask > 0
                display[mask_area] = [0, 0, 255]  # BGR format
        else:
            # Show just the mask
            if use_color:
                # Apply colormap to mask
                display = cv2.applyColorMap((mask * 255).astype(np.uint8), colormap)
            else:
                # Grayscale mask
                display = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        
        # Add simple info text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display, f"Frame: {current_frame+1}/{total_frames}", 
                   (10, 30), font, 0.7, (255, 255, 255), 2)
        
        fluid_pixels = np.sum(mask)
        fluid_percentage = (fluid_pixels / (height * width)) * 100
        cv2.putText(display, f"Fluid: {int(fluid_pixels)} pixels ({fluid_percentage:.2f}%)", 
                   (10, 60), font, 0.7, (255, 255, 255), 2)
                   
        # Show the frame
        cv2.imshow(window_name, display)
        
        # Handle frame timing for playback
        if playing:
            current_time = cv2.getTickCount()
            elapsed = (current_time - last_time) / cv2.getTickFrequency()
            wait_time = max(1, int((1.0/fps - elapsed) * 1000))
        else:
            wait_time = 0  # Wait indefinitely
            
        # Check for key presses
        key = cv2.waitKey(wait_time)
        
        # Update for next frame if playing
        if playing:
            current_frame = (current_frame + 1) % total_frames
            last_time = cv2.getTickCount()
        
        # Handle key presses
        if key == -1:  # No key pressed
            continue
            
        if key == 27 or key == ord('q'):  # ESC or Q
            break
            
        elif key == ord(' '):  # Space
            playing = not playing
            last_time = cv2.getTickCount()
            
        elif key == 81 or key == ord('a'):  # Left arrow or A
            playing = False
            current_frame = (current_frame - 1) % total_frames
            
        elif key == 83 or key == ord('d'):  # Right arrow or D
            playing = False
            current_frame = (current_frame + 1) % total_frames
            
        elif key == ord('c'):  # C key
            # Toggle color mode
            use_color = not use_color
            
        elif key == ord('o') and has_frames:  # O key
            # Toggle overlay mode
            overlay_mode = not overlay_mode
            
        elif key == ord('s'):  # S key
            # Save current frame
            filename = f"frame_{current_frame}.png"
            cv2.imwrite(filename, display)
            print(f"Saved current frame to {filename}")
        
        # Add your custom function calls here
        # This is where you can add your own processing
        # Example:
        # if key == ord('p'):
        #     process_current_frame(mask, frame_data)
    
    cv2.destroyAllWindows()
    print("Viewer closed")
    
    return video_data  # Return the data for further processing if needed

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Simple OpenCV-based mask viewer')
    parser.add_argument('--data_path', help='Path to .npz file with fluid mask data')
    parser.add_argument('--no-frames', action='store_true', help='Skip loading original frames')
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.data_path):
        print(f"Error: File {args.data_path} not found")
        return 1
    
    simple_mask_viewer(args.data_path, load_original_frames=not args.no_frames)
    return 0

if __name__ == "__main__":
    sys.exit(main())
