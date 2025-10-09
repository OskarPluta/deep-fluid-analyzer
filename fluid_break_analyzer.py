import sys
import cv2 as cv
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict, List, Optional


sys.path.append(str(Path(__file__).parent / 'src'))
from dataset import get_validation_transforms
from model import create_model, FluidSegmentationModel


class FluidBreakAnalyzer:
    """
    A class for analyzing fluid break dynamics in video sequences using deep learning segmentation.
    
    This class provides methods to:
    - Detect end of expansion phase
    - Find break points in fluid
    - Measure height progression at break points or mask center
    - Visualize and save analysis results
    """
    
    def __init__(self, model_path: str = 'best_model/best_model.pth', device: str = None):
        """
        Initialize the FluidBreakAnalyzer.
        
        Args:
            model_path: Path to the trained model checkpoint
            device: Computing device ('cuda' or 'cpu'). If None, auto-detects.
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.transform = get_validation_transforms(image_size=(512, 512))
        
        # Analysis parameters
        self.distance_window_len = 7
        self.velocity_window_len = 5
        
        # Results storage
        self.horizontal_distances = []
        self.smooth_distances = []
        self.velocities = []
        self.end_of_expansion_frame = None
        self.break_moment_frame = None
    
    def _load_model(self, model_path: str) -> FluidSegmentationModel:
        """Load the segmentation model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        encoder_name = checkpoint.get('encoder_name', 'resnet101')
        print(f"Using encoder: {encoder_name}")
        
        model = create_model(encoder_name=encoder_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model
    
    def find_narrowest_point_and_break(self, mask: np.ndarray) -> Tuple[int, int, bool, int, int]:
        """
        Find the narrowest point in mask and detect if/where the mask breaks.
        
        Args:
            mask: Binary mask as numpy array (height, width, channels)
            
        Returns:
            Tuple of (narrowest_x, narrowest_height, break_detected, break_x, break_y)
        """
        height, width, _ = mask.shape
        mask = mask[:, :, 0]  
        
        # Find the leftmost and rightmost columns where the mask exists
        column_sums = mask.sum(axis=0)
        mask_columns = np.where(column_sums > 0)[0]
        
        if len(mask_columns) == 0:
            return width // 2, 0, False, -1, -1
        
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
        narrowest_height = search_column_sums[narrowest_relative_index] // 255
        
        # Check for break detection
        break_detected = False
        break_x = -1
        break_y = -1
        
        # Method 1: Check if narrowest point has zero height (complete break)
        if narrowest_height == 0:
            break_detected = True
            break_x = narrowest_index
            break_y = height // 2
        
        # Method 2: Check for significant gaps in the fluid at the narrowest point
        if not break_detected:
            column = mask[:, narrowest_index]
            white_pixels = np.where(column > 127)[0]
            
            if len(white_pixels) > 0:
                gaps = []
                gap_positions = []
                
                for i in range(1, len(white_pixels)):
                    gap_size = white_pixels[i] - white_pixels[i-1] - 1
                    if gap_size > 3:
                        gaps.append(gap_size)
                        gap_positions.append((white_pixels[i-1], white_pixels[i]))
                
                if gaps and max(gaps) > 10:
                    break_detected = True
                    break_x = narrowest_index
                    largest_gap_idx = gaps.index(max(gaps))
                    gap_start, gap_end = gap_positions[largest_gap_idx]
                    break_y = (gap_start + gap_end) // 2
        
        # Method 3: Check multiple columns around narrowest point for consistency
        if not break_detected:
            check_range = min(5, (right_bound - left_bound) // 4)
            break_positions = []
            
            for offset in range(-check_range, check_range + 1):
                check_x = narrowest_index + offset
                if left_bound <= check_x <= right_bound:
                    column = mask[:, check_x]
                    white_pixels = np.where(column > 127)[0]
                    
                    if len(white_pixels) == 0:
                        break_positions.append(check_x)
                    else:
                        for i in range(1, len(white_pixels)):
                            gap_size = white_pixels[i] - white_pixels[i-1] - 1
                            if gap_size > 15:
                                break_positions.append(check_x)
                                break
            
            if break_positions:
                break_detected = True
                center_idx = len(break_positions) // 2
                break_x = break_positions[center_idx]
                
                column = mask[:, break_x]
                white_pixels = np.where(column > 127)[0]
                
                if len(white_pixels) == 0:
                    break_y = height // 2
                else:
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
    
    def _moving_average(self, values: List[float], window_len: int) -> np.ndarray:
        """Calculate moving average of values."""
        weights = np.ones(window_len) / window_len
        return np.convolve(values, weights, mode='valid')
    
    def _check_end_of_expansion(self, cap: cv.VideoCapture) -> Optional[int]:
        """Check if expansion phase has ended."""
        if len(self.horizontal_distances) > self.distance_window_len:
            smooth_distance = self._moving_average(
                self.horizontal_distances[-self.distance_window_len-1:-1],
                self.distance_window_len
            )
            self.smooth_distances.append(smooth_distance)
            
        if len(self.smooth_distances) > 2:
            velocity = self.smooth_distances[-1] - self.smooth_distances[-2]
            self.velocities.append(velocity[0])

        if len(self.velocities) > self.velocity_window_len:
            smooth_velocity = self._moving_average(
                self.velocities[-self.velocity_window_len-1:-1],
                self.velocity_window_len
            )[0]

            if (self.horizontal_distances[-1] > self.horizontal_distances[0] * 1.5 and 
                np.abs(smooth_velocity) < 0.5):
                
                end_frame = cap.get(cv.CAP_PROP_POS_FRAMES) - (
                    self.velocity_window_len // 2 + self.distance_window_len // 2
                )
                print(f'End of expansion detected at frame: {end_frame}')
                return int(end_frame)
        return None
    
    def _check_break_moment(self, contours: List, cap: cv.VideoCapture) -> Optional[int]:
        """Check if break moment has occurred based on contour analysis."""
        if len(contours) > 1:
            areas = [cv.contourArea(cnt) for cnt in contours]
            largest_area = max(areas)
            areas.remove(largest_area)
            second_largest_area = max(areas) if areas else 0
            
            if second_largest_area > largest_area * 0.5:
                break_frame = cap.get(cv.CAP_PROP_POS_FRAMES)
                print(f'Break detected at frame: {break_frame}')
                return int(break_frame)
        return None
    
    def _process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Process a single frame through the segmentation model."""
        orig_frame = frame.copy()
        orig_frame = cv.resize(orig_frame, (512, 512))
        
        # Prepare frame for model
        model_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        model_frame = self.transform(image=model_frame)['image'].unsqueeze(0)
        model_frame = model_frame.to(self.device)

        # Generate prediction
        with torch.inference_mode():
            pred_mask = torch.sigmoid(self.model(model_frame))
        pred_mask = (pred_mask > 0.5).float()
        pred_mask = pred_mask.squeeze().cpu().numpy()
        pred_mask = (pred_mask * 255).astype(np.uint8)
        
        return orig_frame, pred_mask
    
    def measure_height_at_break_progression(self, end_frame: int, break_frame: int, 
                                          cap: cv.VideoCapture, visualize: bool = True) -> Dict:
        """
        Measure height progression at the break point from end_of_expansion to break detection.
        
        Args:
            end_frame: Frame where expansion ends
            break_frame: Frame where break is detected
            cap: Video capture object
            visualize: Whether to show real-time visualization
            
        Returns:
            Dictionary containing analysis results
        """
        results = {
            'frames': [],
            'heights': [],
            'break_x': None,
            'break_frame': None,
            'analysis_type': 'break'
        }
        
        # Find precise break point
        search_start_frame = max(end_frame, break_frame - 10)
        cap.set(cv.CAP_PROP_POS_FRAMES, search_start_frame)
        break_x_coordinate = None
        
        print(f"Scanning frames {search_start_frame} to {break_frame} to find precise break point...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            current_frame = int(cap.get(cv.CAP_PROP_POS_FRAMES) - 1)
            if current_frame >= break_frame:
                break
                
            orig_frame, pred_mask = self._process_frame(frame)
            pred_mask_3ch = cv.cvtColor(pred_mask, cv.COLOR_GRAY2BGR)
            
            narrowest_x, narrowest_height, break_detected, break_x, break_y = \
                self.find_narrowest_point_and_break(pred_mask_3ch)
            
            if break_detected and break_x_coordinate is None:
                break_x_coordinate = break_x
                results['break_x'] = break_x_coordinate
                results['break_frame'] = current_frame
                print(f"PRECISE BREAK DETECTED at frame {current_frame}, x-coordinate: {break_x_coordinate}")
                break
        
        # If no break found, use narrowest point
        if break_x_coordinate is None:
            print("No precise break detected, using narrowest point from frame near break")
            cap.set(cv.CAP_PROP_POS_FRAMES, break_frame - 5)
            ret, frame = cap.read()
            if ret:
                orig_frame, pred_mask = self._process_frame(frame)
                pred_mask_3ch = cv.cvtColor(pred_mask, cv.COLOR_GRAY2BGR)
                narrowest_x, _, _, _, _ = self.find_narrowest_point_and_break(pred_mask_3ch)
                break_x_coordinate = narrowest_x
                results['break_x'] = break_x_coordinate
        
        # Measure height progression
        cap.set(cv.CAP_PROP_POS_FRAMES, end_frame)
        print(f"Measuring height progression at break x-coordinate: {break_x_coordinate}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            current_frame = int(cap.get(cv.CAP_PROP_POS_FRAMES) - 1)
            stop_frame = results['break_frame'] if results['break_frame'] else break_frame
            if current_frame >= stop_frame:
                break
                
            orig_frame, pred_mask = self._process_frame(frame)
            
            # Measure height at break x-coordinate
            column = pred_mask[:, break_x_coordinate]
            white_pixels = np.where(column > 127)[0]
            height = len(white_pixels) if len(white_pixels) > 0 else 0
            
            results['frames'].append(current_frame)
            results['heights'].append(height)
            
            if visualize:
                self._visualize_height_measurement(orig_frame, pred_mask, break_x_coordinate, 
                                                 white_pixels, height, current_frame, 
                                                 len(results["frames"]), stop_frame-end_frame,
                                                 "Height Measurement at Break Point")
            
            print(f"Frame {current_frame}: Height at break x={break_x_coordinate} = {height}px")
        
        if visualize:
            cv.destroyWindow('Height Measurement at Break Point')
        return results
    
    def measure_height_at_mask_center(self, end_frame: int, cap: cv.VideoCapture, 
                                    visualize: bool = True) -> Dict:
        """
        Measure height progression at the center of the mask when no break is detected.
        
        Args:
            end_frame: Frame where expansion ends
            cap: Video capture object
            visualize: Whether to show real-time visualization
            
        Returns:
            Dictionary containing analysis results
        """
        results = {
            'frames': [],
            'heights': [],
            'center_x': None,
            'analysis_type': 'center'
        }
        
        # Find center x-coordinate
        cap.set(cv.CAP_PROP_POS_FRAMES, end_frame + 5)
        ret, frame = cap.read()
        
        if ret:
            orig_frame, pred_mask = self._process_frame(frame)
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
            center_x_coordinate = 256
            results['center_x'] = center_x_coordinate
        
        # Measure height progression
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        cap.set(cv.CAP_PROP_POS_FRAMES, end_frame)
        
        print(f"Measuring height progression at mask center x-coordinate: {center_x_coordinate}")
        print(f"From frame {end_frame} to end of video (frame {total_frames})")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            current_frame = int(cap.get(cv.CAP_PROP_POS_FRAMES) - 1)
            orig_frame, pred_mask = self._process_frame(frame)
            
            # Measure height at center x-coordinate
            column = pred_mask[:, center_x_coordinate]
            white_pixels = np.where(column > 127)[0]
            height = len(white_pixels) if len(white_pixels) > 0 else 0
            
            results['frames'].append(current_frame)
            results['heights'].append(height)
            
            if visualize:
                self._visualize_height_measurement(orig_frame, pred_mask, center_x_coordinate, 
                                                 white_pixels, height, current_frame, 
                                                 len(results["frames"]), total_frames-end_frame,
                                                 "Height Measurement at Mask Center", is_center=True)
            
            print(f"Frame {current_frame}: Height at center x={center_x_coordinate} = {height}px")
        
        if visualize:
            cv.destroyWindow('Height Measurement at Mask Center')
        return results
    
    def _visualize_height_measurement(self, orig_frame: np.ndarray, pred_mask: np.ndarray, 
                                    x_coord: int, white_pixels: np.ndarray, height: int, 
                                    current_frame: int, progress: int, total: int, 
                                    window_name: str, is_center: bool = False):
        """Helper method to visualize height measurements."""
        pred_mask_vis = cv.cvtColor(pred_mask, cv.COLOR_GRAY2BGR)
        
        # Draw vertical line
        color = (255, 0, 0) if is_center else (0, 0, 255)
        cv.line(orig_frame, (x_coord, 0), (x_coord, orig_frame.shape[0]), color, 2)
        cv.line(pred_mask_vis, (x_coord, 0), (x_coord, pred_mask_vis.shape[0]), color, 2)
        
        # Draw measurement line if fluid exists
        if len(white_pixels) > 0:
            top_y = white_pixels[0]
            bottom_y = white_pixels[-1]
            
            cv.line(orig_frame, (x_coord, top_y), (x_coord, bottom_y), (0, 255, 0), 4)
            cv.line(pred_mask_vis, (x_coord, top_y), (x_coord, bottom_y), (0, 255, 0), 4)
            
            cv.circle(orig_frame, (x_coord, top_y), 3, (255, 255, 0), -1)
            cv.circle(orig_frame, (x_coord, bottom_y), 3, (255, 255, 0), -1)
            cv.circle(pred_mask_vis, (x_coord, top_y), 3, (255, 255, 0), -1)
            cv.circle(pred_mask_vis, (x_coord, bottom_y), 3, (255, 255, 0), -1)
        
        # Add text information
        cv.putText(orig_frame, f'Frame: {current_frame}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        label = 'Center X' if is_center else 'Break X'
        cv.putText(orig_frame, f'{label}: {x_coord}', (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv.putText(orig_frame, f'Height: {height}px', (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv.putText(orig_frame, f'Progress: {progress}/{total}', (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        if is_center:
            cv.putText(orig_frame, f'NO BREAK - Center Analysis', (10, 150), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        combined = cv.hconcat([orig_frame, pred_mask_vis])
        cv.imshow(window_name, combined)
        
        key = cv.waitKey(1) & 0xFF
        if key == 27:  # ESC to exit
            return False
        elif key == ord(' '):  # SPACE to pause
            cv.waitKey(0)
        return True
    
    def analyze_video(self, video_path: str, visualize_process: bool = True, 
                     save_results: bool = True) -> Dict:
        """
        Analyze a complete video for fluid break dynamics.
        
        Args:
            video_path: Path to the video file
            visualize_process: Whether to show real-time processing
            save_results: Whether to save plots and CSV files
            
        Returns:
            Dictionary containing complete analysis results
        """
        cap = cv.VideoCapture(video_path)
        
        # Reset analysis state
        self.horizontal_distances = []
        self.smooth_distances = []
        self.velocities = []
        self.end_of_expansion_frame = None
        self.break_moment_frame = None
        
        print(f"Analyzing video: {video_path}")
        
        # Phase 1: Find end of expansion and break moment
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            orig_frame, pred_mask = self._process_frame(frame)
            
            contours = cv.findContours(pred_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
            largest_contour = max(contours, key=cv.contourArea) if contours else None

            if largest_contour is not None:
                max_left = largest_contour[:, 0, 0].min()
                max_right = largest_contour[:, 0, 0].max()
                horizontal_distance = max_right - max_left
                self.horizontal_distances.append(horizontal_distance)

                if not self.end_of_expansion_frame:
                    self.end_of_expansion_frame = self._check_end_of_expansion(cap)
                    
                if self.end_of_expansion_frame:
                    self.break_moment_frame = self._check_break_moment(contours, cap)
                
                if self.break_moment_frame:
                    break

            if visualize_process:
                if largest_contour is not None:
                    cv.drawContours(orig_frame, [largest_contour], -1, (0, 255, 0), 2)
                pred_mask_vis = cv.cvtColor(pred_mask, cv.COLOR_GRAY2BGR)
                combined = cv.hconcat([orig_frame, pred_mask_vis])
                cv.imshow('Segmentation Inference', combined)

                key_code = cv.waitKey(1)  
                if key_code == 27:  
                    cv.destroyAllWindows()
                    break
        
        cv.destroyAllWindows()
        
        # Phase 2: Detailed analysis
        analysis_results = {}
        
        if self.end_of_expansion_frame and self.break_moment_frame:
            print(f"\n=== ANALYZING BREAK PROGRESSION ===")
            print(f"From frame {self.end_of_expansion_frame} to {self.break_moment_frame}")
            
            height_analysis = self.measure_height_at_break_progression(
                self.end_of_expansion_frame, self.break_moment_frame, cap, visualize_process
            )
            analysis_results = height_analysis
            
        elif self.end_of_expansion_frame and not self.break_moment_frame:
            print(f"\n=== NO BREAK DETECTED - ANALYZING MASK CENTER ===")
            print(f"From frame {self.end_of_expansion_frame} to end of video")
            
            height_analysis = self.measure_height_at_mask_center(
                self.end_of_expansion_frame, cap, visualize_process
            )
            analysis_results = height_analysis
            
        else:
            print(f"\n=== NO END OF EXPANSION DETECTED ===")
            print("The fluid may still be expanding or the analysis parameters need adjustment.")
            analysis_results = {'error': 'No end of expansion detected'}
        
        # Save results if requested
        if save_results and analysis_results.get('frames'):
            self.save_analysis_results(analysis_results, video_path)
        
        # Add metadata
        analysis_results.update({
            'video_path': video_path,
            'end_of_expansion_frame': self.end_of_expansion_frame,
            'break_moment_frame': self.break_moment_frame,
            'total_frames_analyzed': len(analysis_results.get('frames', []))
        })
        
        cap.release()
        cv.destroyAllWindows()
        
        return analysis_results
    
    def save_analysis_results(self, results: Dict, video_path: str):
        """Save analysis results to files."""
        if not results.get('frames'):
            print("No data to save")
            return
            
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        analysis_type = results.get('analysis_type', 'unknown')
        x_coord = results.get('break_x') or results.get('center_x')
        
        if analysis_type == 'break':
            plt.scatter(results['frames'], results['heights'], c='b', s=2, label='Height at break point')
            title = f'Fluid Height Evolution at x={x_coord} (Break Point)'
            
            break_frame = results.get('break_frame')
            if break_frame and break_frame in results['frames']:
                break_idx = results['frames'].index(break_frame)
                plt.axvline(x=break_frame, color='red', linestyle='--', linewidth=2, 
                           label=f'Break detected (frame {break_frame})')
                plt.plot(break_frame, results['heights'][break_idx], 'ro', markersize=10, 
                        label='Break point')
        else:
            plt.scatter(results['frames'], results['heights'], c='g', s=2, label='Height at mask center')
            title = f'Fluid Height Evolution at x={x_coord} (Mask Center - No Break)'
        
        plt.title(title)
        plt.xlabel('Frame Number')
        plt.ylabel('Height (pixels)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        plt.savefig('height_progression_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save CSV data
        header = 'Frame,Height_at_break_point' if analysis_type == 'break' else 'Frame,Height_at_mask_center'
        np.savetxt('height_progression_data.csv', 
                  np.column_stack([results['frames'], results['heights']]),
                  delimiter=',', header=header, comments='')
        
        print(f"Results saved:")
        print(f"- Plot: height_progression_analysis.png")
        print(f"- Data: height_progression_data.csv")
        
        # Print summary
        avg_height = np.mean(results['heights'])
        min_height = min(results['heights'])
        max_height = max(results['heights'])
        
        print(f"\n=== ANALYSIS SUMMARY ===")
        print(f"Video: {Path(video_path).name}")
        print(f"Analysis type: {analysis_type}")
        print(f"X-coordinate: {x_coord}")
        print(f"Frames analyzed: {len(results['frames'])}")
        print(f"Height range: {min_height} - {max_height} pixels")
        print(f"Average height: {avg_height:.1f} pixels")
        
        if analysis_type == 'break' and results.get('break_frame'):
            print(f"Break detected at frame: {results['break_frame']}")


# Example usage and backward compatibility functions
def analyze_video_simple(video_path: str, model_path: str = 'best_model/best_model.pth') -> Dict:
    """Simple function interface for backward compatibility."""
    analyzer = FluidBreakAnalyzer(model_path)
    return analyzer.analyze_video(video_path)


if __name__ == "__main__":
    # Example usage
    analyzer = FluidBreakAnalyzer()
    results = analyzer.analyze_video('video/problematyczny1.MP4')
    print("Analysis complete!")
