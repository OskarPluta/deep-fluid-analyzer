# FluidBreakAnalyzer Class

A refactored, easy-to-import class for analyzing fluid break dynamics in video sequences using deep learning segmentation.

## Features

- **Object-oriented design**: Clean, reusable class interface
- **Flexible analysis**: Supports both break detection and center-based analysis
- **Real-time visualization**: Optional live preview during processing
- **Batch processing**: Analyze multiple videos programmatically
- **Result persistence**: Automatic saving of plots and CSV data
- **Backward compatibility**: Simple function interface available

## Quick Start

### Basic Usage

```python
from fluid_break_analyzer import FluidBreakAnalyzer

# Initialize the analyzer
analyzer = FluidBreakAnalyzer(model_path='best_model/best_model.pth')

# Analyze a video
results = analyzer.analyze_video('video/my_video.MP4')

# Access results
print(f"Analysis type: {results['analysis_type']}")
print(f"Frames analyzed: {len(results['frames'])}")
print(f"Height measurements: {results['heights']}")
```

### Simple Function Interface

```python
from fluid_break_analyzer import analyze_video_simple

# One-line analysis
results = analyze_video_simple('video/my_video.MP4')
```

## Class Methods

### `FluidBreakAnalyzer(model_path, device=None)`

Initialize the analyzer with a trained model.

**Parameters:**
- `model_path` (str): Path to the model checkpoint file
- `device` (str, optional): Computing device ('cuda' or 'cpu'). Auto-detects if None.

### `analyze_video(video_path, visualize_process=True, save_results=True)`

Analyze a complete video for fluid break dynamics.

**Parameters:**
- `video_path` (str): Path to the video file
- `visualize_process` (bool): Show real-time processing visualization
- `save_results` (bool): Save plots and CSV files automatically

**Returns:**
- `dict`: Analysis results containing frames, heights, break information, etc.

### `measure_height_at_break_progression(end_frame, break_frame, cap, visualize=True)`

Measure height progression at the detected break point.

### `measure_height_at_mask_center(end_frame, cap, visualize=True)`

Measure height progression at the center of the mask (when no break is detected).

### `find_narrowest_point_and_break(mask)`

Enhanced break detection algorithm that finds the narrowest point and detects breaks.

**Returns:**
- `tuple`: (narrowest_x, narrowest_height, break_detected, break_x, break_y)

## Usage Examples

### Custom Configuration

```python
analyzer = FluidBreakAnalyzer(
    model_path='checkpoints/my_model.pth',
    device='cuda'
)

results = analyzer.analyze_video(
    video_path='video/test.MP4',
    visualize_process=False,  # Disable visualization for faster processing
    save_results=True
)
```

### Batch Processing

```python
video_files = ['video1.MP4', 'video2.MP4', 'video3.MP4']
analyzer = FluidBreakAnalyzer()

for video in video_files:
    results = analyzer.analyze_video(
        video_path=video,
        visualize_process=False,
        save_results=True
    )
    print(f"{video}: {results['analysis_type']} analysis completed")
```

### Manual Step-by-Step Analysis

```python
import cv2 as cv

analyzer = FluidBreakAnalyzer()
cap = cv.VideoCapture('video/my_video.MP4')

# Use individual methods for custom workflows
# Process frame
orig_frame, pred_mask = analyzer._process_frame(frame)

# Detect breaks
narrowest_x, height, break_detected, break_x, break_y = \
    analyzer.find_narrowest_point_and_break(pred_mask_3ch)

# Manual height measurement
if break_detected:
    results = analyzer.measure_height_at_break_progression(
        end_frame, break_frame, cap, visualize=False
    )
else:
    results = analyzer.measure_height_at_mask_center(
        end_frame, cap, visualize=False
    )

cap.release()
```

## Output Files

When `save_results=True`, the analyzer automatically creates:

- `height_progression_analysis.png`: Visualization plot
- `height_progression_data.csv`: Raw measurement data

## Result Dictionary Structure

```python
{
    'frames': [list of frame numbers],
    'heights': [list of height measurements in pixels],
    'analysis_type': 'break' or 'center',
    'break_x': x-coordinate of break point (if break detected),
    'center_x': x-coordinate of mask center (if no break),
    'break_frame': frame number where break was detected,
    'video_path': path to analyzed video,
    'end_of_expansion_frame': frame where expansion ends,
    'break_moment_frame': frame where break moment occurs,
    'total_frames_analyzed': number of frames processed
}
```

## Migration from Original Code

The new class maintains all functionality from the original `break_finder.py` script:

- All analysis algorithms are preserved
- Visualization capabilities are maintained  
- Results format is compatible
- Performance is equivalent or improved

### Key Improvements

1. **Modular design**: Easy to import and use in other projects
2. **Better error handling**: Graceful handling of edge cases
3. **Configurable visualization**: Can disable UI for automated processing
4. **Cleaner interfaces**: Simplified method signatures
5. **Documentation**: Comprehensive docstrings and examples

## Dependencies

- PyTorch
- OpenCV (cv2)
- NumPy
- Matplotlib
- Custom modules: `dataset`, `model` (from src/ directory)

## Example Script

See `example_usage.py` for comprehensive usage examples including:
- Basic analysis
- Batch processing
- Custom configuration
- Step-by-step analysis
