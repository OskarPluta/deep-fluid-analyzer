#!/usr/bin/env python3
"""
Example usage of the FluidBreakAnalyzer class.

This script demonstrates how to use the refactored fluid break analysis functionality.
"""

from fluid_break_analyzer import FluidBreakAnalyzer, analyze_video_simple


def main():
    """Demonstrate different ways to use the FluidBreakAnalyzer."""
    
    # Method 1: Simple function interface (backward compatibility)
    print("=== Method 1: Simple function interface ===")
    results = analyze_video_simple('video/problematyczny1.MP4')
    print(f"Analysis completed. Analyzed {results.get('total_frames_analyzed', 0)} frames.")
    
    # Method 2: Class-based interface with custom configuration
    print("\n=== Method 2: Class-based interface ===")
    
    # Initialize analyzer with custom model path
    analyzer = FluidBreakAnalyzer(
        model_path='best_model/best_model.pth',
        device='cuda'  # or 'cpu'
    )
    
    # Analyze video with custom settings
    results = analyzer.analyze_video(
        video_path='video/problematyczny1.MP4',
        visualize_process=True,  # Show real-time processing
        save_results=True        # Save plots and CSV files
    )
    
    # Access detailed results
    print(f"Analysis type: {results.get('analysis_type')}")
    print(f"End of expansion frame: {results.get('end_of_expansion_frame')}")
    print(f"Break moment frame: {results.get('break_moment_frame')}")
    
    if results.get('frames'):
        print(f"Height measurements: {len(results['frames'])} data points")
        print(f"Height range: {min(results['heights'])} - {max(results['heights'])} pixels")
    
    # Method 3: Step-by-step analysis for custom workflows
    print("\n=== Method 3: Step-by-step analysis ===")
    
    import cv2 as cv
    
    # Initialize analyzer
    analyzer = FluidBreakAnalyzer()
    
    # Open video manually for custom processing
    cap = cv.VideoCapture('video/problematyczny1.MP4')
    
    # You can now use individual methods:
    # - analyzer.find_narrowest_point_and_break(mask)
    # - analyzer.measure_height_at_break_progression(end_frame, break_frame, cap)
    # - analyzer.measure_height_at_mask_center(end_frame, cap)
    # - analyzer.save_analysis_results(results, video_path)
    
    cap.release()
    
    print("Example completed!")


def analyze_multiple_videos():
    """Example of analyzing multiple videos in batch."""
    video_files = [
        'video/n1.MP4',
        'video/n5.MP4',
        'video/n8.MP4',
        'video/problematyczny1.MP4'
    ]
    
    analyzer = FluidBreakAnalyzer()
    
    results_summary = []
    
    for video_file in video_files:
        print(f"\nAnalyzing {video_file}...")
        try:
            results = analyzer.analyze_video(
                video_path=video_file,
                visualize_process=False,  # Disable visualization for batch processing
                save_results=True
            )
            
            summary = {
                'video': video_file,
                'analysis_type': results.get('analysis_type'),
                'end_frame': results.get('end_of_expansion_frame'),
                'break_frame': results.get('break_moment_frame'),
                'measurements': len(results.get('frames', [])),
                'success': True
            }
            
            if results.get('heights'):
                summary.update({
                    'avg_height': sum(results['heights']) / len(results['heights']),
                    'min_height': min(results['heights']),
                    'max_height': max(results['heights'])
                })
            
        except Exception as e:
            summary = {
                'video': video_file,
                'success': False,
                'error': str(e)
            }
        
        results_summary.append(summary)
    
    # Print batch results
    print("\n=== BATCH ANALYSIS SUMMARY ===")
    for summary in results_summary:
        if summary['success']:
            print(f"{summary['video']}: {summary['analysis_type']} analysis, "
                  f"{summary['measurements']} measurements")
        else:
            print(f"{summary['video']}: FAILED - {summary['error']}")


if __name__ == "__main__":
    main()
    
    # Uncomment to run batch analysis
    # analyze_multiple_videos()
