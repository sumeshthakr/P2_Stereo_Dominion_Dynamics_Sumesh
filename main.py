#!/usr/bin/env python3
"""
Stereo Vision Depth Estimation System - Main Entry Point
=========================================================

Real-time depth estimation from stereo camera pairs with object detection,
point cloud generation, and 3D visualization.

Author: Sumesh Thakur (sumeshthkr@gmail.com)

Usage:
    python main.py --left video_left.mp4 --right video_right.mp4 --config calibration.json
    python main.py --webcam 0 1 --config calibration.json



References:
- OpenCV Python Tutorials: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
- Stereo Vision: Hartley & Zisserman, "Multiple View Geometry in Computer Vision"
- KITTI Dataset: https://www.cvlibs.net/datasets/kitti/
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.config import (
    CameraConfig,
    load_config_from_json,
    create_default_config,
    interactive_config_entry,
)
from src.video_input import StereoVideoSource, open_video_pair
from src.depth_estimation import DepthEstimator, DepthMethod, compare_methods
from src.object_detection import ObjectDetector, DetectorType, draw_detections
from src.visualization import (
    colorize_disparity,
    colorize_depth,
    create_depth_overlay,
    create_visualization_grid,
    depth_to_point_cloud,
    create_bird_eye_view,
    draw_3d_bbox_on_bev,
    save_point_cloud_ply,
    add_depth_legend,
    add_depth_stats_overlay,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-time Stereo Vision Depth Estimation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process stereo video files
    python main.py --left left.mp4 --right right.mp4 --config calibration.json
    
    # Use webcams (indices 0 and 1)
    python main.py --webcam 0 1 --config calibration.json
    
    # Compare depth estimation methods
    python main.py --left left.mp4 --right right.mp4 --config calibration.json --compare-methods
    
    # Save outputs
    python main.py --left left.mp4 --right right.mp4 --config calibration.json --output outputs/

Author: Sumesh Thakur (sumeshthkr@gmail.com)
        """,
    )

    # Input sources
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--webcam",
        nargs=2,
        type=int,
        metavar=("LEFT", "RIGHT"),
        help="Webcam indices for left and right cameras",
    )
    input_group.add_argument("--left", type=str, help="Path to left video file")

    parser.add_argument(
        "--right",
        type=str,
        help="Path to right video file (required when using --left)",
    )

    # Configuration
    parser.add_argument(
        "--config", type=str, help="Path to camera calibration JSON file"
    )
    parser.add_argument(
        "--interactive-config",
        action="store_true",
        help="Interactively enter camera parameters",
    )

    # Depth estimation method
    parser.add_argument(
        "--method",
        type=str,
        choices=["bm", "sgbm", "sgbm_3way"],
        default="sgbm",
        help="Depth estimation method (default: sgbm)",
    )
    parser.add_argument(
        "--compare-methods",
        action="store_true",
        help="Compare all depth estimation methods",
    )

    # Object detection
    parser.add_argument(
        "--detector",
        type=str,
        choices=[
            "contour",
            "background_sub",
            "hog_svm",
            "color_blob",
            "mobilenet_ssd",
            "yolo_nano",
            "none",
        ],
        default="contour",
        help="Object detection method (default: contour). ML-based: mobilenet_ssd, yolo_nano",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=2,
        help="Maximum number of objects to detect (default: 2)",
    )

    # Performance
    parser.add_argument(
        "--downsample",
        type=float,
        default=1.0,
        help="Downsample factor for performance (default: 1.0 = no downsampling)",
    )
    parser.add_argument(
        "--num-disparities",
        type=int,
        default=128,
        help="Number of disparities (default: 128, must be divisible by 16)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=5,
        help="Block size for stereo matching (default: 5, must be odd)",
    )

    # Output
    parser.add_argument(
        "--output", type=str, help="Output directory for saving results"
    )
    parser.add_argument(
        "--save-pointcloud", action="store_true", help="Save point cloud to PLY file"
    )
    parser.add_argument(
        "--auto-save",
        action="store_true",
        help="Automatically save frames to output directory",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=30,
        help="Frame interval for auto-saving (default: 30, i.e., save every 30th frame)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without display (useful for headless processing)",
    )

    # Visualization
    parser.add_argument(
        "--show-bev", action="store_true", help="Show Bird's Eye View visualization"
    )
    parser.add_argument(
        "--verify-depth",
        action="store_true",
        help="Enable depth verification mode with statistics overlay",
    )

    return parser.parse_args()


def setup_config(args) -> CameraConfig:
    """Load or create camera configuration."""
    if args.config:
        print(f"Loading configuration from: {args.config}")
        return load_config_from_json(args.config)
    elif args.interactive_config:
        return interactive_config_entry()
    else:
        print("Using default camera configuration")
        return create_default_config()


def setup_video_source(args, config: CameraConfig) -> StereoVideoSource:
    """Set up video source based on arguments."""
    if args.webcam:
        left_idx, right_idx = args.webcam
        print(f"Opening webcams: left={left_idx}, right={right_idx}")
        return StereoVideoSource(
            left_source=left_idx,
            right_source=right_idx,
            config=config,
            downsample_factor=args.downsample,
        )
    else:
        if not args.right:
            print("Error: --right is required when using --left")
            sys.exit(1)

        print(f"Opening video files: left={args.left}, right={args.right}")
        return open_video_pair(
            args.left, args.right, config=config, downsample_factor=args.downsample
        )


def setup_depth_estimator(args, config: CameraConfig) -> DepthEstimator:
    """Set up depth estimator based on arguments."""
    method_map = {
        "bm": DepthMethod.BM,
        "sgbm": DepthMethod.SGBM,
        "sgbm_3way": DepthMethod.SGBM_3WAY,
    }

    return DepthEstimator(
        config=config,
        method=method_map[args.method],
        num_disparities=args.num_disparities,
        block_size=args.block_size,
    )


def setup_object_detector(args) -> Optional[ObjectDetector]:
    """Set up object detector based on arguments."""
    if args.detector == "none":
        return None

    detector_map = {
        "contour": DetectorType.CONTOUR,
        "background_sub": DetectorType.BACKGROUND_SUB,
        "hog_svm": DetectorType.HOG_SVM,
        "color_blob": DetectorType.COLOR_BLOB,
        "mobilenet_ssd": DetectorType.MOBILENET_SSD,
        "yolo_nano": DetectorType.YOLO_NANO,
    }

    return ObjectDetector(
        detector_type=detector_map[args.detector], max_objects=args.max_objects
    )


def run_method_comparison(video_source: StereoVideoSource, config: CameraConfig):
    """Run comparison of all depth estimation methods."""
    print("\n=== Depth Method Comparison ===\n")

    # Get first frame
    frame = video_source.read()
    if frame is None:
        print("Error: Could not read frame from video source")
        return

    results = compare_methods(frame.left, frame.right, config)

    print("\nResults:")
    print("-" * 40)
    for method_name, result in results.items():
        valid_depth = result.depth_map[result.depth_map > 0]
        if len(valid_depth) > 0:
            mean_depth = np.mean(valid_depth)
            median_depth = np.median(valid_depth)
        else:
            mean_depth = median_depth = 0

        print(f"{method_name}:")
        print(f"  Time: {result.computation_time_ms:.2f} ms")
        print(f"  Mean depth: {mean_depth:.2f} m")
        print(f"  Median depth: {median_depth:.2f} m")
        print()

    # Visualize comparison
    h, w = frame.left.shape[:2]
    comparison = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)

    # Top-left: original
    comparison[:h, :w] = frame.left
    cv2.putText(
        comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
    )

    # Top-right: BM
    bm_depth = colorize_depth(results["bm"].depth_map)
    if bm_depth.shape[:2] != (h, w):
        bm_depth = cv2.resize(bm_depth, (w, h))
    comparison[:h, w:] = bm_depth
    cv2.putText(
        comparison,
        f"BM ({results['bm'].computation_time_ms:.1f}ms)",
        (w + 10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    # Bottom-left: SGBM
    sgbm_depth = colorize_depth(results["sgbm"].depth_map)
    if sgbm_depth.shape[:2] != (h, w):
        sgbm_depth = cv2.resize(sgbm_depth, (w, h))
    comparison[h:, :w] = sgbm_depth
    cv2.putText(
        comparison,
        f"SGBM ({results['sgbm'].computation_time_ms:.1f}ms)",
        (10, h + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    # Bottom-right: SGBM 3-way
    sgbm3_depth = colorize_depth(results["sgbm_3way"].depth_map)
    if sgbm3_depth.shape[:2] != (h, w):
        sgbm3_depth = cv2.resize(sgbm3_depth, (w, h))
    comparison[h:, w:] = sgbm3_depth
    cv2.putText(
        comparison,
        f"SGBM-3WAY ({results['sgbm_3way'].computation_time_ms:.1f}ms)",
        (w + 10, h + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Method Comparison", comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    """Main entry point."""
    args = parse_args()

    # Print header
    print("=" * 60)
    print("  Stereo Vision Depth Estimation System")
    print("=" * 60)
    print()

    # Setup
    config = setup_config(args)
    print(f"Baseline: {config.baseline:.4f} m")
    print(f"Focal length: {config.focal_length_left:.2f} px")
    print(f"Image size: {config.image_size}")
    print()

    video_source = setup_video_source(args, config)
    if not video_source.open():
        print("Error: Could not open video source")
        sys.exit(1)

    # Method comparison mode
    if args.compare_methods:
        run_method_comparison(video_source, config)
        video_source.close()
        return

    depth_estimator = setup_depth_estimator(args, config)
    object_detector = setup_object_detector(args)

    # Create output directory if needed
    output_dir = None
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Method: {args.method}")
    print(f"Detector: {args.detector}")
    print(f"Downsample: {args.downsample}x")
    print()
    print("Processing...")
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save frame")
    print("  'p' - Save point cloud")
    print("  'm' - Cycle depth method")
    print("  'SPACE' - Pause/Resume")
    print("  '+' or '=' - Increase delay (slower)")
    print("  '-' - Decrease delay (faster)")
    print()

    # FPS calculation
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0.0

    # Playback control
    paused = False
    frame_delay = 1  # Default delay in ms (1 = fastest)

    frame_idx = 0

    try:
        for stereo_frame in video_source.frames():
            frame_start = time.time()

            # Compute depth
            depth_result = depth_estimator.compute_depth(
                stereo_frame.left, stereo_frame.right
            )

            # Object detection
            objects = []
            if object_detector is not None:
                objects = object_detector.detect(
                    stereo_frame.left, depth_result.depth_map, config.focal_length_left
                )

            # Update FPS
            fps_frame_count += 1
            if fps_frame_count >= 10:
                elapsed = time.time() - fps_start_time
                current_fps = fps_frame_count / elapsed
                fps_start_time = time.time()
                fps_frame_count = 0

            # Visualization
            if not args.no_display:
                # Main visualization grid
                grid = create_visualization_grid(
                    stereo_frame.left,
                    stereo_frame.right,
                    depth_result.disparity,
                    depth_result.depth_map,
                    depth_result.confidence,
                    objects,
                    current_fps,
                )

                # Add depth verification stats if enabled
                if args.verify_depth:
                    grid = add_depth_stats_overlay(
                        grid,
                        depth_result.disparity,
                        depth_result.depth_map,
                        config.baseline,
                        config.focal_length_left,
                    )

                # Resize if too large
                max_height = 900
                if grid.shape[0] > max_height:
                    scale = max_height / grid.shape[0]
                    grid = cv2.resize(grid, None, fx=scale, fy=scale)

                cv2.imshow("Stereo Vision Depth Estimation", grid)

                # Bird's Eye View
                if args.show_bev:
                    points, colors = depth_to_point_cloud(
                        depth_result.depth_map, stereo_frame.left, config, subsample=4
                    )

                    bev = create_bird_eye_view(points)
                    bev = draw_3d_bbox_on_bev(bev, objects)
                    bev = cv2.resize(bev, (400, 400))
                    cv2.imshow("Bird's Eye View", bev)

                # Handle key presses
                key = cv2.waitKey(frame_delay) & 0xFF

                # Handle pause state
                while paused:
                    pause_key = cv2.waitKey(100) & 0xFF
                    if pause_key == ord(" "):  # Space to unpause
                        paused = False
                        print("Resumed")
                    elif pause_key == ord("q"):
                        print("\nQuitting...")
                        paused = False
                        key = ord("q")  # Set key to quit
                        break

                if key == ord("q"):
                    print("\nQuitting...")
                    break
                elif key == ord(" "):  # Space to pause
                    paused = True
                    print("Paused (press SPACE to resume)")
                elif key == ord("+") or key == ord("="):
                    # Increase delay (slower playback)
                    frame_delay = min(frame_delay + 10, 500)
                    print(f"Frame delay: {frame_delay}ms")
                elif key == ord("-"):
                    # Decrease delay (faster playback)
                    frame_delay = max(frame_delay - 10, 1)
                    print(f"Frame delay: {frame_delay}ms")
                elif key == ord("s"):
                    # Save current frame
                    if output_dir:
                        cv2.imwrite(
                            str(output_dir / f"frame_{frame_idx:04d}_left.png"),
                            stereo_frame.left,
                        )
                        cv2.imwrite(
                            str(output_dir / f"frame_{frame_idx:04d}_depth.png"),
                            colorize_depth(depth_result.depth_map),
                        )
                        print(f"Saved frame {frame_idx}")
                elif key == ord("p"):
                    # Save point cloud
                    if output_dir:
                        points, colors = depth_to_point_cloud(
                            depth_result.depth_map, stereo_frame.left, config
                        )
                        save_point_cloud_ply(
                            points,
                            colors,
                            str(output_dir / f"pointcloud_{frame_idx:04d}.ply"),
                        )
                        print(f"Saved point cloud for frame {frame_idx}")
                elif key == ord("m"):
                    # Cycle through methods
                    methods = [DepthMethod.BM, DepthMethod.SGBM, DepthMethod.SGBM_3WAY]
                    current_idx = methods.index(depth_estimator.method)
                    next_method = methods[(current_idx + 1) % len(methods)]
                    depth_estimator.set_method(next_method)
                    print(f"Switched to method: {next_method.value}")

            # Print stats periodically
            if frame_idx % 100 == 0 and frame_idx > 0:
                valid_depth = depth_result.depth_map[depth_result.depth_map > 0]
                if len(valid_depth) > 0:
                    print(
                        f"Frame {frame_idx}: FPS={current_fps:.1f}, "
                        f"Depth={np.median(valid_depth):.2f}m, "
                        f"Objects={len(objects)}"
                    )

            # Auto-save frames if enabled (saves first frame and then every save_interval)
            if (
                output_dir
                and args.auto_save
                and (frame_idx == 0 or frame_idx % args.save_interval == 0)
            ):
                cv2.imwrite(
                    str(output_dir / f"frame_{frame_idx:04d}_left.png"),
                    stereo_frame.left,
                )
                cv2.imwrite(
                    str(output_dir / f"frame_{frame_idx:04d}_depth.png"),
                    colorize_depth(depth_result.depth_map),
                )
                if args.save_pointcloud:
                    points, colors = depth_to_point_cloud(
                        depth_result.depth_map, stereo_frame.left, config
                    )
                    save_point_cloud_ply(
                        points,
                        colors,
                        str(output_dir / f"pointcloud_{frame_idx:04d}.ply"),
                    )

            frame_idx += 1

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        video_source.close()
        cv2.destroyAllWindows()

    print(f"\nProcessed {frame_idx} frames")
    print("Done!")


if __name__ == "__main__":
    main()
