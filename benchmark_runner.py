#!/usr/bin/env python3
"""
Combined benchmark runner that tests multiple Tarantool versions and outputs results to a text file.
Reads configuration from a JSON file.

Example usage:
    python benchmark_runner_text.py benchmark_config.json --tarantool-path ../tarantool

Config file format:
    {
        "output": {
            "file": "benchmark_results.txt",
            "format": "csv"  // "csv" or "table"
        },
        "plot": {
            "enabled": true,  // Optional: enable plotting
            "output": "benchmark_plot.jpg",  // Optional: plot output file
            "title": "Benchmark Results",  // Optional: plot title
            "figure_size": [12, 8],  // Optional: figure size
            "dpi": 300  // Optional: DPI for output
        },
        "benchmark": {
            "benchmark_file": "benchmark.lua",  // Global benchmark file
            "arguments": {
                "index_count": 3,
                "fiber_count": 100,
                "REPLACE_PER_TXN_COUNT": 2,
                "divisor": 3
            },
            "argument_order": ["index_count", "fiber_count", "REPLACE_PER_TXN_COUNT", "divisor"],
            // Each argument can be: single value, array [1, 5, 10], or range {"start": 1, "end": 100, "points": 10}
            // Only one argument can have non-single values (array or range) at a time
            "test_iters": 5,
            "timeout": 60
        },
        "build_mode": "release",  // Global build mode: "release" or "debug"
        "versions": [
            {
                "branch": "master",
                "name": "Version Name",
                "benchmark_config": {
                    "benchmark_file": "other_benchmark.lua"  // Optional: overrides global benchmark_file
                }  // Optional: overrides global benchmark config
            }
        ]
    }
"""

import argparse
import json
import subprocess
import time
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path


def run_command(cmd: List[str], cwd: str = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=check
    )
    if result.returncode != 0 and check:
        print(f"Error output: {result.stderr}")
    return result


def checkout_branch(branch: str, tarantool_path: str) -> None:
    """Checkout the specified git branch."""
    print(f"Checking out branch: {branch}")
    run_command(['git', 'checkout', branch], cwd=tarantool_path)


def update_submodules(tarantool_path: str) -> None:
    """Update git submodules."""
    print("Updating submodules...")
    run_command(['git', 'submodule', 'update', '--init', '--recursive'], cwd=tarantool_path)


def build_tarantool(tarantool_path: str, build_mode: str = "release") -> None:
    """
    Build Tarantool in the release-build or debug-build directory.
    
    Args:
        tarantool_path: Path to Tarantool repository
        build_mode: "release" or "debug"
    """
    if build_mode not in ["release", "debug"]:
        raise ValueError(f"Invalid build_mode: {build_mode}. Must be 'release' or 'debug'")
    
    build_dir_name = f"{build_mode}-build"
    build_path = os.path.join(tarantool_path, build_dir_name)
    
    # Create build directory if it doesn't exist
    os.makedirs(build_path, exist_ok=True)
    
    # Check if CMakeLists.txt exists in tarantool_path
    cmake_lists = os.path.join(tarantool_path, 'CMakeLists.txt')
    if not os.path.exists(cmake_lists):
        raise FileNotFoundError(f"CMakeLists.txt not found in {tarantool_path}")
    
    # Run cmake if build directory is empty or CMakeCache doesn't exist
    cmake_cache = os.path.join(build_path, 'CMakeCache.txt')
    if not os.path.exists(cmake_cache):
        print(f"Running cmake with build type: {build_mode.upper()}...")
        cmake_build_type = "Release" if build_mode == "release" else "Debug"
        run_command(['cmake', f'-DCMAKE_BUILD_TYPE={cmake_build_type}', '..'], cwd=build_path)
    
    # Build
    print(f"Building Tarantool ({build_mode} mode)...")
    run_command(['make', '-j'], cwd=build_path)
    
    # Verify tarantool binary exists
    tarantool_binary = os.path.join(build_path, 'src', 'tarantool')
    if not os.path.exists(tarantool_binary):
        raise FileNotFoundError(f"Tarantool binary not found at {tarantool_binary}")


def run_tarantool_benchmark(
    tarantool_binary: str,
    benchmark_file: str,
    argument_values: List[int],
    timeout: int,
    work_dir: str
) -> float:
    """
    Run Tarantool benchmark once (with test_iters=1) and return execution time in seconds.
    Returns float('inf') on error or timeout.
    
    Args:
        argument_values: List of argument values to pass to the benchmark script in order
    """
    cmd = [tarantool_binary, benchmark_file] + [str(v) for v in argument_values]
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        end_time = time.time()
        
        if result.returncode != 0:
            print(f"  Error: {result.stderr[:200]}")
            return float('inf')
            
        return end_time - start_time
        
    except subprocess.TimeoutExpired:
        print(f"  Timeout")
        return float('inf')
    except Exception as e:
        print(f"  Exception: {e}")
        return float('inf')


def parse_range_config(value: Any, default: int) -> Tuple[List[int], bool]:
    """
    Parse a value that can be either a single number, an array of values, or a range object.
    
    Args:
        value: Either a number, a list/array of numbers, or a dict with 'start', 'end', 'points'
        default: Default value if value is None
    
    Returns:
        Tuple of (list of values, is_range_object)
    """
    if value is None:
        return [default], False
    
    if isinstance(value, dict):
        # Range object
        start = value.get('start', 1)
        end = value.get('end', 1000)
        points = value.get('points', 100)
        if points == 1:
            return [start], True
        else:
            return np.linspace(start, end, points, dtype=int).tolist(), True
    elif isinstance(value, (list, tuple)):
        # Array of values
        return [int(v) for v in value], False
    else:
        # Single value
        return [int(value)], False


def run_benchmark_suite(
    tarantool_binary: str,
    benchmark_file: str,
    benchmark_config: Dict[str, Any],
    work_dir: str
) -> List[Tuple[Dict[str, int], float]]:
    """
    Run a full benchmark suite and return list of (argument_values_dict, time).
    For each combination, runs the benchmark test_iters times and averages the results.
    """
    test_iters = benchmark_config.get('test_iters', 5)
    timeout = benchmark_config.get('timeout', 60)
    
    # Get arguments configuration
    arguments_config = benchmark_config.get('arguments', {})
    argument_order = benchmark_config.get('argument_order', list(arguments_config.keys()))
    
    if not arguments_config:
        raise ValueError("No arguments specified in benchmark config")
    
    # Parse all arguments
    parsed_arguments = {}
    non_single_count = 0
    non_single_arg_name = None
    
    for arg_name in argument_order:
        if arg_name not in arguments_config:
            raise ValueError(f"Argument '{arg_name}' in argument_order not found in arguments config")
        
        arg_value = arguments_config[arg_name]
        parsed_values, is_range = parse_range_config(arg_value, 0)
        parsed_arguments[arg_name] = parsed_values
        
        # Check if this is a non-single value (array or range)
        if len(parsed_values) > 1:
            non_single_count += 1
            non_single_arg_name = arg_name
    
    # Validate that at most one argument has non-single values
    if non_single_count > 1:
        raise ValueError(f"Only one argument can have non-single values (array or range), "
                        f"but found multiple: {non_single_count}")
    
    # Print configuration
    print(f"  Running benchmark:")
    for arg_name in argument_order:
        values = parsed_arguments[arg_name]
        if len(values) == 1:
            print(f"    {arg_name}: {values[0]}")
        else:
            print(f"    {arg_name}: range {values[0]}-{values[-1]} ({len(values)} points)")
    
    # Generate all combinations
    # Create list of value lists in argument order
    value_lists = [parsed_arguments[arg_name] for arg_name in argument_order]
    
    # Generate all combinations
    all_combinations = list(product(*value_lists))
    total_combinations = len(all_combinations)
    
    results = []
    current_combination = 0
    
    for combination in all_combinations:
        current_combination += 1
        # Create dict of argument values
        arg_values = {arg_name: value for arg_name, value in zip(argument_order, combination)}
        
        # Create display string for progress
        arg_display = ", ".join([f"{k}={v}" for k, v in arg_values.items()])
        print(f"  Progress: {current_combination}/{total_combinations}, {arg_display}", end="")
        
        # Create argument values list in order
        argument_values = [arg_values[arg_name] for arg_name in argument_order]
        
        # Run the benchmark test_iters times and sum the total time
        total_time = 0.0
        successful_runs = 0
        
        for iter_num in range(test_iters):
            execution_time = run_tarantool_benchmark(
                tarantool_binary,
                benchmark_file,
                argument_values,
                timeout,
                work_dir
            )
            
            if execution_time != float('inf'):
                total_time += execution_time
                successful_runs += 1
            else:
                # If any run fails, skip this combination
                break
        
        if successful_runs == test_iters:
            # Calculate average time per iteration
            avg_time = total_time / test_iters
            results.append((arg_values, avg_time))
            print(f" - total: {total_time:.2f}s ({test_iters} runs), avg: {avg_time:.4f}s/run")
        else:
            print(f" - SKIPPED (error or timeout, {successful_runs}/{test_iters} successful)")
    
    return results


def process_version(
    version_config: Dict[str, Any],
    tarantool_path: str,
    work_dir: str,
    global_config: Dict[str, Any]
) -> Tuple[List[Tuple[Dict[str, int], float]], str, List[str]]:
    """
    Process a single version: checkout, build, run benchmark.
    Returns (results, version_name, argument_order) where results is a list of (argument_values_dict, time).
    """
    branch = version_config['branch']
    version_name = version_config['name']
    # benchmark_config is already merged in main() with global config
    benchmark_config = version_config.get('benchmark_config', {})
    
    # Get benchmark_file from merged benchmark config
    benchmark_file = benchmark_config.get('benchmark_file')
    
    if not benchmark_file:
        raise ValueError(f"benchmark_file not specified in benchmark config for version {version_name}")
    
    # Get build mode from global config
    build_mode = global_config.get('build_mode', 'release')
    
    print(f"\n{'='*60}")
    print(f"Processing version: {version_name} (branch: {branch}, build: {build_mode})")
    print(f"{'='*60}")
    
    # Checkout branch
    checkout_branch(branch, tarantool_path)
    
    # Update submodules
    update_submodules(tarantool_path)
    
    # Build Tarantool
    build_tarantool(tarantool_path, build_mode)
    
    # Get tarantool binary path
    build_dir_name = f"{build_mode}-build"
    tarantool_binary = os.path.join(tarantool_path, build_dir_name, 'src', 'tarantool')
    
    # Resolve benchmark file path (relative to work_dir or absolute)
    if not os.path.isabs(benchmark_file):
        benchmark_file = os.path.join(work_dir, benchmark_file)
    
    if not os.path.exists(benchmark_file):
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_file}")
    
    # Run benchmark (use config, but remove benchmark_file as it's handled separately)
    benchmark_config_for_suite = {k: v for k, v in benchmark_config.items() if k != 'benchmark_file'}
    
    results = run_benchmark_suite(
        tarantool_binary,
        benchmark_file,
        benchmark_config_for_suite,
        work_dir
    )
    
    # Get argument order from config
    argument_order = benchmark_config.get('argument_order', 
                                          list(benchmark_config.get('arguments', {}).keys()))
    
    print(f"Completed version: {version_name} ({len(results)} successful combinations)")
    
    return results, version_name, argument_order


def write_results_csv(
    results: List[Tuple[List[Tuple[Dict[str, int], float]], str, List[str]]],
    output_file: str
) -> None:
    """Write results in CSV format."""
    print(f"\n{'='*60}")
    print("Writing results to text file (CSV format)...")
    print(f"{'='*60}")
    
    if not results:
        print("No results to write")
        return
    
    # Get argument order from first result (all should have the same order)
    _, _, argument_order = results[0]
    
    with open(output_file, 'w') as f:
        # Write header
        header = "version," + ",".join(argument_order) + ",time_per_iteration_seconds\n"
        f.write(header)
        
        # Write data for each version
        for version_results, version_name, _ in results:
            if len(version_results) == 0:
                print(f"Warning: No data for {version_name}, skipping")
                continue
            
            for arg_values, time in version_results:
                arg_values_str = ",".join([str(arg_values[arg_name]) for arg_name in argument_order])
                f.write(f"{version_name},{arg_values_str},{time}\n")
            
            print(f"Wrote data for {version_name} ({len(version_results)} combinations)")
    
    print(f"Results saved to: {output_file}")


def write_results_table(
    results: List[Tuple[List[Tuple[Dict[str, int], float]], str, List[str]]],
    output_file: str
) -> None:
    """Write results in table format."""
    print(f"\n{'='*60}")
    print("Writing results to text file (table format)...")
    print(f"{'='*60}")
    
    if not results:
        print("No results to write")
        return
    
    # Get argument order from first result (all should have the same order)
    _, _, argument_order = results[0]
    
    # Calculate column widths based on header names and data
    time_col_header = "Time per Iteration (s)"
    time_col_width = max(len(time_col_header), 25)
    
    # Calculate width for each argument column
    col_widths = {}
    for arg_name in argument_order:
        # Start with header width
        col_widths[arg_name] = len(arg_name)
    
    # Check data values to see if we need wider columns
    for version_results, _, _ in results:
        for arg_values, time in version_results:
            for arg_name in argument_order:
                value_str = str(arg_values[arg_name])
                col_widths[arg_name] = max(col_widths[arg_name], len(value_str))
    
    # Add some padding (minimum 2 spaces between columns)
    min_col_width = 12
    for arg_name in argument_order:
        col_widths[arg_name] = max(col_widths[arg_name] + 2, min_col_width)
    time_col_width = max(time_col_width + 2, 25)
    
    # Calculate total width
    total_width = sum(col_widths.values()) + time_col_width
    
    with open(output_file, 'w') as f:
        # Write header
        f.write("=" * total_width + "\n")
        f.write("Benchmark Results\n")
        f.write("=" * total_width + "\n\n")
        
        # Write data for each version
        for version_results, version_name, _ in results:
            if len(version_results) == 0:
                print(f"Warning: No data for {version_name}, skipping")
                continue
            
            f.write(f"Version: {version_name}\n")
            f.write("-" * total_width + "\n")
            
            # Write column headers
            header_parts = [f"{arg_name:<{col_widths[arg_name]}}" for arg_name in argument_order]
            header_parts.append(f"{time_col_header:<{time_col_width}}")
            f.write("".join(header_parts) + "\n")
            f.write("-" * total_width + "\n")
            
            # Write data rows
            for arg_values, time in version_results:
                row_parts = [f"{arg_values[arg_name]:<{col_widths[arg_name]}}" for arg_name in argument_order]
                row_parts.append(f"{time:<{time_col_width}.6f}")
                f.write("".join(row_parts) + "\n")
            
            f.write("\n")
            print(f"Wrote data for {version_name} ({len(version_results)} combinations)")
        
        f.write("=" * total_width + "\n")
    
    print(f"Results saved to: {output_file}")


def write_results(
    results: List[Tuple[List[Tuple[Dict[str, int], float]], str, List[str]]],
    output_config: Dict[str, Any],
    output_file: str
) -> None:
    """Write results to text file in the specified format."""
    output_format = output_config.get('format', 'csv').lower()
    
    if output_format == 'csv':
        write_results_csv(results, output_file)
    elif output_format == 'table':
        write_results_table(results, output_file)
    else:
        raise ValueError(f"Unknown output format: {output_format}. Must be 'csv' or 'table'")


def create_combined_plot(
    results: List[Tuple[List[Tuple[Dict[str, int], float]], str, List[str]]],
    plot_config: Dict[str, Any],
    output_file: str,
    benchmark_config: Dict[str, Any],
    build_mode: str
) -> None:
    """
    Create a combined plot from all benchmark results.
    X-axis is the argument with multiple values, other constants are shown as text.
    """
    print(f"\n{'='*60}")
    print("Creating combined plot...")
    print(f"{'='*60}")
    
    if not results:
        print("No results to plot")
        return
    
    # Get argument order from first result
    _, _, argument_order = results[0]
    
    # Find which argument has multiple values (the one to plot on x-axis)
    # Check the first version's results to find the varying argument
    first_version_results, _, _ = results[0]
    if not first_version_results:
        print("No data to plot")
        return
    
    # Count unique values for each argument
    arg_value_counts = {}
    for arg_values, _ in first_version_results:
        for arg_name in argument_order:
            if arg_name not in arg_value_counts:
                arg_value_counts[arg_name] = set()
            arg_value_counts[arg_name].add(arg_values[arg_name])
    
    # Find the argument with multiple values
    x_axis_arg = None
    for arg_name in argument_order:
        if len(arg_value_counts[arg_name]) > 1:
            if x_axis_arg is not None:
                raise ValueError("Multiple arguments have varying values. Only one can be plotted on x-axis.")
            x_axis_arg = arg_name
    
    if x_axis_arg is None:
        print("Warning: No argument with multiple values found. Cannot create plot.")
        return
    
    print(f"Plotting {x_axis_arg} on x-axis")
    
    # Create figure
    fig, ax = plt.subplots(figsize=plot_config.get('figure_size', [12, 8]))
    
    # Colors for plots
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    # Plot each version
    for idx, (version_results, version_name, _) in enumerate(results):
        if len(version_results) == 0:
            print(f"Warning: No data for {version_name}, skipping")
            continue
        
        # Extract x and y values
        x_values = []
        y_values = []
        
        # Sort by x-axis argument value
        sorted_results = sorted(version_results, key=lambda r: r[0][x_axis_arg])
        
        for arg_values, time in sorted_results:
            x_values.append(arg_values[x_axis_arg])
            y_values.append(time)
        
        ax.plot(
            x_values,
            y_values,
            label=version_name,
            color=colors[idx],
            linewidth=2,
            marker='o',
            markersize=4
        )
        print(f"Added plot: {version_name} ({len(x_values)} points)")
    
    # Configure plot
    ax.set_xlabel(x_axis_arg)
    ax.set_ylabel('Time per Iteration (seconds)')
    ax.set_title(plot_config.get('title', 'Benchmark Results'))
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Get constant arguments (all arguments except x_axis_arg)
    # Use values from first result
    first_arg_values = first_version_results[0][0]
    constant_args = {k: v for k, v in first_arg_values.items() if k != x_axis_arg}
    
    # Add constant arguments and build mode as text
    info_lines = []
    for arg_name, arg_value in constant_args.items():
        info_lines.append(f"{arg_name} = {arg_value}")
    
    # Add build mode
    info_lines.append(f"build mode: {build_mode}")
    
    if info_lines:
        info_text = "\n".join(info_lines)
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_file, dpi=plot_config.get('dpi', 300), bbox_inches='tight')
    print(f"Plot saved as: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Run benchmarks on multiple Tarantool versions and output results to text file'
    )
    parser.add_argument(
        'config',
        help='Path to JSON configuration file'
    )
    parser.add_argument(
        '--tarantool-path',
        default='/home/astronomax/dev/tarantool',
        help='Path to Tarantool repository (default: /home/astronomax/dev/tarantool)'
    )
    parser.add_argument(
        '--work-dir',
        default=None,
        help='Working directory for benchmarks (default: directory of config file)'
    )
    parser.add_argument(
        '--output',
        '-o',
        default=None,
        help='Output file for results (default: from config or benchmark_results.txt)'
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = os.path.abspath(args.config)
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Determine work directory
    if args.work_dir:
        work_dir = os.path.abspath(args.work_dir)
    else:
        work_dir = os.path.dirname(config_path)
    
    # Resolve tarantool path
    tarantool_path = os.path.abspath(args.tarantool_path)
    if not os.path.exists(tarantool_path):
        raise FileNotFoundError(f"Tarantool path does not exist: {tarantool_path}")
    
    # Get versions and output config
    versions = config.get('versions', [])
    output_config = config.get('output', {})
    global_benchmark_config = config.get('benchmark', {})
    global_build_mode = config.get('build_mode', 'release')
    
    # Determine output file
    output_file = args.output or output_config.get('file', 'benchmark_results.txt')
    if not os.path.isabs(output_file):
        output_file = os.path.join(work_dir, output_file)
    
    if len(versions) == 0:
        print("Error: No versions specified in config")
        sys.exit(1)
    
    print(f"Configuration loaded:")
    print(f"  Tarantool path: {tarantool_path}")
    print(f"  Work directory: {work_dir}")
    print(f"  Output file: {output_file}")
    print(f"  Output format: {output_config.get('format', 'csv')}")
    print(f"  Global build mode: {global_build_mode}")
    print(f"  Versions to test: {len(versions)}")
    
    # Process each version
    results = []
    
    for version_config in versions:
        # Merge global benchmark config with version-specific config
        merged_benchmark_config = {**global_benchmark_config, **version_config.get('benchmark_config', {})}
        version_config['benchmark_config'] = merged_benchmark_config
        
        try:
            version_results, version_name, argument_order = process_version(
                version_config,
                tarantool_path,
                work_dir,
                config
            )
            results.append((version_results, version_name, argument_order))
        except Exception as e:
            print(f"Error processing version {version_config.get('name', 'unknown')}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(results) == 0:
        print("Error: No versions were successfully processed")
        sys.exit(1)
    
    # Write results to text file
    write_results(results, output_config, output_file)
    
    # Create plot if enabled
    plot_config = config.get('plot', {})
    if plot_config.get('enabled', False):
        plot_output = plot_config.get('output', 'benchmark_plot.jpg')
        if not os.path.isabs(plot_output):
            plot_output = os.path.join(work_dir, plot_output)
        
        # Use the merged benchmark config (all versions should have the same argument structure)
        # We'll use the global config merged with the first version's overrides
        first_version_config = versions[0] if versions else {}
        merged_for_plot = {**global_benchmark_config, **first_version_config.get('benchmark_config', {})}
        
        create_combined_plot(
            results,
            plot_config,
            plot_output,
            merged_for_plot,
            global_build_mode
        )
    
    print(f"\n{'='*60}")
    print("Benchmark run completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
