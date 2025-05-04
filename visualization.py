#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import re
import argparse

def parse_timing_log(log_file):
    """Parse timing information from log file"""
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract number of processes
    procs_match = re.search(r'Running SSSP with (\d+) processes', content)
    num_procs = int(procs_match.group(1)) if procs_match else None
    
    # Extract computation time
    time_match = re.search(r'Computation completed in ([\d.]+) seconds', content)
    compute_time = float(time_match.group(1)) if time_match else None
    
    # Extract vertices info
    vertices_match = re.search(r'Reachable vertices from source \d+: (\d+) out of (\d+)', content)
    reachable = int(vertices_match.group(1)) if vertices_match else None
    total = int(vertices_match.group(2)) if vertices_match else None
    
    return {
        'num_procs': num_procs,
        'compute_time': compute_time,
        'reachable': reachable,
        'total': total
    }

def plot_scaling(log_files, output_dir):
    """Create scaling plots from log files"""
    data = []
    for log_file in log_files:
        result = parse_timing_log(log_file)
        if result['num_procs'] and result['compute_time']:
            data.append(result)
    
    # Sort by number of processes
    data.sort(key=lambda x: x['num_procs'])
    
    # Extract data for plotting
    procs = [item['num_procs'] for item in data]
    times = [item['compute_time'] for item in data]
    
    # Calculate speedup and efficiency
    base_time = times[0] if times else 1
    speedup = [base_time / time for time in times]
    efficiency = [s / p for s, p in zip(speedup, procs)]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot compute time
    plt.figure(figsize=(10, 6))
    plt.plot(procs, times, 'o-', linewidth=2)
    plt.xlabel('Number of Processes')
    plt.ylabel('Computation Time (seconds)')
    plt.grid(True)
    plt.title('Scaling Performance: Computation Time')
    plt.savefig(os.path.join(output_dir, 'compute_time.png'))
    
    # Plot speedup
    plt.figure(figsize=(10, 6))
    plt.plot(procs, speedup, 'o-', linewidth=2)
    plt.plot(procs, procs, '--', linewidth=1, label='Ideal Speedup')
    plt.xlabel('Number of Processes')
    plt.ylabel('Speedup')
    plt.grid(True)
    plt.legend()
    plt.title('Scaling Performance: Speedup')
    plt.savefig(os.path.join(output_dir, 'speedup.png'))
    
    # Plot efficiency
    plt.figure(figsize=(10, 6))
    plt.plot(procs, efficiency, 'o-', linewidth=2)
    plt.xlabel('Number of Processes')
    plt.ylabel('Efficiency')
    plt.grid(True)
    plt.title('Scaling Performance: Efficiency')
    plt.axhline(y=1.0, linestyle='--', color='r', linewidth=1)
    plt.savefig(os.path.join(output_dir, 'efficiency.png'))
    
    print(f"Plots saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Create scaling plots from log files')
    parser.add_argument('log_files', nargs='+', help='Log files to process')
    parser.add_argument('--output', '-o', default='plots', help='Output directory for plots')
    args = parser.parse_args()
    
    plot_scaling(args.log_files, args.output)

if __name__ == '__main__':
    main()
