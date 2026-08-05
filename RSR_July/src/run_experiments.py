"""
Batch Orchestration Module for RSR Experiments

This script acts as the top-level test runner, designed to execute 
all four variations of the model (vanilla_all, vanilla_targetwords, rsr_all, rsr_targetwords) 
across multiple random seeds for rigorous statistical evaluation.
It supports both sequential and concurrent (parallel) execution.

Workflow Context:
- The highest level entry point for running the main experimental suite.
- Invokes: `train.py` as a subprocess (sequentially or concurrently).
- Followed by: `plot_results.py` to aggregate and visualize the generated logs.
"""

import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Batch Orchestration of RSR Experiments")
    parser.add_argument("--steps", type=int, default=2000, help="Number of training steps per run.")
    parser.add_argument("--replicates", type=int, default=5, help="Number of statistical replicates.")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save training/eval logs.")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save timing logs.")
    parser.add_argument("--add_date", action="store_true", help="Append current date (YYYY-MM-DD) to results_dir and log_dir.")
    parser.add_argument("--add_host", action="store_true", help="Append hostname and current date (e.g. _hostname_YYYY-MM-DD) to results_dir and log_dir.")
    parser.add_argument("--eval_mode", type=str, default="bare", choices=["bare", "wiki_avg"], help="Evaluation mode.")
    parser.add_argument("--rsr_layer", type=int, default=None, help="The layer to apply RSR regularisation to.")
    parser.add_argument("--eval_steps", type=int, default=400, help="How often to run evaluation.")
    parser.add_argument("--parallel", action="store_true", help="Run the training modes concurrently.")
    
    # Check if positional arguments are passed instead (for backward compatibility)
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        # Parse manually as positional args: steps [replicates] [results_dir] [eval_mode] [rsr_layer] [eval_steps]
        args = parser.parse_args([]) # get defaults
        try:
            args.steps = int(sys.argv[1])
        except ValueError:
            pass
        if len(sys.argv) > 2:
            try:
                args.replicates = int(sys.argv[2])
            except ValueError:
                pass
        if len(sys.argv) > 3:
            args.results_dir = sys.argv[3]
        if len(sys.argv) > 4:
            args.eval_mode = sys.argv[4]
        if len(sys.argv) > 5:
            try:
                args.rsr_layer = int(sys.argv[5]) if sys.argv[5] != 'None' else None
            except ValueError:
                pass
        if len(sys.argv) > 6:
            try:
                args.eval_steps = int(sys.argv[6])
            except ValueError:
                pass
    else:
        args = parser.parse_args()

    # Resolve folder suffix (hostname and/or date)
    suffix = ""
    if args.add_host:
        import socket
        hostname = socket.gethostname().lower()
        suffix += f"_{hostname}"
    if args.add_date or args.add_host:
        from datetime import datetime
        date_str = datetime.today().strftime('%Y-%m-%d')
        suffix += f"_{date_str}"

    if suffix:
        args.results_dir = args.results_dir.rstrip('/\\') + suffix
        args.log_dir = args.log_dir.rstrip('/\\') + suffix

    # Save exact command line arguments to log folder
    os.makedirs(args.log_dir, exist_ok=True)
    params_file = os.path.join(args.log_dir, "run_params_experiments.log")
    with open(params_file, "w", encoding="utf-8") as f:
        f.write(f"Command line: {' '.join(sys.argv)}\n")
        f.write(f"Raw sys.argv: {sys.argv}\n")
        f.write(f"Parsed arguments: {vars(args)}\n")

    modes = [
        "vanilla_all",
        "vanilla_targetwords",
        "rsr_all",
        "rsr_targetwords"
    ]
    
    print(f"Starting orchestration of {len(modes)} training runs for {args.steps} steps each, across {args.replicates} replicates...")
    print(f"Results Directory: {args.results_dir} | Log Directory: {args.log_dir}")
    print(f"Eval Mode: {args.eval_mode} | RSR Layer: {args.rsr_layer} | Eval Steps: {args.eval_steps}")
    
    # Rename existing old logs to _run0.txt if they exist
    os.makedirs(args.results_dir, exist_ok=True)
    for mode in modes:
        old_log = os.path.join(args.results_dir, f"eval_log_{mode}.txt")
        new_log = os.path.join(args.results_dir, f"eval_log_{mode}_run0.txt")
        if os.path.exists(old_log):
            os.rename(old_log, new_log)
            
    for run_id in range(1, args.replicates + 1):
        print(f"\n{'#'*60}")
        print(f"STARTING REPLICATE RUN {run_id}/{args.replicates}")
        print(f"{'#'*60}")
        
        if args.parallel:
            active_processes = []
            try:
                for mode in modes:
                    print(f"Launching Training Run: {mode} (Run {run_id})")
                    
                    # Build command
                    cmd = [
                        sys.executable, 
                        os.path.join("src", "train.py"), 
                        "--mode", mode, 
                        "--steps", str(args.steps), 
                        "--run_id", str(run_id),
                        "--results_dir", args.results_dir,
                        "--log_dir", args.log_dir,
                        "--eval_mode", args.eval_mode,
                        "--eval_steps", str(args.eval_steps)
                    ]
                    if args.rsr_layer is not None:
                        cmd.extend(["--rsr_layer", str(args.rsr_layer)])
                        
                    # Execute and stream output to log files
                    out_log_path = os.path.join(args.log_dir, f"train_stdout_{mode}_run{run_id}.txt")
                    err_log_path = os.path.join(args.log_dir, f"train_stderr_{mode}_run{run_id}.txt")
                    
                    os.makedirs(args.log_dir, exist_ok=True)
                    out_file = open(out_log_path, "w")
                    err_file = open(err_log_path, "w")
                    
                    process = subprocess.Popen(cmd, stdout=out_file, stderr=err_file)
                    active_processes.append((mode, process, out_file, err_file))
                    
                print(f"\nWaiting for all {len(modes)} training modes to finish concurrently for Run {run_id}...\n")
                
                # Wait for all processes to complete
                for mode, process, out_file, err_file in active_processes:
                    process.communicate()
                    out_file.close()
                    err_file.close()
                    
                    if process.returncode != 0:
                        print(f"Error: Training run for {mode} failed with exit code {process.returncode}. Check logs in {args.log_dir}.")
                        sys.exit(1)
                    else:
                        print(f"Success: {mode} (Run {run_id}) completed.")
            finally:
                # Ensure all file handles are closed even if an exception occurs
                for _, _, out_file, err_file in active_processes:
                    if not out_file.closed:
                        out_file.close()
                    if not err_file.closed:
                        err_file.close()
        else:
            for mode in modes:
                print(f"Running Training Run: {mode} (Run {run_id})...")
                
                # Build command
                cmd = [
                    sys.executable, 
                    os.path.join("src", "train.py"), 
                    "--mode", mode, 
                    "--steps", str(args.steps), 
                    "--run_id", str(run_id),
                    "--results_dir", args.results_dir,
                    "--log_dir", args.log_dir,
                    "--eval_mode", args.eval_mode,
                    "--eval_steps", str(args.eval_steps)
                ]
                if args.rsr_layer is not None:
                    cmd.extend(["--rsr_layer", str(args.rsr_layer)])
                    
                # Execute and stream output to log files
                out_log_path = os.path.join(args.log_dir, f"train_stdout_{mode}_run{run_id}.txt")
                err_log_path = os.path.join(args.log_dir, f"train_stderr_{mode}_run{run_id}.txt")
                
                os.makedirs(args.log_dir, exist_ok=True)
                with open(out_log_path, "w") as out_file, open(err_log_path, "w") as err_file:
                    result = subprocess.run(cmd, stdout=out_file, stderr=err_file)
                    
                if result.returncode != 0:
                    print(f"Error: Training run for {mode} failed with exit code {result.returncode}. Check logs in {args.log_dir}.")
                    sys.exit(1)
                else:
                    print(f"Success: {mode} (Run {run_id}) completed.")
                
    print(f"\nAll {args.replicates} statistical replicates completed successfully!")
    
    # Run plot_results.py to aggregate and visualize
    print("\nRunning plot_results.py to aggregate and visualize trajectories...")
    plot_cmd = [sys.executable, os.path.join("src", "plot_results.py"), "--results_dir", args.results_dir]
    subprocess.run(plot_cmd)

if __name__ == "__main__":
    main()
