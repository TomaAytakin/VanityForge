import os
import shutil
import subprocess
import sys

def run_command(command, cwd=None, shell=False):
    """Runs a shell command and prints its output."""
    print(f"Running: {' '.join(command) if isinstance(command, list) else command}")
    try:
        subprocess.check_call(command, cwd=cwd, shell=shell)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command}")
        raise e

def main():
    dist_dir = "dist"
    cpu_miner_dir = "redpanda-cpu"
    gpu_miner_src = os.path.join("solanity-gpu", "main.cu")
    launcher_script = "launcher.py"

    # 1. Setup
    print("--- Step 1: Setup ---")
    if os.path.exists(dist_dir):
        print(f"Removing existing '{dist_dir}' directory...")
        shutil.rmtree(dist_dir)
    os.makedirs(dist_dir)
    print(f"Created '{dist_dir}' directory.")

    # 2. Compile CPU Miner
    print("\n--- Step 2: Compile CPU Miner ---")
    # Check if CPU miner directory exists
    if not os.path.exists(cpu_miner_dir):
        print(f"Error: Directory '{cpu_miner_dir}' not found. Cannot compile CPU miner.")
        sys.exit(1)

    try:
        run_command(["cargo", "build", "--release"], cwd=cpu_miner_dir)

        # Source path for the compiled binary
        # Assuming standard cargo layout: target/release/redpanda-cpu.exe
        # Note: On non-Windows systems, this might just be 'redpanda-cpu'
        cpu_bin_name = "redpanda-cpu.exe"
        src_cpu_bin = os.path.join(cpu_miner_dir, "target", "release", cpu_bin_name)

        # If strictly for Windows build automation as requested, we look for .exe
        if not os.path.exists(src_cpu_bin):
            # Fallback check for extensionless binary if running on Linux for verification
            # But the user asked for a Windows release script.
            print(f"Warning: Expected binary '{src_cpu_bin}' not found.")
            # We try to copy anyway to trigger the error if it's really missing

        dst_cpu_bin = os.path.join(dist_dir, cpu_bin_name)
        shutil.copy2(src_cpu_bin, dst_cpu_bin)
        print(f"Successfully compiled and copied '{cpu_bin_name}' into '{dist_dir}/'.")

    except Exception as e:
        print(f"Failed to compile CPU miner: {e}")
        # "handle errors ... gracefully" - we exit here because the miner is core
        sys.exit(1)

    # 3. Compile GPU Miner
    print("\n--- Step 3: Compile GPU Miner ---")
    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        gpu_bin_name = "solanity-gpu.exe"
        dst_gpu_bin = os.path.join(dist_dir, gpu_bin_name)

        try:
            # Compile solanity-gpu/main.cu into dist/solanity-gpu.exe
            # Using basic flags. Add optimization if needed, e.g. -O3
            run_command(["nvcc", gpu_miner_src, "-o", dst_gpu_bin, "-O3"])
            print(f"Successfully compiled '{gpu_bin_name}' into '{dist_dir}/'.")
        except Exception as e:
            print(f"Warning: Failed to compile GPU miner: {e}")
            # Continue as per instructions
    else:
        print("Warning: 'nvcc' not found in system path. Skipping GPU miner compilation.")

    # 4. Bundle GUI
    print("\n--- Step 4: Bundle GUI ---")
    try:
        # PyInstaller command: pyinstaller --onefile --noconsole --name VanityForge_GUI launcher.py
        # By default, PyInstaller puts the output in dist/.
        # Since our dist_dir is also 'dist', it matches.
        run_command(["pyinstaller", "--onefile", "--noconsole", "--name", "VanityForge_GUI", launcher_script])

        # Verify the file is in dist/
        gui_bin_name = "VanityForge_GUI.exe"
        expected_output = os.path.join("dist", gui_bin_name)

        if os.path.exists(expected_output):
             # Move? It is already in dist/.
             # If PyInstaller output to a different dist (e.g. if we changed spec), we'd move.
             # But here we just verify.
             print(f"Successfully bundled '{gui_bin_name}' into '{dist_dir}/'.")
        else:
            print(f"Error: Expected output '{expected_output}' not found after PyInstaller run.")
            sys.exit(1)

    except Exception as e:
        print(f"Failed to bundle GUI: {e}")
        sys.exit(1)

    # 5. Cleanup
    print("\n--- Step 5: Cleanup ---")
    build_dir = "build"
    spec_file = "VanityForge_GUI.spec"

    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
        print(f"Removed '{build_dir}' directory.")

    if os.path.exists(spec_file):
        os.remove(spec_file)
        print(f"Removed '{spec_file}'.")

    print("\nBuild process completed successfully.")

if __name__ == "__main__":
    main()
