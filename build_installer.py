import os
import subprocess
import shutil
import platform
import sys

# Configuration
APP_NAME = "VanityForge"
VERSION = "1.0.0"
DIST_DIR = "dist"
BIN_DIR = os.path.join(DIST_DIR, "bin")
BUILD_DIR = "build"

# Paths to source
GPU_MINER_SRC_DIR = "solanity-gpu"
CPU_MINER_SRC_DIR = "gpu-worker"
LAUNCHER_SCRIPT = "launcher.py"

def clean():
    print("Cleaning build directories...")
    if os.path.exists(DIST_DIR):
        shutil.rmtree(DIST_DIR)
    if os.path.exists(BUILD_DIR):
        shutil.rmtree(BUILD_DIR)
    os.makedirs(BIN_DIR, exist_ok=True)

def build_gpu_miner():
    print("Building GPU Miner (C++/CUDA)...")
    # Assuming 'nvcc' is in PATH.
    # On Windows, you might need to run this from a Developer Command Prompt.

    # We need to compile solanity-gpu/main.cu to dist/bin/solanity-gpu.exe
    src_file = os.path.join(GPU_MINER_SRC_DIR, "main.cu")
    output_file = os.path.join(BIN_DIR, "solanity-gpu.exe")

    if not os.path.exists(src_file):
        print(f"Error: Source file {src_file} not found.")
        return False

    # Compiler flags matching Makefile but adjusted for direct command line if needed
    # Note: Using Makefile logic is safer if 'make' is available, but for Windows we invoke nvcc directly
    # Flags: -O3 --use_fast_math --ptxas-options=-v -gencode arch=compute_89,code=sm_89 ...
    cmd = [
        "nvcc",
        "-o", output_file,
        src_file,
        "-O3",
        "--use_fast_math",
        # "--ptxas-options=-v", # Optional verbose
        "-gencode", "arch=compute_86,code=sm_86", # Ampere (RTX 30xx)
        "-gencode", "arch=compute_89,code=sm_89"  # Ada (RTX 40xx)
    ]

    try:
        subprocess.check_call(cmd)
        print(f"GPU Miner built: {output_file}")
        return True
    except subprocess.CalledProcessError:
        print("Failed to build GPU miner.")
        return False
    except FileNotFoundError:
        print("Error: 'nvcc' not found. Ensure CUDA Toolkit is installed and in PATH.")
        return False

def build_cpu_miner():
    print("Building CPU Miner (Rust)...")
    # Assuming 'cargo' is in PATH

    # Run cargo build --release in gpu-worker/
    try:
        subprocess.check_call(["cargo", "build", "--release"], cwd=CPU_MINER_SRC_DIR)
    except subprocess.CalledProcessError:
        print("Failed to build CPU miner.")
        return False
    except FileNotFoundError:
        print("Error: 'cargo' not found. Ensure Rust is installed.")
        return False

    # Copy binary
    # Rust output is in target/release/vanity-grinder.exe (on Windows)
    src_bin = os.path.join(CPU_MINER_SRC_DIR, "target", "release", "vanity-grinder.exe")
    # If on Linux for testing, it won't have .exe
    if not os.path.exists(src_bin) and os.path.exists(src_bin[:-4]):
        src_bin = src_bin[:-4]

    if os.path.exists(src_bin):
        dest_bin = os.path.join(BIN_DIR, "solanity-cpu.exe")
        shutil.copy2(src_bin, dest_bin)
        print(f"CPU Miner built: {dest_bin}")
        return True
    else:
        print(f"Error: Rust binary not found at {src_bin}")
        return False

def freeze_launcher():
    print("Freezing Launcher (PyInstaller)...")

    # pyinstaller --onefile --noconsole --name VanityForge launcher.py --distpath dist
    # We put the exe in dist/ (root of distribution)
    # The 'bin' folder is already in dist/bin, so it will sit next to the exe

    cmd = [
        "pyinstaller",
        "--onefile",
        "--noconsole",
        "--name", APP_NAME,
        "--distpath", DIST_DIR,
        "--workpath", BUILD_DIR,
        "--clean",
        LAUNCHER_SCRIPT
    ]

    try:
        subprocess.check_call(cmd)
        print("Launcher frozen successfully.")
        return True
    except FileNotFoundError:
        print("Error: 'pyinstaller' not found. pip install pyinstaller.")
        return False
    except subprocess.CalledProcessError:
        print("Failed to freeze launcher.")
        return False

def create_installer_script():
    print("Creating Inno Setup Script...")

    # Absolute paths required for Inno Setup sometimes, but relative works if run from root
    # We want to package everything in dist/

    iss_content = f"""
[Setup]
AppName={APP_NAME}
AppVersion={VERSION}
DefaultDirName={{autopf}}\\{APP_NAME}
DefaultGroupName={APP_NAME}
UninstallDisplayIcon={{app}}\\{APP_NAME}.exe
Compression=lzma2
SolidCompression=yes
OutputDir=installer
OutputBaseFilename={APP_NAME}_Setup_{VERSION}
; "ArchitecturesInstallIn64BitMode=x64" requests that the install be
; done in "64-bit mode" on x64, meaning it should use the native
; 64-bit Program Files directory and the 64-bit view of the registry.
ArchitecturesInstallIn64BitMode=x64

[Files]
; The main executable
Source: "dist\\{APP_NAME}.exe"; DestDir: "{{app}}"; Flags: ignoreversion
; The bin folder with miners
Source: "dist\\bin\\*"; DestDir: "{{app}}\\bin"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{{group}}\\{APP_NAME}"; Filename: "{{app}}\\{APP_NAME}.exe"
Name: "{{autodesktop}}\\{APP_NAME}"; Filename: "{{app}}\\{APP_NAME}.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "{APP_NAME}"; GroupDescription: "Additional icons:"; Flags: unchecked

[Run]
Filename: "{{app}}\\{APP_NAME}.exe"; Description: "Launch {APP_NAME}"; Flags: nowait postinstall skipifsilent
"""

    with open("setup.iss", "w") as f:
        f.write(iss_content)

    return True

def run_iscc():
    print("Compiling Installer (ISCC)...")
    # check for ISCC.exe in path
    try:
        subprocess.check_call(["ISCC", "setup.iss"])
        print("Installer created successfully in installer/ folder.")
    except FileNotFoundError:
        print("Error: 'ISCC' not found. Ensure Inno Setup is installed and in PATH.")
        print("You can still compile the installer manually using 'setup.iss'.")

def main():
    if platform.system() != "Windows":
        print("Warning: This build script is intended for Windows.")
        print("Some steps (nvcc, pyinstaller .exe, ISCC) may fail or produce invalid artifacts on Linux.")

    clean()

    # Step A: Compile Miners
    # We continue even if one fails, for partial builds/testing
    if not build_gpu_miner():
        print("Warning: GPU miner build failed or skipped.")

    if not build_cpu_miner():
        print("Warning: CPU miner build failed or skipped.")

    # Step B: Freeze GUI
    if not freeze_launcher():
        print("Error: Launcher build failed.")
        sys.exit(1)

    # Step C: Create Installer
    create_installer_script()
    run_iscc()

    print("\nBuild process finished.")
    print(f"Check the '{DIST_DIR}' folder for the portable executables.")
    print("Check 'setup.iss' for the installer script.")

if __name__ == "__main__":
    main()
