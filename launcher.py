import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import threading
import sys
import os
import platform
import json
import time
import requests
import zipfile
import io
import shutil

# Configuration
# Replace with your actual GitHub User and Repo
GITHUB_REPO_URL = "https://api.github.com/repos/YourUser/VanityForge/releases/latest"
BIN_DIR = "bin"
GPU_MINER_NAME = "solanity-gpu.exe" if platform.system() == "Windows" else "solanity-gpu"
CPU_MINER_NAME = "solanity-cpu.exe" if platform.system() == "Windows" else "solanity-cpu"

class VanityForgeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("VanityForge Studio")
        self.root.geometry("800x600")
        self.root.configure(bg="#1e1e1e")

        # Styling
        self.style = ttk.Style()
        self.style.theme_use("clam")

        # Dark Theme Colors
        self.colors = {
            "bg": "#1e1e1e",
            "fg": "#ffffff",
            "accent": "#007acc",
            "success": "#4caf50",
            "warning": "#ff9800",
            "panel": "#2d2d2d",
            "text_box": "#111111"
        }

        self.style.configure(".", background=self.colors["bg"], foreground=self.colors["fg"], fieldbackground=self.colors["text_box"])
        self.style.configure("TLabel", background=self.colors["bg"], foreground=self.colors["fg"], font=("Segoe UI", 10))
        self.style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"))
        self.style.configure("TButton", padding=6, relief="flat", background=self.colors["panel"])
        self.style.map("TButton", background=[("active", self.colors["accent"])])
        self.style.configure("TFrame", background=self.colors["bg"])
        self.style.configure("Panel.TFrame", background=self.colors["panel"])

        # State
        self.miner_process = None
        self.is_mining = False
        self.miner_thread = None
        self.miner_mode = tk.StringVar(value="GPU")

        # UI Components
        self.create_header()
        self.create_hardware_info()
        self.create_controls()
        self.create_dashboard()
        self.create_key_display()
        self.create_footer()

        # Initialize
        self.detect_hardware()

    def create_header(self):
        header_frame = ttk.Frame(self.root, padding="20 20 20 0")
        header_frame.pack(fill=tk.X)

        title_label = ttk.Label(header_frame, text="VanityForge Studio", style="Header.TLabel")
        title_label.pack(side=tk.LEFT)

        update_btn = ttk.Button(header_frame, text="Check for Updates", command=self.check_for_updates)
        update_btn.pack(side=tk.RIGHT)

    def create_hardware_info(self):
        info_frame = ttk.Frame(self.root, padding="20 10")
        info_frame.pack(fill=tk.X)

        self.cpu_label = ttk.Label(info_frame, text="CPU: Detecting...", foreground="#aaaaaa")
        self.cpu_label.pack(anchor=tk.W)

        self.gpu_label = ttk.Label(info_frame, text="GPU: Detecting...", foreground="#aaaaaa")
        self.gpu_label.pack(anchor=tk.W)

    def create_controls(self):
        control_frame = ttk.Frame(self.root, padding="20 10")
        control_frame.pack(fill=tk.X)

        # Mode Switch
        mode_frame = ttk.LabelFrame(control_frame, text="Mining Mode", padding="10")
        mode_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))

        ttk.Radiobutton(mode_frame, text="GPU Miner (C++)", variable=self.miner_mode, value="GPU", command=self.on_mode_change).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="CPU Miner (Rust)", variable=self.miner_mode, value="CPU", command=self.on_mode_change).pack(anchor=tk.W)

        # Start/Stop Buttons
        action_frame = ttk.Frame(control_frame)
        action_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.start_btn = tk.Button(action_frame, text="START MINING", bg=self.colors["success"], fg="white",
                                   font=("Segoe UI", 12, "bold"), relief="flat", padx=20, pady=10,
                                   command=self.start_mining)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.stop_btn = tk.Button(action_frame, text="STOP", bg=self.colors["warning"], fg="white",
                                  font=("Segoe UI", 12, "bold"), relief="flat", padx=20, pady=10,
                                  command=self.stop_mining, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT)

    def create_dashboard(self):
        dash_frame = ttk.Frame(self.root, padding="20 10")
        dash_frame.pack(fill=tk.X)

        # Speed Meter
        self.speed_label = ttk.Label(dash_frame, text="0 MH/s", font=("Segoe UI", 24, "bold"), foreground=self.colors["accent"])
        self.speed_label.pack()

        ttk.Label(dash_frame, text="Current Speed").pack()

        # Log/Status
        self.status_label = ttk.Label(dash_frame, text="Ready", foreground="#888888")
        self.status_label.pack(pady=(10, 0))

    def create_key_display(self):
        key_frame = ttk.LabelFrame(self.root, text="Found Wallet", padding="20")
        key_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Private Key
        ttk.Label(key_frame, text="Private Key:").pack(anchor=tk.W)
        self.priv_key_entry = tk.Entry(key_frame, bg=self.colors["text_box"], fg=self.colors["success"],
                                       font=("Consolas", 10), relief="flat", insertbackground="white")
        self.priv_key_entry.pack(fill=tk.X, pady=(0, 10))

        # Public Key
        ttk.Label(key_frame, text="Public Key:").pack(anchor=tk.W)
        self.pub_key_entry = tk.Entry(key_frame, bg=self.colors["text_box"], fg=self.colors["accent"],
                                      font=("Consolas", 10), relief="flat", insertbackground="white")
        self.pub_key_entry.pack(fill=tk.X)

    def create_footer(self):
        footer_frame = ttk.Frame(self.root, padding="10")
        footer_frame.pack(fill=tk.X, side=tk.BOTTOM)
        ttk.Label(footer_frame, text="Powered by Solanity Engine", font=("Segoe UI", 8)).pack(side=tk.RIGHT)

    def detect_hardware(self):
        cpu_name = "Unknown"
        gpu_name = "Unknown"

        if platform.system() == "Windows":
            try:
                cpu_cmd = 'wmic cpu get name'
                cpu_out = subprocess.check_output(cpu_cmd, shell=True).decode().split('\n')[1].strip()
                if cpu_out: cpu_name = cpu_out

                gpu_cmd = 'wmic path win32_videocontroller get name'
                gpu_out = subprocess.check_output(gpu_cmd, shell=True).decode().split('\n')[1].strip()
                if gpu_out: gpu_name = gpu_out
            except Exception as e:
                print(f"Error detecting hardware: {e}")
        else:
            # Linux Fallback
            try:
                # Try to get CPU info on Linux
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            cpu_name = line.split(":")[1].strip()
                            break
                # Simple GPU check via lspci
                try:
                    gpu_out = subprocess.check_output("lspci | grep -i vga", shell=True).decode()
                    if ":" in gpu_out:
                        gpu_name = gpu_out.split(":")[2].strip()
                    else:
                        gpu_name = "Linux GPU"
                except:
                    gpu_name = "Linux GPU (Generic)"
            except:
                pass

        self.cpu_label.config(text=f"CPU: {cpu_name}")
        self.gpu_label.config(text=f"GPU: {gpu_name}")

    def on_mode_change(self):
        if self.is_mining:
            messagebox.showinfo("Mining Active", "Please stop mining before switching modes.")
            # Revert selection
            current = self.miner_mode.get()
            self.miner_mode.set("CPU" if current == "GPU" else "GPU")

    def start_mining(self):
        if self.is_mining: return

        miner_bin = GPU_MINER_NAME if self.miner_mode.get() == "GPU" else CPU_MINER_NAME
        miner_path = os.path.join(BIN_DIR, miner_bin)

        if not os.path.exists(miner_path):
            # Fallback to check if binary is in current dir
            if os.path.exists(miner_bin):
                miner_path = miner_bin
            else:
                messagebox.showerror("Error", f"Miner executable not found: {miner_path}\nPlease run updates or build the project.")
                return

        try:
            # Prepare arguments
            # Note: Add logic to pass parameters to the miner (e.g., prefix)
            args = [miner_path]

            # Create subprocess
            creation_flags = 0
            if platform.system() == "Windows":
                creation_flags = subprocess.CREATE_NO_WINDOW

            self.miner_process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                creationflags=creation_flags
            )

            self.is_mining = True
            self.start_btn.config(state=tk.DISABLED, bg=self.colors["panel"])
            self.stop_btn.config(state=tk.NORMAL, bg=self.colors["warning"])
            self.status_label.config(text="Mining started...")

            # Start monitoring thread
            self.miner_thread = threading.Thread(target=self.monitor_miner, daemon=True)
            self.miner_thread.start()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start miner: {e}")

    def stop_mining(self):
        if not self.is_mining or not self.miner_process:
            return

        self.status_label.config(text="Stopping...")

        # Kill process
        try:
            self.miner_process.terminate()
            self.miner_process = None
        except:
            pass

        self.is_mining = False
        self.start_btn.config(state=tk.NORMAL, bg=self.colors["success"])
        self.stop_btn.config(state=tk.DISABLED, bg=self.colors["panel"])
        self.status_label.config(text="Stopped")
        self.speed_label.config(text="0 MH/s")

    def monitor_miner(self, process=None):
        while self.is_mining and self.miner_process:
            try:
                line = self.miner_process.stdout.readline()
                if not line:
                    break

                line = line.strip()
                if not line: continue

                # Speed Parsing Logic
                if "MH/s" in line:
                    try:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if "MH/s" in part:
                                speed = parts[i-1]
                                self.root.after(0, lambda s=speed: self.speed_label.config(text=f"{s} MH/s"))
                    except:
                        pass

                # JSON Success Logic
                if "{" in line and "}" in line and "private_key" in line:
                    try:
                        start = line.find('{')
                        end = line.rfind('}') + 1
                        json_str = line[start:end]
                        data = json.loads(json_str)

                        if "private_key" in data and "public_key" in data:
                            self.root.after(0, lambda d=data: self.display_success(d))
                    except:
                        pass
            except Exception:
                break

        if self.is_mining:
            self.root.after(0, self.stop_mining)

    def display_success(self, data):
        self.priv_key_entry.delete(0, tk.END)
        self.priv_key_entry.insert(0, data["private_key"])

        self.pub_key_entry.delete(0, tk.END)
        self.pub_key_entry.insert(0, data["public_key"])

        self.status_label.config(text="KEY FOUND!", foreground=self.colors["success"])
        messagebox.showinfo("Success", "Vanity Address Found!")
        self.stop_mining()

    def check_for_updates(self):
        self.status_label.config(text="Checking for updates...")
        threading.Thread(target=self._perform_update, daemon=True).start()

    def _perform_update(self):
        try:
            # 1. Fetch latest release info
            try:
                response = requests.get(GITHUB_REPO_URL)
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                # If placeholder URL fails, we show message but don't crash
                self.root.after(0, lambda: messagebox.showinfo("Update Check", "Could not connect to update server.\n(Configure GITHUB_REPO_URL in source)"))
                self.root.after(0, lambda: self.status_label.config(text="Update check failed."))
                return

            assets = data.get("assets", [])
            downloaded = 0

            if not os.path.exists(BIN_DIR):
                os.makedirs(BIN_DIR)

            for asset in assets:
                name = asset["name"]
                if name in [GPU_MINER_NAME, CPU_MINER_NAME]:
                    download_url = asset["browser_download_url"]
                    self.root.after(0, lambda n=name: self.status_label.config(text=f"Downloading {n}..."))

                    # Download
                    r = requests.get(download_url)
                    with open(os.path.join(BIN_DIR, name), "wb") as f:
                        f.write(r.content)
                    downloaded += 1

            if downloaded > 0:
                self.root.after(0, lambda: messagebox.showinfo("Update Complete", f"Successfully updated {downloaded} components.\nMiners are ready."))
                self.root.after(0, lambda: self.status_label.config(text="Update complete."))
            else:
                self.root.after(0, lambda: messagebox.showinfo("No Updates", "No matching binaries found in the latest release."))
                self.root.after(0, lambda: self.status_label.config(text="Up to date."))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Update Error", str(e)))
            self.root.after(0, lambda: self.status_label.config(text="Error."))

if __name__ == "__main__":
    if not os.path.exists(BIN_DIR):
        os.makedirs(BIN_DIR)

    root = tk.Tk()
    app = VanityForgeApp(root)
    root.mainloop()
