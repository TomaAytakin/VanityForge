import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import subprocess
import threading
import sys
import os
import platform
import json
import queue

class VanityForgeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("VanityForge Studio")
        self.root.geometry("600x700")

        # Style
        style = ttk.Style()
        style.theme_use('clam')

        # Variables
        self.prefix_var = tk.StringVar()
        self.suffix_var = tk.StringVar()
        self.case_sensitive_var = tk.BooleanVar()
        self.mode_var = tk.StringVar(value="CPU") # Default to CPU
        self.is_running = False
        self.process = None
        self.log_queue = queue.Queue()

        # UI Layout
        self.create_widgets()

        # Periodic check for logs
        self.root.after(100, self.process_logs)

        # Safety: Ensure process is killed on close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        # Header
        header = ttk.Label(self.root, text="VanityForge Studio", font=("Helvetica", 16, "bold"))
        header.pack(pady=10)

        # Inputs Frame
        input_frame = ttk.LabelFrame(self.root, text="Settings", padding="10")
        input_frame.pack(fill="x", padx=10, pady=5)

        # Prefix
        ttk.Label(input_frame, text="Prefix (Starts with):").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(input_frame, textvariable=self.prefix_var).grid(row=0, column=1, sticky="ew", padx=5, pady=2)

        # Suffix
        ttk.Label(input_frame, text="Suffix (Ends with):").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(input_frame, textvariable=self.suffix_var).grid(row=1, column=1, sticky="ew", padx=5, pady=2)

        # Case Sensitive
        ttk.Checkbutton(input_frame, text="Case Sensitive", variable=self.case_sensitive_var).grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=5)

        input_frame.columnconfigure(1, weight=1)

        # Mode Frame
        mode_frame = ttk.LabelFrame(self.root, text="Mode", padding="10")
        mode_frame.pack(fill="x", padx=10, pady=5)

        ttk.Radiobutton(mode_frame, text="CPU (RedPanda)", variable=self.mode_var, value="CPU").pack(anchor="w", padx=5)
        ttk.Radiobutton(mode_frame, text="GPU (Solanity)", variable=self.mode_var, value="GPU").pack(anchor="w", padx=5)

        # Controls Frame
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill="x", padx=10, pady=5)

        self.start_btn = ttk.Button(control_frame, text="Start Forge", command=self.start_mining)
        self.start_btn.pack(side="left", fill="x", expand=True, padx=5)

        self.stop_btn = ttk.Button(control_frame, text="Stop", command=self.stop_mining, state="disabled")
        self.stop_btn.pack(side="right", fill="x", expand=True, padx=5)

        # Logs Frame
        log_frame = ttk.LabelFrame(self.root, text="Logs", padding="10")
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.log_area = scrolledtext.ScrolledText(log_frame, state="disabled", height=15)
        self.log_area.pack(fill="both", expand=True)

    def start_mining(self):
        if self.is_running:
            return

        prefix = self.prefix_var.get().strip()
        suffix = self.suffix_var.get().strip()
        case_sensitive = self.case_sensitive_var.get()
        mode = self.mode_var.get()

        # Determine Executable
        if mode == "CPU":
            binary_name = "redpanda-cpu"
        else:
            binary_name = "solanity-gpu"

        if platform.system() == "Windows":
            binary_name += ".exe"

        # Check if binary exists (optional, but good practice)
        if not os.path.exists(binary_name) and not any(os.access(os.path.join(path, binary_name), os.X_OK) for path in os.environ["PATH"].split(os.pathsep)):
            # We will still try to run it, but log a warning if not found in cwd
            self.log_message(f"Warning: {binary_name} not found in current directory. Attempting to run from PATH...")

        # Construct Command
        cmd = [binary_name]
        if prefix:
            cmd.extend(["--prefix", prefix])
        if suffix:
            cmd.extend(["--suffix", suffix])
        if case_sensitive:
            cmd.append("--case-sensitive")

        self.log_message(f"Starting {mode} Forge...")
        self.log_message(f"Command: {' '.join(cmd)}")

        try:
            # On Windows, creationflags=subprocess.CREATE_NO_WINDOW hides the console window
            creation_flags = 0
            if platform.system() == "Windows":
                creation_flags = subprocess.CREATE_NO_WINDOW

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                creationflags=creation_flags
            )

            self.is_running = True
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")

            # Start monitoring thread
            threading.Thread(target=self.monitor_output, daemon=True).start()

        except Exception as e:
            self.log_message(f"Error starting process: {e}")
            messagebox.showerror("Error", f"Failed to start miner:\n{e}")

    def stop_mining(self):
        if self.process and self.is_running:
            self.log_message("Stopping...")
            try:
                self.process.terminate()
            except Exception as e:
                self.log_message(f"Error stopping process: {e}")

            self.process = None
            self.is_running = False
            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            self.log_message("Stopped.")

    def monitor_output(self):
        while self.is_running and self.process:
            try:
                line = self.process.stdout.readline()
                if not line and self.process.poll() is not None:
                    break

                if line:
                    self.log_queue.put({"type": "log", "content": line.strip()})

                    # Success Detection
                    if '"public_key"' in line and '"secret_key"' in line:
                         try:
                             # Extract JSON part if mixed with text
                             start_idx = line.find('{')
                             end_idx = line.rfind('}') + 1
                             if start_idx != -1 and end_idx != -1:
                                 json_str = line[start_idx:end_idx]
                                 data = json.loads(json_str)
                                 self.log_queue.put({"type": "success", "data": data})
                         except json.JSONDecodeError:
                             pass # Not valid JSON, ignore
            except Exception as e:
                self.log_queue.put({"type": "log", "content": f"Error reading output: {e}"})
                break

        if self.is_running:
             # Process exited unexpectedly or finished
             self.log_queue.put({"type": "finished"})

    def process_logs(self):
        try:
            while True:
                item = self.log_queue.get_nowait()

                if item["type"] == "log":
                    self.log_message(item["content"])

                elif item["type"] == "success":
                    data = item["data"]
                    self.log_message(f"SUCCESS! Wallet Found: {data.get('public_key')}")
                    self.stop_mining()
                    messagebox.showinfo("Wallet Found!", f"Public Key: {data.get('public_key')}\n\nCheck logs for details.")

                elif item["type"] == "finished":
                     if self.is_running:
                         self.stop_mining()
                         self.log_message("Process finished.")

        except queue.Empty:
            pass

        self.root.after(100, self.process_logs)

    def log_message(self, msg):
        self.log_area.config(state="normal")
        self.log_area.insert(tk.END, msg + "\n")
        self.log_area.see(tk.END)
        self.log_area.config(state="disabled")

    def on_close(self):
        if self.is_running:
            self.stop_mining()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VanityForgeApp(root)
    root.mainloop()
