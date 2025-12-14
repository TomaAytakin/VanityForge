from playwright.sync_api import sync_playwright
import os
import sys
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler

PORT = 8001

def start_server():
    httpd = HTTPServer(('localhost', PORT), SimpleHTTPRequestHandler)
    httpd.serve_forever()

def verify_sdk_load():
    # Start server in background thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    # Give it a moment to start
    time.sleep(1)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        console_errors = []
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda exc: console_errors.append(str(exc)))

        url = f"http://localhost:{PORT}/vvvip.html"
        print(f"Loading {url}...")

        try:
            page.goto(url)
            # Wait for modules to load.
            # We can wait for window.SNSUtils to be defined.
            try:
                page.wait_for_function("() => typeof window.SNSUtils !== 'undefined'", timeout=5000)
                sns_utils_defined = True
            except:
                sns_utils_defined = False

            buffer_defined = page.evaluate("typeof window.Buffer !== 'undefined'")
            print(f"window.Buffer defined: {buffer_defined}")
            print(f"window.SNSUtils defined: {sns_utils_defined}")

            print("Console Errors:")
            found_sdk_error = False
            for err in console_errors:
                print(f" - {err}")
                if "Bonfida SDK not loaded" in err or "Buffer is not defined" in err:
                    found_sdk_error = True

            if buffer_defined and sns_utils_defined and not found_sdk_error:
                print("SUCCESS: SDK loaded and Buffer defined.")
            else:
                print("FAILURE: Issues detected.")

        except Exception as e:
            print(f"Exception during verification: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    verify_sdk_load()
