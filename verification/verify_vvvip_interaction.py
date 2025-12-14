from playwright.sync_api import sync_playwright
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler

PORT = 8002

def start_server():
    httpd = HTTPServer(('localhost', PORT), SimpleHTTPRequestHandler)
    httpd.serve_forever()

def verify_frontend():
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    time.sleep(1)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        url = f"http://localhost:{PORT}/vvvip.html"
        print(f"Navigating to {url}")
        page.goto(url)

        # Wait for module to load
        page.wait_for_function("() => typeof window.SNSUtils !== 'undefined'")

        # Check Buffer
        is_buffer = page.evaluate("typeof window.Buffer !== 'undefined'")
        print(f"Buffer defined: {is_buffer}")

        # Simulate interaction
        # Fill input
        page.fill('#sns-domain-input', 'testdomain')

        # Click search
        page.click('#sns-check-btn')

        # Wait for result (it might fail due to backend proxy not running, but UI should update to spinner or error)
        # The code does: btn.innerHTML = spinner, then fetch.
        # If fetch fails, it shows alert or error in console.
        # We can check if spinner appeared or if console has the fetch error (which proves the function ran).

        # Take screenshot of the "Search" state or error
        time.sleep(2) # Wait for potential UI update

        screenshot_path = "verification/vvvip_interaction.png"
        page.screenshot(path=screenshot_path)
        print(f"Screenshot saved to {screenshot_path}")

        browser.close()

if __name__ == "__main__":
    verify_frontend()
