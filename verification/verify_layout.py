from playwright.sync_api import sync_playwright
import os

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        # Load the local index.html file
        page.goto(f"file://{os.getcwd()}/index.html")

        # Unhide the job queue section and app interface to verify layout
        page.evaluate("""
            document.getElementById('job-queue-section').classList.remove('hidden');
        """)

        # Take a screenshot of the entire page to see the layout
        if not os.path.exists("verification"):
            os.makedirs("verification")
        page.screenshot(path="verification/layout_snapshot.png", full_page=True)
        print("Screenshot saved to verification/layout_snapshot.png")
        browser.close()

if __name__ == "__main__":
    run()