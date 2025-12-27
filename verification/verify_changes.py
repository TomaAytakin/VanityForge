from playwright.sync_api import sync_playwright

def verify_changes():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # 1. Verify Home Page (index.html)
        print("Verifying Index Page...")
        page.goto("http://localhost:8080/")
        page.wait_for_selector("nav")

        # Capture Header
        page.screenshot(path="verification/index_header.png", clip={"x":0, "y":0, "width": 1280, "height": 300})

        # Verify Heartbeat (Green Dot)
        # Look for the span with animate-ping
        try:
            page.wait_for_selector(".animate-ping", timeout=2000)
            print("Heartbeat element found.")
        except:
            print("Heartbeat element NOT found!")

        # 2. Verify Roadmap
        print("Verifying Roadmap Page...")
        page.goto("http://localhost:8080/roadmap")
        page.screenshot(path="verification/roadmap_header.png", clip={"x":0, "y":0, "width": 1280, "height": 200})

        # 3. Verify FAQ
        print("Verifying FAQ Page...")
        page.goto("http://localhost:8080/faq")
        page.screenshot(path="verification/faq_header.png", clip={"x":0, "y":0, "width": 1280, "height": 200})

        # 4. Verify VVVIP
        print("Verifying VVVIP Page...")
        page.goto("http://localhost:8080/vvvip")
        page.screenshot(path="verification/vvvip_header.png", clip={"x":0, "y":0, "width": 1280, "height": 200})

        browser.close()

if __name__ == "__main__":
    verify_changes()
