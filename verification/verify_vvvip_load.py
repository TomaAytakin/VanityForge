from playwright.sync_api import sync_playwright, expect
import os

def run(playwright):
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()

    # Go to VVVIP page (hosted by local server)
    try:
        page.goto("http://127.0.0.1:8080/vvvip")

        # Verify title
        expect(page).to_have_title("VVVIP Addresses - VanityForge")

        # Verify header
        header = page.get_by_role("heading", name="VVVIP Addresses")
        expect(header).to_be_visible()

        # Verify "Claim Your .sol Username" section is visible
        claim_section = page.get_by_text("Claim Your .sol Username")
        expect(claim_section).to_be_visible()

        # Check for SNS Utils loading errors by checking if window.SNSUtils exists
        # This confirms sns.js loaded and executed its top level code
        result = page.evaluate("typeof window.SNSUtils")
        if result != "object":
            print(f"Error: window.SNSUtils is {result}, expected object")
            exit(1)
        else:
            print("window.SNSUtils loaded successfully")

        # Take screenshot
        os.makedirs("/home/jules/verification", exist_ok=True)
        page.screenshot(path="/home/jules/verification/vvvip_loaded.png")
        print("Verification successful, screenshot saved.")

    except Exception as e:
        print(f"Verification failed: {e}")
        # Capture screenshot on failure if possible
        try:
            page.screenshot(path="/home/jules/verification/vvvip_failed.png")
        except:
            pass
        exit(1)
    finally:
        browser.close()

with sync_playwright() as playwright:
    run(playwright)
