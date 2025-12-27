from playwright.sync_api import sync_playwright

def verify_stats(page):
    # Go to the local homepage
    page.goto("http://127.0.0.1:8080/")

    # Wait for the stats element to be visible
    stats_locator = page.locator("#identity-counter")
    stats_locator.wait_for(state="visible", timeout=10000)

    # The stats text might start as "..." or "0" and then update.
    # We want to wait until it's a number > 0.
    # The odometer creates complex DOM, so we check innerText generally.

    # Wait for non-zero content
    page.wait_for_function("""
        () => {
            const el = document.getElementById('identity-counter');
            return el && el.innerText.trim() !== '0' && el.innerText.trim() !== '...';
        }
    """, timeout=10000)

    # Take a screenshot
    page.screenshot(path="/home/jules/verification/stats_verification.png")
    print("Screenshot taken at /home/jules/verification/stats_verification.png")

if __name__ == "__main__":
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            verify_stats(page)
        except Exception as e:
            print(f"Verification failed: {e}")
        finally:
            browser.close()
