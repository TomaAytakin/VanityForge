from playwright.sync_api import sync_playwright

def verify_frontend():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to the locally running Flask server
        page.goto("http://127.0.0.1:8080")

        # 1. Verify Banner is visible always
        banner = page.get_by_text("Now Powered Exclusively by NVIDIA L4")
        if banner.is_visible():
            print("SUCCESS: NVIDIA L4 Banner is visible.")
        else:
            print("FAILURE: NVIDIA L4 Banner not found.")

        # 2. Simulate Login to reveal #app-interface
        # Since we are mocking, we can just force the display of #app-interface
        # or we can mock the firebase login flow? No, simpler to just inject JS to show it.
        # But wait, the banner should be visible even if not logged in?
        # The banner is inside `.w-full.max-w-2xl.mb-8.flex.flex-col...` which is outside #auth-section and #app-interface.
        # So Banner verification passing is good.

        # 3. Reveal app interface
        page.evaluate("document.getElementById('auth-section').classList.add('hidden')")
        page.evaluate("document.getElementById('app-interface').classList.remove('hidden')")

        # 4. Verify Estimates
        # Input 'A' into prefix to trigger updateEstimates
        # Note: #prefix is inside #forge-section which is inside #app-interface
        # #forge-section is visible by default when #app-interface is visible (based on showSection logic, forge is default?)
        # Let's check if #forge-section is hidden?
        # In HTML: <div id="forge-section"> ... it doesn't have 'hidden' class initially.
        # But #app-interface has 'hidden'.

        page.fill("#prefix", "A")

        # Calculate expected:
        # Len = 1. Difficulty = 0.5 * 58^1 = 29.
        # Speed = 300,000,000.
        # Time = (29 / 300000000) + 600 = ~600 seconds.
        # 600s / 60 = 10m.
        # Expected text: "~10m"

        # Wait for update
        page.wait_for_timeout(1000)

        time_display = page.locator("#calc-time")
        time_text = time_display.inner_text()
        print(f"Time Display for 1 char: {time_text}")

        if "~10m" in time_text:
            print("SUCCESS: Time estimate is ~10m (includes buffer).")
        else:
             print(f"FAILURE: Unexpected time estimate: {time_text}")

        # 5. Verify no visible GPU checkbox
        # We checked earlier it should be checked and disabled, but parent container might be hidden?
        # In HTML: <div class="hidden flex items-center gap-2 p-2 bg-purple-900/20 rounded border border-purple-500/30">
        # So the user doesn't see it.

        gpu_checkbox = page.locator("#use-gpu")
        # Since parent is hidden, is_visible() should be false
        if not gpu_checkbox.is_visible():
             print("SUCCESS: GPU checkbox is hidden from user.")
        else:
             print("FAILURE: GPU checkbox IS visible.")

        # Take screenshot
        page.screenshot(path="verification/frontend_verify.png", full_page=True)
        browser.close()

if __name__ == "__main__":
    verify_frontend()
