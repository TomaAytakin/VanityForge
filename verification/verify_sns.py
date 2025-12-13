from playwright.sync_api import sync_playwright, expect
import time

def run(playwright):
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()

    page.on("console", lambda msg: print(f"CONSOLE: {msg.text}"))
    page.on("pageerror", lambda err: print(f"PAGE ERROR: {err}"))

    print("Navigating to home...")
    page.goto("http://localhost:8080")

    # Bypass login by exposing the interface manually via script
    print("Bypassing login...")
    page.evaluate("document.getElementById('auth-section').classList.add('hidden')")
    page.evaluate("document.getElementById('app-interface').classList.remove('hidden')")

    # Wait for animation
    page.wait_for_timeout(1000)

    # 1. Check "bonfida" (Taken)
    print("Checking 'bonfida'...")
    page.fill("#sns-domain-input", "bonfida")
    page.click("#sns-check-btn")

    # Wait for result
    try:
        page.wait_for_selector("#sns-result:not(.hidden)", state="visible", timeout=10000)
    except Exception as e:
        print(f"Timeout checking bonfida: {e}")
        page.screenshot(path="verification/timeout_debug.png")
        browser.close()
        return

    # Verify "Taken" text
    content = page.inner_text("#sns-result")
    print(f"Result for bonfida: {content}")
    if "Taken" not in content:
        print("FAILED: bonfida should be taken")
    else:
        print("PASSED: bonfida is taken")

    # 2. Check random available
    random_name = f"jules-test-{int(time.time())}"
    print(f"Checking '{random_name}'...")
    page.fill("#sns-domain-input", random_name)
    page.click("#sns-check-btn")

    # Wait for result to update (text change)
    page.wait_for_timeout(2000)

    content = page.inner_text("#sns-result")
    print(f"Result for {random_name}: {content}")
    if "Available" not in content:
        print(f"FAILED: {random_name} should be available")
    else:
        print(f"PASSED: {random_name} is available")

    # Screenshot
    print("Taking screenshot...")
    page.screenshot(path="verification/sns_verification.png")

    browser.close()

with sync_playwright() as playwright:
    run(playwright)
