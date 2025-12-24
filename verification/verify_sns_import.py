from playwright.sync_api import sync_playwright, expect
import time

def run(playwright):
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()

    errors = []
    page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
    page.on("pageerror", lambda exc: errors.append(str(exc)))

    # Go to localhost where server is running
    page.goto("http://localhost:8000/index.html")

    # Wait a bit for scripts to load
    page.wait_for_timeout(3000)

    # Check for specific error
    import_error = any("Cannot use import statement outside a module" in e for e in errors)

    print(f"Console Errors Found: {len(errors)}")
    for e in errors:
        print(f" - {e}")

    if import_error:
        print("FAILURE: Found 'Cannot use import statement outside a module' error.")
        exit(1)
    else:
        print("SUCCESS: No module import error found.")

    page.screenshot(path="verification/index_load.png")
    browser.close()

if __name__ == "__main__":
    with sync_playwright() as playwright:
        run(playwright)
