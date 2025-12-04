from playwright.sync_api import sync_playwright
import os

def run(playwright):
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()

    # Load the local index.html
    cwd = os.getcwd()
    page.goto(f"file://{cwd}/index.html")

    # Check if jobListenerUnsubscribe is defined in global scope
    # We can evaluate JS
    is_defined = page.evaluate("typeof jobListenerUnsubscribe !== 'undefined'")
    print(f"jobListenerUnsubscribe defined: {is_defined}")

    initial_value = page.evaluate("jobListenerUnsubscribe")
    print(f"jobListenerUnsubscribe initial value: {initial_value}")

    # Check if pollJobStatus is using the new logic (conceptually)
    # We can get the string representation of the function
    func_str = page.evaluate("pollJobStatus.toString()")
    if "jobListenerUnsubscribe" in func_str and "onSnapshot" in func_str:
        print("pollJobStatus contains expected keywords.")
    else:
        print("pollJobStatus might not be updated correctly.")
        print(func_str)

    page.screenshot(path="verification/index_loaded.png")

    browser.close()

with sync_playwright() as playwright:
    run(playwright)
