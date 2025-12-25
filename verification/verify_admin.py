from playwright.sync_api import sync_playwright, expect
import time

def test_admin_dashboard():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        # 1. Navigate to home
        page.goto("http://localhost:8080")

        # 2. Simulate Admin Login by setting the cookie (simulating backend response)
        # Note: In real flow, user POSTs to /admin-login. We can try that too.

        response = page.request.post("http://localhost:8080/admin-login", data={"password": "admin123"})
        print(f"Login Status: {response.status}")

        # Reload to trigger checkAdmin()
        page.reload()

        # 3. Check for DEV tab
        dev_btn = page.locator("#nav-dev")
        expect(dev_btn).to_be_visible()

        # 4. Click DEV tab
        dev_btn.click()

        # 5. Check Dashboard Visibility
        dashboard = page.locator("#admin-dashboard")
        expect(dashboard).to_be_visible()

        # 6. Check Stats (Wait for fetch)
        # The fetch might fail because no Firestore, but structure should be there.
        # Check specific element
        expect(page.locator("text=God Mode Dashboard")).to_be_visible()

        # Screenshot
        page.screenshot(path="verification/admin_dashboard.png")
        print("Screenshot saved to verification/admin_dashboard.png")

        browser.close()

if __name__ == "__main__":
    test_admin_dashboard()
