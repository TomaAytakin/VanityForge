
from playwright.sync_api import sync_playwright, expect

def test_landing_page():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to localhost
        try:
            page.goto("http://localhost:8080")
            print("Navigated to localhost:8080")

            # Wait for key elements
            # 1. Check Headline
            expect(page.get_by_role("heading", name="Forge Your Custom Identity on Solana.")).to_be_visible()
            print("Headline found")

            # 2. Check "Why Forge?" Selling Points (e.g. "2 Free Wallet Generations")
            expect(page.get_by_text("2 Free Wallet Generations")).to_be_visible()
            print("Selling point found")

            # 3. Check Login Button
            expect(page.get_by_role("button", name="Login with Google")).to_be_visible()
            print("Login button found")

            # Take Landing Page Screenshot
            page.screenshot(path="verification/landing_page.png")
            print("Screenshot landing_page.png saved")

            # 4. Simulate Login Success to verify toggle
            # We must set currentUser first to avoid null error in onLoginSuccess
            page.evaluate("window.currentUser = 'test_user_123';")
            page.evaluate("window.onLoginSuccess()")

            # Wait for transition
            page.wait_for_timeout(1000)

            # Check that landing container is hidden
            landing_container = page.locator("#landing-container")
            expect(landing_container).not_to_be_visible() # or have class 'hidden'

            # Check that app interface is visible
            app_interface = page.locator("#app-interface")
            expect(app_interface).to_be_visible()
            print("Login simulation successful - Landing hidden, App shown")

            # Take Logged In Screenshot
            page.screenshot(path="verification/logged_in_view.png")
            print("Screenshot logged_in_view.png saved")

        except Exception as e:
            print(f"Error: {e}")
            page.screenshot(path="verification/error.png")
        finally:
            browser.close()

if __name__ == "__main__":
    test_landing_page()
