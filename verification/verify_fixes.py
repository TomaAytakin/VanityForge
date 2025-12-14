from playwright.sync_api import sync_playwright, expect
import time

def verify_fixes():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        # Mock window.solana
        page.add_init_script("""
            window.solana = {
                isPhantom: true,
                isConnected: false,
                connect: async (opts) => {
                    window.solana.isConnected = true;
                    return { publicKey: { toString: () => "MockPublicKey" } };
                },
                disconnect: async () => { window.solana.isConnected = false; },
                signAndSendTransaction: async () => { return { signature: "sig" }; }
            };
        """)

        # Mock fetch for /check-user
        page.route("**/check-user", lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body='{"has_pin": false, "trials_used": 0}'
        ))

        # Navigate to index.html
        print("Navigating to index.html...")
        page.goto("http://localhost:8081/index.html")

        # Wait for potential auto-connect logic
        time.sleep(2)

        # Check if Numpad is hidden (Premature PIN fix)
        # The modal has id="numpad-modal" and classes "hidden" initially.
        # If openNumpad is called, "hidden" is removed.
        numpad = page.locator("#numpad-modal")
        print("Checking Numpad visibility...")
        if numpad.is_visible():
            print("FAILURE: Numpad is visible! Premature PIN fix failed.")
        else:
            print("SUCCESS: Numpad is hidden. Premature PIN fix works.")

        # Take screenshot
        page.screenshot(path="verification/index_verified.png")

        # Verify window exports in vvvip.html
        print("Navigating to vvvip.html...")
        page.goto("http://localhost:8081/vvvip.html")

        # Check window.buyDomain
        is_buy_domain = page.evaluate("typeof window.buyDomain === 'function'")
        if is_buy_domain:
             print("SUCCESS: window.buyDomain is a function.")
        else:
             print("FAILURE: window.buyDomain is NOT a function.")

        # Check window.Buffer
        # We need to wait a bit for scripts to load if they are external
        page.wait_for_timeout(1000)
        is_buffer = page.evaluate("!!window.Buffer")
        if is_buffer:
             print("SUCCESS: window.Buffer is defined.")
        else:
             print("FAILURE: window.Buffer is NOT defined.")

        browser.close()

if __name__ == "__main__":
    verify_fixes()
