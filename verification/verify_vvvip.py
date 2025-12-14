from playwright.sync_api import sync_playwright

def verify_vvvip_page():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Mock the /api/check-sns response to ensure we get an "Available" result
        page.route("**/api/check-sns", lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body='{"result": {"value": null}}' # null value means account not found = Available
        ))

        # Navigate to the new page
        print("Navigating to http://localhost:8080/vvvip")
        page.goto("http://localhost:8080/vvvip")

        # Check Title
        title = page.title()
        assert "VVVIP Addresses" in title
        print(f"Title Verified: {title}")

        # Check Header
        header = page.inner_text("h1")
        assert "VVVIP Addresses" in header
        print("Header Verified")

        # Check Nav Link
        nav_link = page.query_selector('a[href="/vvvip"]')
        assert nav_link is not None
        print("Nav Link Verified on VVVIP page")

        # Check SNS Input
        sns_input = page.query_selector('#sns-domain-input')
        assert sns_input is not None
        print("SNS Input Field Verified")

        # Perform Interaction: Check Domain
        print("Typing domain...")
        sns_input.fill("myuniquedomain12345")

        print("Clicking Check...")
        page.click("#sns-check-btn")

        # Wait for Result
        print("Waiting for result...")
        page.wait_for_selector("#sns-result:not(.hidden)", state="visible", timeout=5000)

        # Verify "Buy Now" button appears
        buy_btn = page.query_selector('button[onclick^="buyDomain"]')
        if buy_btn:
            print("✅ Buy Now Button Found!")
            text = buy_btn.inner_text()
            print(f"Button Text: {text}")
            assert "Buy Now" in text
        else:
            print("❌ Buy Now Button NOT Found!")
            raise Exception("Buy Now button did not appear")

        # Screenshot
        page.screenshot(path="verification/vvvip_interaction.png")
        print("Screenshot saved to verification/vvvip_interaction.png")

        browser.close()

if __name__ == "__main__":
    try:
        verify_vvvip_page()
        print("✅ Verification Successful")
    except Exception as e:
        print(f"❌ Verification Failed: {e}")
        exit(1)
