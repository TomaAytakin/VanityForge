
from playwright.sync_api import sync_playwright
import os

def run():
    # Get the absolute path to index.html
    cwd = os.getcwd()
    file_url = f'file://{cwd}/index.html'

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Load the page
        print(f'Loading {file_url}')
        page.goto(file_url)

        # Verify Referrals Link exists
        print('Checking Navigation Link...')
        ref_link = page.locator('#nav-referrals')
        if ref_link.is_visible():
            print('Referrals link found.')
        else:
            print('Referrals link NOT found.')

        # Take Screenshot of Forge Page with new Input
        page.screenshot(path='verification/forge_page.png')
        print('Screenshot forge_page.png saved.')

        # Click Referrals Link to open Dashboard
        # Note: Since there is no backend, the API call will fail.
        # But we can check if the section toggles visibility.
        print('Clicking Referrals Link...')
        ref_link.click()

        # Wait for potential animation/toggle
        page.wait_for_timeout(1000)

        # Take Screenshot of Referral Dashboard (it might show loading or error state)
        page.screenshot(path='verification/referral_page.png')
        print('Screenshot referral_page.png saved.')

        browser.close()

if __name__ == '__main__':
    run()
