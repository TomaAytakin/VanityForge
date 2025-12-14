from playwright.sync_api import sync_playwright
import os
import threading
import http.server
import socketserver

PORT = 8081

def start_server():
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print("serving at port", PORT)
        httpd.serve_forever()

def test_wallet_connection():
    # Start server
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Mock Phantom Wallet
        page.add_init_script("""
            window.solana = {
                isPhantom: true,
                isConnected: false,
                connect: async (options) => {
                    console.log('Connect called with:', options);
                    // Verify correct options passed
                    if (options && options.onlyIfTrusted === false) {
                        window.solana.isConnected = true;
                        const key = '58PwtjSDuFHuUkYjH9BYnnQKHfwo9reZhC2zMJv9JPkx'; // dummy key
                        return { publicKey: { toString: () => key } };
                    }
                    if (options && options.onlyIfTrusted === true) {
                        // Simulate rejection for auto-connect if any
                        throw new Error('User rejected the request.');
                    }
                    // Default connect without options (should not happen if code is correct)
                     window.solana.isConnected = true;
                     const key = '58PwtjSDuFHuUkYjH9BYnnQKHfwo9reZhC2zMJv9JPkx';
                     return { publicKey: { toString: () => key } };
                },
                disconnect: async () => {
                    console.log('Disconnect called');
                    window.solana.isConnected = false;
                },
                on: (event, callback) => {
                    if (!window.solana.listeners) window.solana.listeners = {};
                    window.solana.listeners[event] = callback;
                },
                listeners: {}
            };
        """)

        page.goto(f'http://localhost:{PORT}/vvvip.html')

        # Wait for page load
        page.wait_for_load_state('networkidle')

        # Click connect button
        page.click('#connect-phantom-btn')

        # Wait a bit for async actions
        page.wait_for_timeout(2000)

        # Take screenshot
        page.screenshot(path='verification/wallet_connect.png')

        # Verify text
        btn_text = page.inner_text('#wallet-btn-text')
        print(f'Button text: {btn_text}')

        if "58Pw...JPkx" in btn_text:
            print("SUCCESS: Wallet connected and text updated.")
        else:
            print("FAILURE: Wallet text not updated correcty.")

        browser.close()

if __name__ == '__main__':
    test_wallet_connection()
