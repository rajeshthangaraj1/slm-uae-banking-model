from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

def get_scraping_data(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # Keep it False for debugging
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/115.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800}
        )

        page = context.new_page()

        try:
            print(f"🌐 Navigating to: {url}")
            response = page.goto(url, timeout=60000)
            if not response or not response.ok:
                print("❌ Failed to load page or received error response.")
                return None

            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(5500)

            # # Screenshot safely
            # try:
            #     page.screenshot(path="debug_page.png", full_page=True)
            #     print("📸 Screenshot saved.")
            # except Exception as e:
            #     print("⚠️ Screenshot error:", e)

            # Extract HTML content safely
            try:
                if page.is_closed():
                    print("⚠️ Page is closed, skipping content extraction.")
                    return None
                html = page.content()
                print("✅ Page content extracted.")

                if "block-rulebook-content" not in html:
                    print("⚠️ 'block-rulebook-content' not found in HTML.")
                else:
                    print("✅ Found 'block-rulebook-content'.")

                locator = page.locator("#block-rulebook-content")
                locator.wait_for(timeout=10000)
                section_html = locator.inner_text()

                with open("scraped_section.txt", "w", encoding="utf-8") as f:
                    # f.write("<div id='block-rulebook-content'>\n")
                    f.write(section_html)
                    # f.write("\n</div>")

                print("✅ HTML saved to 'scraped_section.html'")
                return section_html

            except Exception as e:
                print("❌ Error extracting content:", e)
                return None

        finally:
            try:
                context.close()
                browser.close()
            except Exception as e:
                print("⚠️ Cleanup error:", e)

def main():
    url = "https://rulebook.centralbank.ae/en/entiresection/40"
    result = get_scraping_data(url)
    print("🧾 Final Result Length:", len(result) if result else "No content extracted")
    return {"result": result}

if __name__ == "__main__":
    main()
