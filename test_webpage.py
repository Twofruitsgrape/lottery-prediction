from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={'width': 1280, 'height': 900})
    
    # Navigate to the local server
    page.goto('http://localhost:8000')
    page.wait_for_load_state('networkidle')
    
    # Take a screenshot
    page.screenshot(path='screenshot.png', full_page=True)
    
    print("Screenshot saved to screenshot.png")
    
    # Check page title
    title = page.title()
    print(f"Page title: {title}")
    
    # Check if prediction data loaded
    issue_element = page.locator('#next-issue')
    issue_text = issue_element.text_content()
    print(f"Next issue: {issue_text}")
    
    # Check red balls
    red_balls = page.locator('.red-ball').all()
    print(f"Red balls count: {len(red_balls)}")
    
    # Check blue ball
    blue_ball = page.locator('.blue-ball').first
    blue_text = blue_ball.text_content()
    print(f"Blue ball: {blue_text}")
    
    browser.close()
