"""Step 14: full-resolution screenshots of the running traffic app, used in the report and the slides.

Start the app first (from the project folder):   streamlit run app/app.py --server.port 8612
Needs Playwright (see requirements-report.txt).

Writes into figures/:  app2_live.png (tabs to animation), app2_slide_live.png (animation only), app2_inspector.png,
app2_compare.png and app2_pareto.png (each from the tabs down).

Usage:  python -I experiments/14_app_screenshots.py [url]
"""
import site
import sys
from pathlib import Path

sys.path.append(site.getusersitepackages())
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8612"
VIEW = {"width": 1360, "height": 1700}
X0, WIDTH = 316, 1030                                     # the page without the sidebar


def box(page, selector, index=0):
    return page.locator(selector + ":visible").nth(index).bounding_box()          # only what is on screen: the other tabs are hidden


def card_top(page, selector):
    """Top edge of the bordered card that holds the first visible match: a crop that ends above it shows no part of that card."""
    return page.locator(selector + ":visible").first.evaluate(
        "el => { for (let n = el; n && n !== document.body; n = n.parentElement) { if (parseFloat(getComputedStyle(n).borderTopWidth) > 0) return n.getBoundingClientRect().top; } return null; }")


def shoot(page, name, top, bottom, pad=10):
    page.screenshot(path=str(OUT / name), clip={"x": X0, "y": max(0, top - pad), "width": WIDTH, "height": bottom - top + 2 * pad})
    print("saved", name)


with sync_playwright() as p:
    try:
        browser = p.chromium.launch(channel="chrome")
    except Exception:
        browser = p.chromium.launch()
    page = browser.new_page(viewport=VIEW, device_scale_factor=2, color_scheme="light")
    page.goto(URL)
    page.wait_for_selector("text=mean delay per car", timeout=180000)
    page.wait_for_timeout(7000)
    frame = page.frame_locator("iframe").first
    frame.locator("#scrub").evaluate("el => { el.value = 400; el.dispatchEvent(new Event('input')); }")
    frame.locator("#play").click()                                      # pause the animation at t = 400 s
    page.wait_for_timeout(900)
    tabs, anim = box(page, '[role="tablist"]'), box(page, "iframe")
    shoot(page, "app2_live.png", tabs["y"], anim["y"] + anim["height"])
    page.screenshot(path=str(OUT / "app2_slide_live.png"), clip={"x": anim["x"], "y": anim["y"], "width": anim["width"], "height": anim["height"]})
    print("saved app2_slide_live.png")

    page.get_by_role("tab", name="Compare controllers").click()
    page.wait_for_timeout(6000)
    last = box(page, '[data-testid="stPlotlyChart"]')
    shoot(page, "app2_compare.png", tabs["y"], last["y"] + last["height"])

    page.get_by_role("tab", name="Fuzzy inspector").click()
    page.wait_for_timeout(4500)
    shoot(page, "app2_inspector.png", tabs["y"], card_top(page, '[data-testid="stPlotlyChart"]') - 18)    # tabs, the two stages and their rules, down to the row of curve cards

    page.get_by_role("tab", name="Pareto lab").click()
    page.wait_for_timeout(4500)
    chart = box(page, '[data-testid="stPlotlyChart"]')
    shoot(page, "app2_pareto.png", tabs["y"], chart["y"] + chart["height"] + 24)
    browser.close()
