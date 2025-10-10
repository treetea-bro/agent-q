import asyncio
import inspect
import traceback

from playwright.async_api import Page
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from typing_extensions import Annotated

from agentq.core.web_driver.playwright import PlaywrightManager
from agentq.utils.dom_mutation_observer import subscribe, unsubscribe
from agentq.utils.logger import logger

# ============================================================
# ============= Public API: click() ==========================
# ============================================================


async def click(
    selector: Annotated[
        str,
        "The properly formed query selector string to identify the element for the click action (e.g. [mmid='114']). When \"mmid\" attribute is present, use it for the query selector.",
    ],
    wait_before_execution: Annotated[
        float,
        "Optional wait time (sec) before executing the click event logic.",
        float,
    ] = 0.0,
) -> Annotated[str, "A message indicating success or failure of the click."]:
    """
    Executes a headless-safe, SPA-compatible click on the element matching the selector.

    Differences from old version:
    - Reuses the singleton PlaywrightManager (no new browser)
    - Removes expect_navigation() to avoid SPA timeouts
    - Adds robust multi-step click logic (locator → JS → mouse)
    - Works reliably in headless/new & CDP-attached sessions
    """
    logger.info(f'Executing ClickElement with "{selector}"')

    browser_manager = PlaywrightManager()  # ✅ Singleton 재사용 (headless 인자 제거)
    page = await browser_manager.get_current_page()
    if page is None:
        raise ValueError("No active page found. OpenURL command opens a new page.")

    function_name = inspect.currentframe().f_code.co_name
    await browser_manager.take_screenshots(f"{function_name}_start", page)
    await browser_manager.highlight_element(selector, True)

    dom_changes_detected = None

    def detect_dom_changes(changes: str):
        nonlocal dom_changes_detected
        dom_changes_detected = changes

    subscribe(detect_dom_changes)

    try:
        # ✅ Short stabilization wait before click
        if wait_before_execution > 0:
            await asyncio.sleep(wait_before_execution)

        # ✅ Headless-safe robust click
        result_summary, result_detail = await _robust_click(page, selector)

        # ✅ SPA: no hard navigation expected, just wait for DOM settle
        try:
            await page.wait_for_load_state("networkidle", timeout=5000)
        except Exception:
            pass  # fine for SPA pages

        result = {"summary_message": result_summary, "detailed_message": result_detail}

    except PlaywrightTimeoutError as e:
        logger.warning(f"Timeout clicking {selector}: {e}")
        result = {
            "summary_message": "Timeout while trying to click element",
            "detailed_message": f"Element {selector} not found within timeout in headless mode.",
        }
    except Exception as e:
        logger.error(f"Error during click operation: {e}")
        traceback.print_exc()
        result = {
            "summary_message": "Click encountered an error",
            "detailed_message": f"Click failed: {str(e)}",
        }

    await asyncio.sleep(0.1)
    unsubscribe(detect_dom_changes)
    await browser_manager.take_screenshots(f"{function_name}_end", page)

    if dom_changes_detected:
        return (
            f"Success: {result['summary_message']}.\n"
            f"As a consequence of this action, new elements have appeared: {dom_changes_detected}. "
            f"This means the action to click {selector} may need further interaction. Get all_fields DOM to complete it."
        )
    return result["detailed_message"]


# ============================================================
# ============= Core Logic ===================================
# ============================================================


async def _robust_click(
    page: Page, selector: str, timeout_ms: int = 10000
) -> tuple[str, str]:
    """
    Tries multiple strategies for clicking:
    1. Normal locator click (visible → scroll → click)
    2. JS click fallback
    3. Mouse coordinate click fallback
    """
    locator = page.locator(selector)

    # === Frame 탐색 (iframe 대응) ===
    if await locator.count() == 0:
        for f in page.frames:
            loc = f.locator(selector)
            if await loc.count() > 0:
                locator = loc
                break

    if await locator.count() == 0:
        await page.wait_for_selector(selector, state="attached", timeout=timeout_ms)
        locator = page.locator(selector)

    locator = locator.first

    # === Scroll & visible 대기 ===
    await locator.wait_for(state="visible", timeout=timeout_ms)
    await locator.scroll_into_view_if_needed()

    # === Step 1: Standard Click ===
    try:
        await locator.click(timeout=timeout_ms)
        return (
            "Click executed successfully",
            f"Clicked {selector} via Playwright locator.",
        )
    except Exception as first_err:
        logger.warning(f"Standard click failed for {selector}: {first_err}")

        # === Step 2: JS Click Fallback ===
        try:
            await locator.evaluate("el => el.click()")
            return (
                "Click executed via JavaScript",
                f"JS-based click succeeded for {selector} after standard click failed.",
            )
        except Exception as second_err:
            logger.warning(f"JS click failed for {selector}: {second_err}")

            # === Step 3: Mouse Fallback ===
            try:
                box = await locator.bounding_box()
                if box:
                    await page.mouse.move(
                        box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
                    )
                    await page.mouse.click(
                        box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
                    )
                    return (
                        "Click executed via mouse coordinates",
                        f"Mouse-based click worked for {selector}.",
                    )
            except Exception as third_err:
                raise PlaywrightTimeoutError(
                    f"All click strategies failed for {selector}. "
                    f"Std: {first_err} | JS: {second_err} | Mouse: {third_err}"
                )


# ============================================================
# ============= Optional helpers (preserved for compat) ======
# ============================================================


async def is_element_present(page: Page, selector: str) -> bool:
    element = await page.query_selector(selector)
    return element is not None


async def perform_javascript_click(page: Page, selector: str):
    """
    Backward-compatible JS click helper (kept for legacy logic that still calls it).
    Used inside do_click() if needed.
    """
    js_code = """(selector) => {
        const element = document.querySelector(selector);
        if (!element) return `Element ${selector} not found`;
        if (element.tagName.toLowerCase() === 'a') {
            element.target = '_self';
            element.removeAttribute('target');
            element.removeAttribute('rel');
        }
        element.click();
        return `Executed JS click on ${selector}`;
    }"""
    try:
        logger.info(f"Executing JS click on {selector}")
        result: str = await page.evaluate(js_code, selector)
        return result
    except Exception as e:
        logger.error(f"JS click failed for {selector}: {e}")
        traceback.print_exc()
        return f"Error executing JS click: {e}"


# ============================================================
# ============= Legacy Compatibility Wrapper ================
# ============================================================


async def do_click(page: Page, selector: str, wait_before_execution: float):
    """
    Legacy compatibility shim for older imports expecting do_click().
    Internally uses the new _robust_click() logic.
    """
    try:
        result_summary, result_detail = await _robust_click(page, selector)
        return {
            "summary_message": result_summary,
            "detailed_message": result_detail,
        }
    except Exception as e:
        return {
            "summary_message": f"do_click() failed for {selector}",
            "detailed_message": f"Error: {str(e)}",
        }
