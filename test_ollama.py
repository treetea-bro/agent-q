import asyncio
import io

import ollama
from icecream import ic
from PIL import Image

from agentq.core.skills.get_dom_with_content_type import get_dom_with_content_type
from agentq.core.web_driver.playwright import PlaywrightManager
from models import ClickVideoParams, FilterParams, SearchParams
from tools import TOOLS

playwright = PlaywrightManager()


async def wait_for_navigation(max_retries=3):
    try:
        for attempt in range(max_retries):
            playwright_manager = PlaywrightManager()
            page = await playwright_manager.get_current_page()
            await page.wait_for_load_state("domcontentloaded", timeout=30000)
            print(f"[DEBUG] Navigation successful on attempt {attempt + 1}")
            return
    except Exception as e:
        print(f"[DEBUG] Navigation error on attempt {attempt + 1}: {str(e)}")
    print(f"[DEBUG] Navigation failed after {max_retries} attempts")


async def get_current_dom() -> str:
    await wait_for_navigation()
    dom = await get_dom_with_content_type(content_type="all_fields")
    return "\n\nCurrent DOM: " + str(dom)


async def get_current_screenshot() -> bytes:
    await wait_for_navigation()
    page = await playwright.get_current_page()
    screenshot_bytes = await page.screenshot(full_page=True)

    # Resize to 896x896 using PIL
    img = Image.open(io.BytesIO(screenshot_bytes))
    img = img.resize((896, 896), Image.LANCZOS)

    # Convert back to bytes (JPEG format)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    return img_byte_arr.getvalue()


# ==============================
# 🧰 YouTube Functions
# ==============================
async def search(params: SearchParams):
    print(f"🔍 Searching: {params.query}")
    page = await playwright.get_current_page()
    await page.wait_for_selector('input[name="search_query"]')
    await page.fill('input[name="search_query"]', params.query)
    await page.press('input[name="search_query"]', "Enter")
    await page.wait_for_selector("ytd-item-section-renderer", timeout=10000)


async def apply_youtube_filters(params: FilterParams, timeout: int = 10000):
    page = await playwright.get_current_page()
    await page.wait_for_selector("#filter-button button", timeout=timeout)
    await page.click("#filter-button button")
    await page.wait_for_selector("ytd-search-filter-group-renderer", timeout=timeout)

    for idx, f in enumerate(params.filters):
        filter_groups = await page.query_selector_all(
            "ytd-search-filter-group-renderer"
        )
        for group in filter_groups:
            name_el = await group.query_selector(
                "#filter-group-name yt-formatted-string"
            )
            name = (await name_el.inner_text()).strip() if name_el else ""
            if name.lower() == f.group_name.lower():
                options = await group.query_selector_all("ytd-search-filter-renderer")
                for opt in options:
                    label_el = await opt.query_selector("#label yt-formatted-string")
                    label = (await label_el.inner_text()).strip() if label_el else ""
                    if label.lower() == f.option_label.lower():
                        link = await opt.query_selector("a#endpoint")
                        if link:
                            href = await link.get_attribute("href")
                            print(
                                f"✅ Applying filter: {f.group_name} → {f.option_label}"
                            )
                            await page.goto(f"https://www.youtube.com{href}")
                            await page.wait_for_selector(
                                "ytd-item-section-renderer", timeout=timeout
                            )
                            if idx < len(params.filters) - 1:
                                await page.wait_for_selector(
                                    "#filter-button button", timeout=timeout
                                )
                                await page.click("#filter-button button")
                                await page.wait_for_selector(
                                    "ytd-search-filter-group-renderer", timeout=timeout
                                )
                            break
                break


async def click_video_by_title(params: ClickVideoParams, timeout: int = 10000):
    page = await playwright.get_current_page()
    await page.wait_for_selector("ytd-rich-item-renderer", timeout=timeout)
    items = await page.query_selector_all("ytd-rich-item-renderer")
    for item in items:
        title_span = await item.query_selector("h3 a span")
        if not title_span:
            continue
        text = (await title_span.inner_text()).strip()
        if text == params.title:
            link = await item.query_selector("h3 a")
            if link:
                await link.click()
                print(f"🎬 Clicked video: {params.title}")
                return True
    print(f"❌ No matching video found: {params.title}")
    return False


async def detect_youtube_language() -> str:
    """Detect whether YouTube is in Korean or English."""
    page = await playwright.get_current_page()
    html_lang = await page.get_attribute("html", "lang")
    if html_lang and html_lang.startswith("ko"):
        print("🌐 YouTube language detected: Korean")
        return "ko"
    print("🌐 YouTube language detected: English")
    return "en"


def get_system_prompt(language: str) -> str:
    """Return system prompt text based on detected language."""
    if language == "ko":
        return """
사용 가능한 필터:
- 업로드 날짜: 지난 1시간, 오늘, 이번 주, 이번 달, 올해
- 구분: 동영상, 채널, 재생목록, 영화
- 길이: 4분 미만, 4~20분, 20분 초과
- 기능별: 라이브, 4K, HD, 자막, 크리에이티브 커먼즈, 360°, VR180, 3D, HDR
- 위치: 구입한 항목
- 정렬기준: 관련성, 업로드 날짜, 조회수, 평점

너는 YouTube 자동화 에이전트야.
현재 페이지의 DOM을 어떤 도구를 호출해야 할지 판단해.
단, **한 번에 하나의 tool만 호출**해야 해.
결과는 반드시 JSON 형식의 tool call로만 응답해야 한다.
        """.strip()
    else:
        return """
Available filters:
- Upload date: Last hour, Today, This week, This month, This year
- Type: Video, Channel, Playlist, Movie
- Duration: Under 4 minutes, 4 - 20 minutes, Over 20 minutes
- Features: Live, 4K, HD, Subtitles/CC, Creative Commons, 360°, VR180, 3D, HDR
- Sort by: Relevance, Upload date, View count, Rating

You are an agent that automates YouTube interactions using tools.
Analyze the current DOM to understand the context
and decide which tool to call next.

⚠️ You must call **only one tool at a time.**
Your response must be a single JSON tool call — nothing else.
        """.strip()


async def run_with_xlam(user_input: str):
    await playwright.async_initialize()
    model_name = "qwen3:14b"

    lang = await detect_youtube_language()
    system_prompt = get_system_prompt(lang)

    for _ in range(5):
        prompt = [
            {
                "role": "system",
                "content": system_prompt + "\n\n" + await get_current_dom(),
            },
            {"role": "user", "content": user_input},
        ]

        response = await ollama.AsyncClient().chat(
            model=model_name,
            messages=prompt,
            tools=TOOLS,
        )
        ic(response)

        try:
            func_calls = response.message.tool_calls
            if not func_calls:
                print("⚠️ No tool calls detected.")
                continue
            call = func_calls[0]
            fn_name = call.function.name
            args = call.function.arguments

            if fn_name == "search":
                await search(SearchParams(**args))
            elif fn_name == "apply_youtube_filters":
                await apply_youtube_filters(FilterParams(**args))
            elif fn_name == "click_video_by_title":
                await click_video_by_title(ClickVideoParams(**args))

        except Exception as error:
            ic(error)


if __name__ == "__main__":
    asyncio.run(
        run_with_xlam("Search for Pokémon AMV, apply 4K filter, then click first video")
    )
