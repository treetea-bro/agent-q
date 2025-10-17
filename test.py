from dotenv import load_dotenv

load_dotenv()

import asyncio
import json

from transformers import AutoModelForCausalLM, AutoTokenizer

from agentq.core.web_driver.playwright import PlaywrightManager
from models import ClickVideoParams, FilterParams, SearchParams
from tools import TOOLS

playwright = PlaywrightManager()


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


# ==============================
# 🔧 Load xLAM Model
# ==============================
model_name = "Salesforce/Llama-xLAM-2-8b-fc-r"
# model_name = "Salesforce/xLAM-2-1b-fc-r"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", dtype="auto"
)


# ==============================
# 🌐 YouTube Language Detection
# ==============================
async def detect_youtube_lang():
    page = await playwright.get_current_page()
    lang = await page.get_attribute("html", "lang")
    if lang:
        lang = lang.lower()
        if lang.startswith("ko"):
            return "ko"
        elif lang.startswith("en"):
            return "en"
    # 기본값 영어
    return "en"


def get_system_prompt(lang: str) -> str:
    if lang == "ko":
        return """
You are a YouTube automation assistant.
사용자의 요청을 수행하기 위해 필요한 모든 단계를 실행하세요.
응답은 반드시 `name`과 `arguments`를 가진 JSON 배열로만 제공합니다.

사용 가능한 액션: search, apply_youtube_filters, click_video_by_title

사용 가능한 필터 (한글):
- 업로드 날짜: 지난 1시간, 오늘, 이번 주, 이번 달, 올해
- 구분: 동영상, 채널, 재생목록, 영화
- 길이: 4분 미만, 4~20분, 20분 초과
- 기능별: 라이브, 4K, HD, 자막, 크리에이티브 커먼즈, 360°, VR180, 3D, HDR
- 위치: 구입한 항목
- 정렬기준: 관련성, 업로드 날짜, 조회수, 평점
"""
    else:
        # 영어
        return """
You are a YouTube automation assistant.
Execute all steps needed to fulfill the user's request.
Respond ONLY in a JSON array of actions with `name` and `arguments`.

Available actions: search, apply_youtube_filters, click_video_by_title

Available filters:
- Upload date: Last hour, Today, This week, This month, This year
- Type: Video, Channel, Playlist, Movie
- Duration: Under 4 minutes, 4 - 20 minutes, Over 20 minutes
- Features: Live, 4K, HD, Subtitles/CC, Creative Commons, 360°, VR180, 3D, HDR
- Location: Purchased
- Sort by: Relevance, Upload date, View count, Rating
"""


# ==============================
# ⚡ Function Calling Runner (리팩토링)
# ==============================
async def run_with_xlam(user_input: str):
    await playwright.async_initialize()

    # YouTube 언어 감지
    lang = await detect_youtube_lang()
    print("Detected YouTube language:", lang)

    # 언어별 시스템 프롬프트 설정
    system_prompt = get_system_prompt(lang)
    print(system_prompt)

    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

    # tokenizer의 chat template 활용
    inputs = tokenizer.apply_chat_template(
        prompt,
        tools=TOOLS,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    input_ids_len = inputs["input_ids"].shape[-1]
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(**inputs, max_new_tokens=1024)
    generated_tokens = outputs[:, input_ids_len:]
    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    print("=== LLM Generated Text ===")
    print(generated_text)

    # JSON function call 순차 실행
    try:
        func_calls = json.loads(generated_text)
        if isinstance(func_calls, list):
            for call in func_calls:
                fn_name = call.get("name")
                args = call.get("arguments", {})

                if fn_name == "search":
                    params = SearchParams(**args)
                    await search(params)
                elif fn_name == "apply_youtube_filters":
                    params = FilterParams(**args)
                    await apply_youtube_filters(params)
                elif fn_name == "click_video_by_title":
                    params = ClickVideoParams(**args)
                    await click_video_by_title(params)

    except Exception as e:
        print("Could not parse function call. LLM output:")
        print(generated_text)
        print("Error:", e)


if __name__ == "__main__":
    asyncio.run(
        run_with_xlam(
            "Search for Pokémon AMV, apply 4K filter, then click the POKEMON [AMV] Legends Never Die"
        )
    )
