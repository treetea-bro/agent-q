from models import ClickVideoParams, FilterParams, SearchParams

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search YouTube by keyword.",
            "parameters": SearchParams.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_youtube_filters",
            "description": "Apply one or more filters to the current YouTube search results.",
            "parameters": FilterParams.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "click_video_by_title",
            "description": "Click a video on YouTube search results by its exact title.",
            "parameters": ClickVideoParams.model_json_schema(),
        },
    },
]
