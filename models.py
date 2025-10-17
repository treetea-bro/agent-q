from pydantic import BaseModel, Field


class YoutubeFilter(BaseModel):
    group_name: str = Field(
        ..., description="필터 그룹 이름 (예: 'Upload date', 'Type', 'Duration')"
    )
    option_label: str = Field(
        ...,
        description="선택할 옵션 라벨 (예: 'This week', 'Video', 'Under 4 minutes')",
    )


class SearchParams(BaseModel):
    query: str = Field(..., description="검색어 (예: 'Pokémon AMV')")


class FilterParams(BaseModel):
    filters: list[YoutubeFilter] = Field(..., description="적용할 유튜브 필터 리스트")


class ClickVideoParams(BaseModel):
    title: str = Field(..., description="정확히 일치하는 영상 제목")
