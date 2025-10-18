from pydantic import BaseModel, Field, Literal


class FilterOption(BaseModel):
    group_name: Literal["업로드 날짜", "구분", "길이", "기능별", "위치", "정렬기준"]
    option_label: str


class SearchParams(BaseModel):
    query: str = Field(..., description="검색어 (예: 'Pokémon AMV')")


class FilterParams(BaseModel):
    filters: list[FilterOption] = Field(..., description="적용할 유튜브 필터 리스트")


class ClickVideoParams(BaseModel):
    title: str = Field(..., description="정확히 일치하는 영상 제목")
