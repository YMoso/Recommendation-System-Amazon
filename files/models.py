from pydantic import BaseModel, Field
from typing import List, Optional

class HistoryItem(BaseModel):
    asin: str
    title: str
    rating: float
    category: Optional[str] = None

class RecommendationItem(BaseModel):
    asin: str
    title: str
    score: float
    model: str
    category: Optional[str] = None

class RecommendRequest(BaseModel):
    user_id: str
    k: int = Field(default=5, ge=1, le=50)

class UserStatusResponse(BaseModel):
    exists: bool
    interaction_count: Optional[int] = None
    cf_eligible: Optional[bool] = None

class RecommendResponse(BaseModel):
    user_id: str
    cf_eligible: bool
    history: List[HistoryItem]
    recommendations: List[RecommendationItem]