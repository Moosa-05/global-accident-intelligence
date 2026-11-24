from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class ModelMetadataItem(BaseModel):
    model_id: str
    model_name: str
    model_version: str
    task_description: str
    target_type: str
    status: str
    commercial_use_permitted: bool
    licensing_category: str
    training_data_source: str
    training_geography: str
    coverage_and_scope_notes: str
    licensing: Dict[str, str]
    metrics: Dict[str, Any]

class ModelsListResponse(BaseModel):
    platform_name: str
    api_version: str
    total_registered_models: int
    commercial_cleared_count: int
    restricted_research_count: int
    global_extensibility: Dict[str, Any]
    models: List[ModelMetadataItem]
