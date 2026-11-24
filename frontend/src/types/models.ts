export interface ModelLicensingInfo {
  status?: string
  notes?: string
  dataset_license?: string
  commercial_status?: string
}

export interface ModelMetadataItem {
  model_id: string
  model_name: string
  model_version: string
  task_description: string
  target_type: string
  status: string
  commercial_use_permitted: boolean
  licensing_category: string
  training_data_source: string
  training_geography: string
  coverage_and_scope_notes: string
  licensing: ModelLicensingInfo
  metrics: Record<string, any>
}

export interface ModelsListResponse {
  platform_name: string
  api_version: string
  total_registered_models: number
  commercial_cleared_count: number
  restricted_research_count: number
  global_extensibility: {
    coordinate_system: string
    supported_units: string[]
    timezone_standard: string
    weather_normalization: string
    provider_adapters: string[]
  }
  models: ModelMetadataItem[]
}
