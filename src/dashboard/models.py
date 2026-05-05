
from __future__ import annotations
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

class ProblemType(str, Enum):
    embarrassingly_parallel = "embarrassingly_parallel"
    reduction               = "reduction"
    stencil                 = "stencil"
    graph_traversal         = "graph_traversal"
    fft                     = "fft"
    linear_algebra          = "linear_algebra"
    machine_learning        = "machine_learning"
    monte_carlo             = "monte_carlo"

PROBLEM_LABELS: dict[str, str] = {
    "embarrassingly_parallel": "Embarrassingly Parallel",
    "reduction":               "Reduction / Aggregation",
    "stencil":                 "Stencil / Neighbor Access",
    "graph_traversal":         "Graph Traversal (irregular)",
    "fft":                     "FFT / Spectral Methods",
    "linear_algebra":          "Dense Linear Algebra (BLAS)",
    "machine_learning":        "Machine Learning / Training",
    "monte_carlo":             "Monte Carlo / Sampling",
}

class Precision(str, Enum):
    fp16 = "fp16"
    fp32 = "fp32"
    fp64 = "fp64"
    int8 = "int8"

class AnalyzeRequest(BaseModel):
    dataset_size_gb:   float        = Field(..., gt=0, le=1_000_000, description="Dataset size in GB")
    problem_type:      ProblemType
    node_count:        int          = Field(default=1, ge=1, le=100_000)
    precision:         Precision    = Precision.fp32
    iterations:        int          = Field(default=100, ge=1)
    memory_bound:      bool         = False
    latency_sensitive: bool         = False
    selected_engines:  List[str]    = Field(min_length=1, max_length=6)

class EngineResultModel(BaseModel):
    engine_id:           str
    engine_name:         str
    category:            str
    vendor:              str
    color:               str
    description:         str
    effective_gflops:    float
    peak_gflops_fp32:    float
    effective_mem_bw:    float
    throughput_score:    float
    memory_score:        float
    latency_score:       float
    scalability_score:   float
    ease_score:          float
    overall_score:       float
    recommendation_tier: str
    speedup_vs_baseline: float
    reasons:             List[str]
    warnings:            List[str]

class AnalysisResultModel(BaseModel):
    engines:            List[EngineResultModel]
    top_engine_id:      Optional[str]
    baseline_engine_id: Optional[str]
    insights:           List[str]
    workload_summary:   str

class EngineInfoModel(BaseModel):
    id:           str
    name:         str
    vendor:       str
    category:     str
    color:        str
    description:  str
    peak_gflops:  dict[str, float]
    mem_bw_gbps:  float
    vram_gb:      Optional[float]
    ram_limit_gb: float
    complexity:   int
