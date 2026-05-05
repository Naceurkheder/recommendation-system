
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from models import AnalyzeRequest, ProblemType, Precision, PROBLEM_LABELS

@dataclass(frozen=True)
class EngineSpec:
    id:             str
    name:           str
    vendor:         str
    category:       str
    color:          str
    description:    str
    peak_gflops:    Dict[str, float]
    mem_bw_gbps:    float
    vram_gb:        Optional[float]
    ram_limit_gb:   float
    latency_penalty: float
    complexity:     int
    efficiency:     Dict[str, float]

ENGINES: Dict[str, EngineSpec] = {
    "CUDA": EngineSpec(
        id="CUDA", name="CUDA", vendor="NVIDIA Corporation", category="GPU",
        color="#76B900",
        description="NVIDIA's parallel computing platform — thousands of CUDA cores deliver unmatched throughput for data-parallel compute kernels.",
        peak_gflops={"fp16": 312_000, "fp32": 77_000, "fp64": 19_500, "int8": 624_000},
        mem_bw_gbps=2_000,
        vram_gb=80.0,
        ram_limit_gb=512.0,
        latency_penalty=0.15,
        complexity=3,
        efficiency={
            "embarrassingly_parallel": 0.82,
            "reduction":               0.65,
            "stencil":                 0.58,
            "graph_traversal":         0.30,
            "fft":                     0.75,
            "linear_algebra":          0.88,
            "machine_learning":        0.92,
            "monte_carlo":             0.80,
        },
    ),
    "OpenMP": EngineSpec(
        id="OpenMP", name="OpenMP", vendor="Open Standard (GCC / LLVM / MSVC)", category="CPU-Parallel",
        color="#FF6B00",
        description="Directive-based shared-memory threading — portable across all x86/ARM CPUs with near-zero setup overhead.",
        peak_gflops={"fp16": 3_200, "fp32": 3_200, "fp64": 1_600, "int8": 6_400},
        mem_bw_gbps=350,
        vram_gb=None,
        ram_limit_gb=256.0,
        latency_penalty=0.04,
        complexity=2,
        efficiency={
            "embarrassingly_parallel": 0.70,
            "reduction":               0.78,
            "stencil":                 0.85,
            "graph_traversal":         0.60,
            "fft":                     0.62,
            "linear_algebra":          0.65,
            "machine_learning":        0.55,
            "monte_carlo":             0.72,
        },
    ),
    "MPI": EngineSpec(
        id="MPI", name="MPI", vendor="OpenMPI / MPICH", category="Distributed",
        color="#0073BB",
        description="Message Passing Interface — the de-facto standard for distributed HPC; scales to hundreds of thousands of cores across nodes.",
        peak_gflops={"fp16": 3_200, "fp32": 3_200, "fp64": 1_600, "int8": 6_400},
        mem_bw_gbps=350,
        vram_gb=None,
        ram_limit_gb=1e9,
        latency_penalty=0.25,
        complexity=4,
        efficiency={
            "embarrassingly_parallel": 0.90,
            "reduction":               0.60,
            "stencil":                 0.75,
            "graph_traversal":         0.40,
            "fft":                     0.68,
            "linear_algebra":          0.72,
            "machine_learning":        0.60,
            "monte_carlo":             0.88,
        },
    ),
    "OpenCL": EngineSpec(
        id="OpenCL", name="OpenCL", vendor="Khronos Group (AMD / Intel / NVIDIA)", category="GPU",
        color="#C8202E",
        description="Cross-vendor GPU compute framework — portable across AMD, Intel, and NVIDIA hardware at the cost of ~15–20% peak throughput vs native CUDA.",
        peak_gflops={"fp16": 96_000, "fp32": 48_000, "fp64": 3_000, "int8": 192_000},
        mem_bw_gbps=960,
        vram_gb=24.0,
        ram_limit_gb=512.0,
        latency_penalty=0.20,
        complexity=4,
        efficiency={
            "embarrassingly_parallel": 0.65,
            "reduction":               0.55,
            "stencil":                 0.50,
            "graph_traversal":         0.28,
            "fft":                     0.62,
            "linear_algebra":          0.70,
            "machine_learning":        0.68,
            "monte_carlo":             0.65,
        },
    ),
    "TBB": EngineSpec(
        id="TBB", name="Intel TBB", vendor="Intel Corporation", category="CPU-Parallel",
        color="#0071C5",
        description="Task-based threading with work-stealing scheduler — ideal for irregular and dynamic parallelism patterns on Intel CPU platforms.",
        peak_gflops={"fp16": 3_400, "fp32": 3_000, "fp64": 1_500, "int8": 6_800},
        mem_bw_gbps=360,
        vram_gb=None,
        ram_limit_gb=256.0,
        latency_penalty=0.05,
        complexity=3,
        efficiency={
            "embarrassingly_parallel": 0.68,
            "reduction":               0.72,
            "stencil":                 0.70,
            "graph_traversal":         0.75,
            "fft":                     0.60,
            "linear_algebra":          0.62,
            "machine_learning":        0.52,
            "monte_carlo":             0.70,
        },
    ),
    "SIMD": EngineSpec(
        id="SIMD", name="SIMD / AVX-512", vendor="Intel / AMD (CPU vendor)", category="Vectorization",
        color="#00897B",
        description="Hardware-level vectorization via AVX-512 — lowest overhead of any option; requires loop-level parallelism and regular memory access patterns.",
        peak_gflops={"fp16": 2_000, "fp32": 1_500, "fp64": 750, "int8": 4_000},
        mem_bw_gbps=350,
        vram_gb=None,
        ram_limit_gb=256.0,
        latency_penalty=0.01,
        complexity=5,
        efficiency={
            "embarrassingly_parallel": 0.90,
            "reduction":               0.85,
            "stencil":                 0.88,
            "graph_traversal":         0.20,
            "fft":                     0.80,
            "linear_algebra":          0.85,
            "machine_learning":        0.55,
            "monte_carlo":             0.80,
        },
    ),
}

_WEIGHT_PROFILES: Dict[str, Dict[str, float]] = {
    "embarrassingly_parallel": dict(throughput=0.45, memory=0.15, latency=0.10, scalability=0.20, ease=0.10),
    "reduction":               dict(throughput=0.35, memory=0.20, latency=0.15, scalability=0.20, ease=0.10),
    "stencil":                 dict(throughput=0.25, memory=0.40, latency=0.15, scalability=0.15, ease=0.05),
    "graph_traversal":         dict(throughput=0.20, memory=0.30, latency=0.20, scalability=0.20, ease=0.10),
    "fft":                     dict(throughput=0.40, memory=0.25, latency=0.15, scalability=0.15, ease=0.05),
    "linear_algebra":          dict(throughput=0.45, memory=0.20, latency=0.10, scalability=0.20, ease=0.05),
    "machine_learning":        dict(throughput=0.50, memory=0.20, latency=0.05, scalability=0.20, ease=0.05),
    "monte_carlo":             dict(throughput=0.40, memory=0.15, latency=0.10, scalability=0.25, ease=0.10),
}

_MPI_COMM_OVERHEAD: Dict[str, float] = {
    "embarrassingly_parallel": 0.02,
    "reduction":               0.22,
    "stencil":                 0.12,
    "graph_traversal":         0.38,
    "fft":                     0.18,
    "linear_algebra":          0.16,
    "machine_learning":        0.26,
    "monte_carlo":             0.02,
}

def _compute_effective_gflops(spec: EngineSpec, req: AnalyzeRequest) -> float:

    base   = spec.peak_gflops.get(req.precision.value, spec.peak_gflops["fp32"])
    eff    = spec.efficiency.get(req.problem_type.value, 0.60)
    result = base * eff

    if spec.id == "MPI" and req.node_count > 1:
        comm = _MPI_COMM_OVERHEAD.get(req.problem_type.value, 0.15)
        log_factor = math.log2(max(2, req.node_count)) / 10.0
        scale = 1.0 + (req.node_count - 1) * (1.0 - comm * log_factor)
        result = base * eff * scale

    if spec.category == "GPU" and spec.vram_gb and req.dataset_size_gb > spec.vram_gb:
        ratio   = req.dataset_size_gb / spec.vram_gb
        penalty = max(0.30, 1.0 - math.log(ratio) * 0.25)
        result *= penalty

    if req.memory_bound:
        arith_intensity = 4.0
        roof = spec.mem_bw_gbps * arith_intensity
        result = min(result, roof * eff)

    if req.latency_sensitive:
        result *= (1.0 - spec.latency_penalty * 0.5)

    return round(max(1.0, result), 1)

def _compute_effective_mem_bw(spec: EngineSpec, req: AnalyzeRequest) -> float:

    if req.memory_bound:
        util = {"CUDA": 0.90, "OpenMP": 0.75, "MPI": 0.70,
                "OpenCL": 0.80, "TBB": 0.72, "SIMD": 0.85}.get(spec.id, 0.70)
    else:
        util = {"CUDA": 0.50, "OpenMP": 0.55, "MPI": 0.45,
                "OpenCL": 0.45, "TBB": 0.50, "SIMD": 0.60}.get(spec.id, 0.50)

    bw = spec.mem_bw_gbps * util

    if spec.id == "MPI" and req.node_count > 1:
        infiniband_gbps = 25.0 * req.node_count
        bw = min(bw, infiniband_gbps)

    return round(bw, 1)

def _score_memory(spec: EngineSpec, req: AnalyzeRequest) -> float:

    ds = req.dataset_size_gb

    if spec.category == "GPU":
        vram = spec.vram_gb or 80.0
        if   ds <= vram * 0.75:  return 100.0
        elif ds <= vram:         return 88.0
        elif ds <= vram * 1.5:   return 62.0
        elif ds <= vram * 3.0:   return 38.0
        elif ds <= vram * 10.0:  return 18.0
        else:                    return 8.0

    elif spec.id == "MPI":

        total_ram_gb = 256.0 * max(1, req.node_count)
        if   ds <= total_ram_gb * 0.5:  return 95.0
        elif ds <= total_ram_gb:        return 85.0
        elif ds <= total_ram_gb * 5:    return 68.0
        else:                           return 50.0

    else:

        ram = spec.ram_limit_gb
        if   ds <= ram * 0.25:  return 100.0
        elif ds <= ram * 0.50:  return 95.0
        elif ds <= ram:         return 80.0
        elif ds <= ram * 2.0:   return 22.0
        else:                   return 5.0

def _score_latency(spec: EngineSpec, req: AnalyzeRequest) -> float:

    base = (1.0 - spec.latency_penalty) * 100.0

    if req.latency_sensitive:
        if spec.id in ("CUDA", "OpenCL"):
            base -= 22.0
        elif spec.id == "MPI":
            base -= 28.0

    return round(max(0.0, min(100.0, base)), 1)

def _score_scalability(spec: EngineSpec, req: AnalyzeRequest) -> float:

    BASE = {"CUDA": 70, "OpenMP": 55, "MPI": 88, "OpenCL": 60, "TBB": 55, "SIMD": 30}
    base = float(BASE.get(spec.id, 50))
    ds   = req.dataset_size_gb

    if spec.id == "MPI" and req.node_count > 1:
        base = min(100.0, base + math.log2(req.node_count) * 4)

    if spec.category == "GPU" and spec.vram_gb:
        if ds > spec.vram_gb:
            base -= min(30.0, (ds / spec.vram_gb - 1.0) * 12.0)

    if spec.id not in ("MPI",) and req.node_count > 1:
        base -= min(25.0, (req.node_count - 1) * 3.0)

    if spec.category != "GPU" and spec.id != "MPI":
        if ds > spec.ram_limit_gb:
            base = min(base, 20.0)

    return round(max(0.0, min(100.0, base)), 1)

def _score_ease(spec: EngineSpec) -> float:
    return round((6 - spec.complexity) / 5.0 * 100.0, 1)

def _compute_overall(scores: Dict[str, float], req: AnalyzeRequest) -> float:

    w = dict(_WEIGHT_PROFILES.get(req.problem_type.value,
             dict(throughput=0.35, memory=0.25, latency=0.15, scalability=0.20, ease=0.05)))

    if req.memory_bound:
        boost = min(0.20, w["memory"])
        w["memory"]     = min(0.60, w["memory"] + 0.20)
        w["throughput"] = max(0.10, w["throughput"] - boost)

    if req.latency_sensitive:
        boost = 0.20
        w["latency"]    = min(0.40, w["latency"] + boost)
        w["throughput"] = max(0.10, w["throughput"] - boost * 0.6)
        w["ease"]       = max(0.05, w["ease"] - boost * 0.4)

    total = sum(w.values())
    w = {k: v / total for k, v in w.items()}

    overall = (
        scores["throughput"]  * w["throughput"] +
        scores["memory"]      * w["memory"]     +
        scores["latency"]     * w["latency"]    +
        scores["scalability"] * w["scalability"] +
        scores["ease"]        * w["ease"]
    )
    return round(max(0.0, min(100.0, overall)), 1)

def _tier(score: float) -> str:
    if   score >= 75: return "recommended"
    elif score >= 55: return "suitable"
    elif score >= 35: return "conditional"
    else:             return "avoid"

def _humanise_ds(gb: float) -> str:
    if gb >= 1000:   return f"{gb/1000:.1f} TB"
    if gb >= 1:      return f"{gb:.1f} GB"
    return f"{gb*1024:.0f} MB"

def _generate_reasons(
    spec: EngineSpec,
    req: AnalyzeRequest,
    gflops: float,
    baseline_gflops: float,
    scores: Dict[str, float],
) -> Tuple[List[str], List[str]]:
    reasons: List[str] = []
    warnings: List[str] = []
    ds_str  = _humanise_ds(req.dataset_size_gb)
    speedup = gflops / max(baseline_gflops, 1.0)
    eff     = spec.efficiency.get(req.problem_type.value, 0.60)
    ptype   = PROBLEM_LABELS.get(req.problem_type.value, req.problem_type.value)

    if gflops >= 1_000:
        reasons.append(
            f"Delivers {gflops:,.0f} GFLOPS effective throughput — {speedup:.1f}× "
            f"faster than the OpenMP single-node baseline"
        )
    else:
        reasons.append(
            f"Delivers {gflops:.0f} GFLOPS effective throughput "
            f"({speedup:.2f}× vs OpenMP baseline)"
        )

    if spec.category == "GPU":
        vram = spec.vram_gb or 80.0
        if req.dataset_size_gb <= vram:
            reasons.append(
                f"Dataset ({ds_str}) fits entirely in VRAM ({vram:.0f} GB) — "
                f"no host-device transfer overhead"
            )
        else:
            ratio   = req.dataset_size_gb / vram
            penalty = (1.0 - max(0.30, 1.0 - math.log(ratio) * 0.25)) * 100
            warnings.append(
                f"Dataset ({ds_str}) exceeds VRAM ({vram:.0f} GB) by {ratio:.1f}× — "
                f"memory tiling required; ~{penalty:.0f}% throughput penalty applied"
            )
    elif spec.id == "MPI":
        total_ram = 256 * max(1, req.node_count)
        if req.dataset_size_gb <= total_ram:
            reasons.append(
                f"Dataset ({ds_str}) distributed across {req.node_count} node(s) "
                f"({total_ram:,} GB aggregate RAM) — sufficient capacity"
            )
        else:
            needed = math.ceil(req.dataset_size_gb / 256)
            warnings.append(
                f"Dataset ({ds_str}) requires ≥ {needed} nodes at 256 GB/node; "
                f"currently configured for {req.node_count} node(s)"
            )
    else:
        ram = spec.ram_limit_gb
        if req.dataset_size_gb <= ram:
            reasons.append(
                f"Dataset ({ds_str}) fits in system RAM ({ram:.0f} GB) — no paging"
            )
        else:
            warnings.append(
                f"Dataset ({ds_str}) exceeds system RAM ({ram:.0f} GB) — OS swap "
                f"will severely degrade throughput (10–100× slowdown typical)"
            )

    reasons.append(
        f"{eff*100:.0f}% compute efficiency on {ptype} workloads "
        f"({spec.peak_gflops.get(req.precision.value, spec.peak_gflops['fp32']):,.0f} GFLOPS peak)"
    )

    pt = req.problem_type.value
    if spec.id == "MPI" and req.node_count > 1:
        reasons.append(
            f"Only engine with native multi-node execution — "
            f"spans {req.node_count} nodes for horizontal scale-out"
        )
    if spec.id == "SIMD" and pt in ("stencil", "embarrassingly_parallel", "linear_algebra"):
        reasons.append(
            "AVX-512 zero-overhead vectorisation — hardware executes 16× FP32 "
            "operations per clock cycle with no threading cost"
        )
    if spec.id == "CUDA" and pt == "machine_learning":
        reasons.append(
            "Tensor Core acceleration via cuDNN / cuBLAS — FP16 matrix operations "
            "run at 312,000 GFLOPS with automatic precision scaling"
        )
    if spec.id == "CUDA" and pt == "linear_algebra":
        reasons.append("cuBLAS achieves >85% of peak FP32 FLOPS on dense GEMM operations")
    if spec.id == "TBB" and pt == "graph_traversal":
        reasons.append(
            "Work-stealing scheduler dynamically rebalances irregular task trees — "
            "outperforms static thread-partitioning on unstructured graphs"
        )
    if spec.id == "OpenMP" and pt == "stencil":
        reasons.append(
            "NUMA-aware first-touch allocation and collapse(N) directives achieve "
            "85%+ memory bandwidth on stencil kernels"
        )

    if req.latency_sensitive and spec.id in ("CUDA", "OpenCL"):
        warnings.append(
            "GPU kernel launch latency (5–20 µs) + PCIe transfer overhead "
            "may violate real-time constraints; consider OpenMP for sub-millisecond SLAs"
        )
    if req.latency_sensitive and spec.id == "MPI":
        warnings.append(
            "MPI network round-trip latency (1–100 µs per collective) "
            "is unsuitable for strict real-time requirements"
        )

    if req.node_count > 1 and spec.id not in ("MPI",):
        warnings.append(
            f"Configured for {req.node_count} node(s) but {spec.name} is a "
            f"single-node runtime — {req.node_count - 1} node(s) will be idle"
        )

    if spec.id == "MPI" and req.node_count == 1:
        warnings.append(
            "MPI on a single node incurs message-passing overhead with no scale-out "
            "benefit — prefer OpenMP for equivalent single-node parallelism"
        )

    return reasons, warnings

def _generate_insights(req: AnalyzeRequest, results: List[dict]) -> List[str]:
    insights: List[str] = []
    ds = req.dataset_size_gb
    pt = req.problem_type.value

    if ds < 1:
        insights.append(
            f"At {_humanise_ds(ds)}, the dataset likely resides in CPU last-level cache — "
            f"SIMD and OpenMP will outperform GPU due to zero data-transfer overhead."
        )
    elif ds <= 80:
        insights.append(
            f"At {_humanise_ds(ds)}, data fits in a single GPU's VRAM — "
            f"CUDA can deliver peak throughput without tiling penalty."
        )
    elif ds <= 500:
        insights.append(
            f"At {_humanise_ds(ds)}, dataset exceeds typical single-GPU VRAM. "
            f"Consider NVLink multi-GPU or MPI+CUDA for distributed execution."
        )
    else:
        insights.append(
            f"At {_humanise_ds(ds)}, only MPI (distributed memory) can accommodate "
            f"the full dataset — GPU options require data partitioning."
        )

    if pt == "graph_traversal":
        insights.append(
            "Graph traversal has highly irregular memory access patterns. "
            "GPU and SIMD perform poorly; TBB's work-stealing scheduler is purpose-built for this."
        )
    elif pt == "machine_learning":
        insights.append(
            "ML training is dominated by dense matrix operations. "
            "CUDA Tensor Cores (312 TFLOPS FP16) deliver a structural 20–100× advantage over CPU alternatives."
        )
    elif pt == "stencil":
        insights.append(
            "Stencil kernels are memory-bandwidth-limited. "
            "CUDA's 2 TB/s HBM2e bandwidth (5.7× DDR5) is the decisive factor for large grids."
        )
    elif pt == "monte_carlo":
        insights.append(
            "Monte Carlo is embarrassingly parallel with minimal communication. "
            "MPI achieves near-linear weak scaling — ideal for large sample counts across nodes."
        )

    if req.node_count > 1:
        insights.append(
            f"With {req.node_count} nodes, MPI can theoretically deliver up to "
            f"{req.node_count}× single-node throughput for embarrassingly parallel tasks. "
            f"Communication-heavy kernels will see diminishing returns."
        )

    if req.precision == Precision.fp16:
        insights.append(
            "FP16 precision gives CUDA a 4× throughput advantage over FP32 "
            "(312,000 vs 77,000 GFLOPS). Ensure your algorithm tolerates reduced precision."
        )
    elif req.precision == Precision.fp64:
        insights.append(
            "FP64 reduces CUDA throughput by 4× (19,500 GFLOPS) — "
            "the GPU advantage over CPUs narrows significantly. "
            "Consider FP32 with mixed-precision accumulation if accuracy allows."
        )

    return insights[:4]

@dataclass
class EngineResult:
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
    reasons:             List[str] = field(default_factory=list)
    warnings:            List[str] = field(default_factory=list)

@dataclass
class AnalysisResult:
    engines:            List[EngineResult]
    top_engine_id:      Optional[str]
    baseline_engine_id: Optional[str]
    insights:           List[str]
    workload_summary:   str

def analyze(req: AnalyzeRequest) -> AnalysisResult:

    unknown = [e for e in req.selected_engines if e not in ENGINES]
    if unknown:
        raise ValueError(f"Unknown engine(s): {unknown}")

    raw: List[Dict] = []
    for eid in req.selected_engines:
        spec = ENGINES[eid]
        gflops  = _compute_effective_gflops(spec, req)
        mem_bw  = _compute_effective_mem_bw(spec, req)
        raw.append({
            "spec":           spec,
            "gflops":         gflops,
            "mem_bw":         mem_bw,
            "memory_score":   _score_memory(spec, req),
            "latency_score":  _score_latency(spec, req),
            "scale_score":    _score_scalability(spec, req),
            "ease_score":     _score_ease(spec),
        })

    max_gflops = max(r["gflops"] for r in raw) or 1.0
    for r in raw:
        r["throughput_score"] = round(r["gflops"] / max_gflops * 100.0, 1)

    baseline_id = "OpenMP" if "OpenMP" in req.selected_engines else req.selected_engines[0]
    baseline_gflops = next(
        (r["gflops"] for r in raw if r["spec"].id == baseline_id), 1.0
    )

    results: List[EngineResult] = []
    for r in raw:
        spec   = r["spec"]
        scores = dict(
            throughput  = r["throughput_score"],
            memory      = r["memory_score"],
            latency     = r["latency_score"],
            scalability = r["scale_score"],
            ease        = r["ease_score"],
        )
        overall  = _compute_overall(scores, req)
        speedup  = round(r["gflops"] / max(baseline_gflops, 1.0), 2)
        reasons, warnings = _generate_reasons(spec, req, r["gflops"], baseline_gflops, scores)

        results.append(EngineResult(
            engine_id           = spec.id,
            engine_name         = spec.name,
            category            = spec.category,
            vendor              = spec.vendor,
            color               = spec.color,
            description         = spec.description,
            effective_gflops    = r["gflops"],
            peak_gflops_fp32    = spec.peak_gflops["fp32"],
            effective_mem_bw    = r["mem_bw"],
            throughput_score    = scores["throughput"],
            memory_score        = scores["memory"],
            latency_score       = scores["latency"],
            scalability_score   = scores["scalability"],
            ease_score          = scores["ease"],
            overall_score       = overall,
            recommendation_tier = _tier(overall),
            speedup_vs_baseline = speedup,
            reasons             = reasons,
            warnings            = warnings,
        ))

    results.sort(key=lambda x: x.overall_score, reverse=True)

    top_id   = results[0].engine_id if results else None
    pt_label = PROBLEM_LABELS.get(req.problem_type.value, req.problem_type.value)
    summary  = (
        f"{_humanise_ds(req.dataset_size_gb)} · {pt_label} · "
        f"{req.precision.value.upper()} · {req.node_count} node(s)"
    )

    return AnalysisResult(
        engines            = results,
        top_engine_id      = top_id,
        baseline_engine_id = baseline_id,
        insights           = _generate_insights(req, raw),
        workload_summary   = summary,
    )
