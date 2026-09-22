"""Pydantic configuration models for the teval pipeline."""

import yaml
import textwrap
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator


class IOConfig(BaseModel):
    """Configuration for Input/Output paths and file patterns."""

    troute_netcdf_dir: Optional[Path] = Field(
        default=None,
        description=(
            "Directory containing T-Route output subdirectories. "
            "Each subdir must follow naming: '{formulation}_{domain}_output' "
            "(directory_naming='suffix') or be nested under a domain folder "
            "(directory_naming='parent')."
        ),
    )
    ensemble_netcdf_dir: Optional[Path] = Field(
        default=None,
        description="Directory containing pre-computed ensemble NC files.",
    )
    hydrofabric_dir: Path = Field(
        default=Path("data/hydrofabric"),
        description=(
            "Directory containing hydrofabric GeoPackages. "
            "Files matched by domain name: '*{domain_name}*.gpkg'."
        ),
    )
    hydrofabric_layer: str = Field(
        default="flowpaths",
        description=(
            "The layer to read from the hydrofabric, flowpaths or flowlines."
        ),
    )
    observations_file: Optional[Path] = Field(
        default=None,
        description=(
            "Path to observations file (.csv or .parquet). "
            "Columns = gage IDs (strings), index = datetime."
        ),
    )
    auto_download_usgs: bool = Field(
        default=False,
        description=(
            "If True and observations_file is missing, automatically download "
            "streamflow data from USGS NWIS for all gages found in the hydrofabric."
        ),
    )
    offline: bool = Field(
        default=False,
        description=(
            "Run without network access, as on compute nodes that have none. "
            "No USGS download is attempted and no basemap is drawn. Observations "
            "must already be in observations_file (see --fetch); the run checks "
            "this before processing and stops if they are not."
        ),
    )
    save_downloaded_obs: Optional[Path] = Field(
        default=None,
        description=(
            "Path to cache auto-downloaded USGS observations. "
            "If null, downloaded data is not saved to disk."
        ),
    )
    output_dir: Path = Field(
        default=Path("output"),
        description="Root directory for all outputs (plots, maps, CSVs, animations).",
    )
    per_domain_output: bool = Field(
        default=True,
        description=(
            "If True, each domain's outputs are saved to output_dir/{domain_name}/. "
            "If False, all outputs go to output_dir/ (useful for single-domain CONUS runs)."
        ),
    )
    directory_naming: Literal["suffix", "parent"] = Field(
        default="suffix",
        description=(
            "Controls how domain name is extracted from the run directory structure.\n"
            "  'suffix': last underscore-segment of '{formulation}_{domain}_output'.\n"
            "  'parent': parent directory name is the domain "
            "(e.g. runs/12009000/cfe_output/)."
        ),
    )
    metrics_output_file: Optional[str] = Field(
        default="metrics.csv",
        description=(
            "Filename for the combined metrics CSV, saved to output_dir. "
            "Set to null to skip saving."
        ),
    )

    @field_validator("output_dir", mode="before")
    def convert_to_path(cls, v):
        """Convert a string value to a Path object."""
        return Path(v) if v else v


class SystemConfig(BaseModel):
    """Configuration for system resources, execution strategy, and runtime behavior."""

    cpu: int = Field(
        default=-1,
        description=(
            "Number of CPUs for all parallel work (Dask compute, hydrograph "
            "rendering, animation frames, skill maps). Used as given, even beyond "
            "a Slurm allocation. -1 = the CPUs allocated to this job: "
            "SLURM_CPUS_PER_TASK under Slurm, otherwise the cores this process "
            "may run on."
        ),
    )
    domain_workers: int = Field(
        default=1,
        description=(
            "Number of domains to process simultaneously. "
            "1 = serial (default, safe for debugging). "
            "-1 = let the planner auto-select based on domain scale and CPU count. "
            "Any value > 1 = explicit parallelism override."
        ),
    )
    stream_to_disk: Optional[bool] = Field(
        default=None,
        description=(
            "Whether to write the full computed ensemble statistics to a NetCDF "
            "file on disk. The *_ensemble.nc is the PRIMARY OUTPUT of this "
            "pipeline — the final post-processed ensemble value for every "
            "feature_id at every timestep. Metrics and figures are "
            "derived from it.\n"
            "  true or null (default) = write the ensemble NC.\n"
            "  false = skip the write; compute only the gage-associated feature "
            "subset into RAM. Use only when you explicitly do not need the "
            "output NC — e.g. a quick metrics-only re-run when the ensemble NC "
            "already exists on disk and is pointed to by ensemble_netcdf_dir."
        ),
    )
    use_dask: Optional[bool] = Field(
    default=None,
    description=(
        "Use Dask for lazy/parallel array loading via xr.open_mfdataset. "
        "null (recommended) = auto-detect. "
        "true = force Dask. false = force eager loading."
        ),
    )
    logging_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description=(
            "Controls log verbosity for the entire run.\n"
            "  DEBUG   : very detailed internal state (development use)\n"
            "  INFO    : standard progress messages (recommended)\n"
            "  WARNING : only warnings and errors\n"
            "  ERROR   : only errors"
        ),
    )
    timing: Literal["none", "simple", "verbose"] = Field(
        default="simple",
        description=(
            "Controls execution timing output.\n"
            "  none    : timing disabled entirely\n"
            "  simple  : timings recorded silently; a summary table is printed at "
            "the end of the run\n"
            "  verbose : start/end of each timed block is logged immediately and "
            "a summary table is printed at the end"
        ),
    )


class DataConfig(BaseModel):
    """Configuration for data slicing and subsetting."""

    time_slice: Optional[List[Union[int, str]]] = Field(
        default=None,
        description=(
            "Subset the data by time. Provide two ISO date strings: "
            "['2023-01-01', '2023-12-31']. Set to null to use all available time."
        ),
    )
    feature_ids: Union[List[int], Literal["all"]] = Field(
        default="all",
        description=(
            "List of integer Feature IDs to process. Use 'all' to process "
            "every feature in the domain."
        ),
    )


class WeightsConfig(BaseModel):
    """
    Configuration for spatially varying ensemble weights.

    Supplying this block switches the ensemble mean from a simple mean to a
    weighted sum over the formulation dimension.  Weights are keyed by nexus
    and by formulation index; every flowpath draining to a nexus receives that
    nexus' weights.  Median and the spread band remain unweighted.

    The weight file format is provisional and expected to change.
    """

    file: Path = Field(
        description=(
            "Path to the weight file (.csv or .parquet). "
            "Columns: 'nexus_id' (string, 'nex-' prefix retained), "
            "'formulation_index' (1-based integer) and 'weight' (float). "
            "One row per (nexus, formulation) pair."
        ),
    )
    formulation_index_map: Dict[int, str] = Field(
        description=(
            "Binding from the integer 'formulation_index' used in the weight "
            "file to the formulation name teval parses from run directories, "
            "e.g. {1: 'cfe', 2: 'noahowp'}. Required because the order of the "
            "formulation dimension follows directory scan order and is not "
            "stable across machines, so indices cannot be read positionally. "
            "Must name exactly the formulations discovered in the run — an "
            "index naming an absent formulation, or a discovered formulation "
            "missing from the map, is an error."
        ),
    )
    on_missing: Literal["warn", "error"] = Field(
        default="warn",
        description=(
            "Policy for features in the run whose nexus is absent from the "
            "weight file.\n"
            "  'warn'  : those features fall back to equal weights (the simple "
            "mean); covered/uncovered counts and the coverage fraction are "
            "logged at warning level.\n"
            "  'error' : incomplete coverage aborts the run, before any "
            "compute is triggered."
        ),
    )
    normalize: bool = Field(
        default=False,
        description=(
            "If True, divide each nexus' weights by their sum, accepting any "
            "positive scale. If False (default), each group must already sum "
            "to 1.0 within a tolerance of 1e-6 or the run aborts."
        ),
    )

    @field_validator("formulation_index_map")
    def validate_formulation_index_map(cls, v):
        """
        Ensure the map names something and numbers it from 1.

        Both rules hold whatever the run turns out to contain — an empty legend
        binds no formulation at all, and 1 is the base of the file format
        itself — so they fail at config load rather than per-domain after
        discovery and hydrofabric loading.  Whether the map names the *right*
        formulations does need the run; the resolver settles that with one set
        comparison, and it is not duplicated here.
        """
        if not v:
            raise ValueError(
                "formulation_index_map is empty; weighting needs a legend "
                "binding each weight-file index to a formulation name."
            )

        bad_indices = sorted(i for i in v if i < 1)
        if bad_indices:
            raise ValueError(
                f"formulation_index_map keys must be 1-based positive integers; "
                f"got {bad_indices}."
            )
        return v


class StatsConfig(BaseModel):
    """Configuration for ensemble statistical calculations."""

    enabled: bool = Field(
            default=False,
            description="Whether to calculate mean/median/spread from ensemble members.",
        )
    quantiles: List[float] = Field(
        default=[0.05, 0.95],
        description=(
            "Quantile bounds for ensemble uncertainty bands (values between 0.0 and 1.0). "
            "Example: [0.05, 0.95] produces the 5th–95th percentile spread."
        ),
    )
    small_domain_threshold: int = Field(
        default=10,
        description=(
            "Ensemble size below which min/max is used for the spread band instead of "
            "quantiles.  Multi-domain calibration runs typically have 3–5 formulations "
            "so min/max is more meaningful; full CONUS runs with 10+ formulations use "
            "the configured quantiles."
        ),
    )
    weights: Optional[WeightsConfig] = Field(
        default=None,
        description=(
            "Optional spatially varying ensemble weights. Omit the block "
            "entirely (or set to null) for the default simple mean. When "
            "present it requires 'file' and 'formulation_index_map'; see the "
            "weights section of the configuration guide. Requires a "
            "hydrofabric, and has no effect when a pre-computed ensemble "
            "NetCDF is reused."
        ),
    )

    @field_validator("quantiles")
    def validate_quantiles(cls, v):
        """Ensure all quantile values are between 0 and 1."""
        if not all(0 <= q <= 1 for q in v):
            raise ValueError("Quantiles must be between 0 and 1.")
        return v


class MetricsConfig(BaseModel):
    """Configuration for performance metrics and hypothesis testing."""

    enabled: bool = Field(
        default=False,
        description="Whether to calculate performance metrics against observations.",
    )
    variables: List[str] = Field(
        default=["nse", "kge", "pbias", "peak_flow_error"],
        description=(
            "List of deterministic metrics to calculate. "
            "Options: 'nse', 'kge', 'pbias', 'peak_flow_error', 'peak_timing_error'."
        ),
    )
    per_formulation: bool = Field(
        default=False,
        description=(
            "If True, calculate metrics for each individual formulation in "
            "addition to the ensemble mean."
        ),
    )
    bootstrap_enabled: bool = Field(
        default=False,
        description=(
            "Whether to perform bootstrapping to estimate confidence intervals "
            "and classify skill (skillful / unskillful / indeterminate)."
        ),
    )
    bootstrap_samples: int = Field(
        default=1000,
        description="Number of bootstrap resampling iterations.",
    )
    confidence_level: float = Field(
        default=0.95,
        description="Confidence level for the bootstrap hypothesis test (e.g. 0.95 = 95% CI).",
    )


class HydrographConfig(BaseModel):
    """Settings for hydrograph time-series plots."""

    enabled: bool = Field(default=True, description="Whether to generate hydrograph plots.")
    target_ids: List[int] = Field(
        default=[],
        description=(
            "Specific Gage IDs to plot. If empty, plots all gages that have "
            "matching observations."
        ),
    )
    plot_uncertainty: bool = Field(
        default=True,
        description="Include a shaded uncertainty band between the configured quantiles.",
    )
    plot_members: bool = Field(
        default=False,
        description=(
            "If True, plot each individual ensemble member as a faint trace "
            "(spaghetti plot style) instead of the uncertainty band."
        ),
    )


class SkillMapsConfig(BaseModel):
    """
    Post-processing skill assessment maps and charts.

    These run after all domains are processed and require metrics.enabled=true.
    Three output types per metric: winner scatter map, score boxplots, and
    (optionally) a VPU regional breakdown stacked bar.
    """

    enabled: bool = Field(default=True, description="Master toggle for all skill map outputs.")
    winner_maps: bool = Field(
        default=True,
        description="Scatter map of which model performed best at each gage.",
    )
    boxplots: bool = Field(
        default=True,
        description="Boxplot of score distributions across formulations.",
    )
    vpu_breakdown: bool = Field(
        default=False,
        description=(
            "Stacked-bar win-rate by VPU. CONUS-scale only — requires a "
            "hydrofabric GeoPackage with a 'flowpath-attributes' layer "
            "containing a 'vpuid' column."
        ),
    )
    variables: List[str] = Field(
        default=["nse", "kge", "pbias"],
        description="Metrics to produce skill maps and boxplots for.",
    )
    score_maps: bool = Field(
        default=True,
        description=(
            "Generate a per-source scatter map for each metric — one PNG per "
            "(metric, source) combination, coloured continuously by score value. "
            "These replace the former metrics_maps outputs."
        ),
    )
    basemap: bool = Field(
        default=True,
        description=(
            "Add a CartoDB.Positron basemap to all spatial map outputs "
            "(requires internet access)."
        ),
    )


class InteractiveMapConfig(BaseModel):
    """Settings for the interactive HTML Folium map."""

    enabled: bool = Field(
        default=True,
        description="Whether to generate an interactive Folium map.",
    )
    variable: str = Field(
        default="streamflow_mean",
        description="Variable to display on the map.",
    )


class AnimationConfig(BaseModel):
    """Settings for generating GIF animations of flow through the network."""

    enabled: bool = Field(
        default=False,
        description="Whether to generate a GIF animation (time-intensive for large domains).",
    )
    variable: str = Field(default="streamflow_mean", description="Variable to animate.")
    fps: int = Field(8, ge=1, le=60, description="Frames per second for the output GIF.")
    log_scale: bool = Field(
        default=True,
        description="Use logarithmic color scaling (strongly recommended for streamflow).",
    )
    cmap: str = Field(
        default="hydro_flow",
        description="Colormap name ('hydro_flow', 'viridis', 'Blues', etc.).",
    )
    time_step: str = Field(
        default="1W",
        description=(
            "Time step between animation frames. "
            "Accepts offset aliases ('1W', '1D', '3D') or integer step counts."
        ),
    )
    min_stream_order: int = Field(
        default=4,
        description=(
            "Minimum Strahler stream order to include. Higher values reduce "
            "the number of flowpaths and speed up rendering."
        ),
    )


class VizConfig(BaseModel):
    """Visualization configuration grouping."""

    hydrographs: HydrographConfig = HydrographConfig()
    skill_maps: SkillMapsConfig = SkillMapsConfig()
    interactive_map: InteractiveMapConfig = InteractiveMapConfig()
    animation: AnimationConfig = AnimationConfig()


class TevalConfig(BaseModel):
    """Root configuration object for TEVAL."""

    io: IOConfig = IOConfig()
    system: SystemConfig = SystemConfig()
    data: DataConfig = DataConfig()
    stats: StatsConfig = StatsConfig()
    metrics: MetricsConfig = MetricsConfig()
    viz: VizConfig = VizConfig()

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "TevalConfig":
        """Load and validate a TevalConfig from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls(**raw)


def generate_default_config(path: str = "teval_config.yaml"):
    """Generates a default YAML configuration file based on model defaults."""
    default_dict = TevalConfig().model_dump()

    def clean_dict(d):
        """Recursively remove None values from a nested dict."""
        for k, v in d.items():
            if isinstance(v, dict):
                clean_dict(v)
            elif isinstance(v, Path):
                d[k] = str(v)

    clean_dict(default_dict)
    with open(path, "w") as f:
        yaml.dump(default_dict, f, sort_keys=False, default_flow_style=False)


def generate_config_help() -> str:
    """Introspects TevalConfig to generate a human-readable configuration guide."""
    lines = [
        "TEVAL CONFIGURATION GUIDE",
        "=" * 80,
        "This guide explains all configuration options for your teval_config.yaml.\n",
    ]
    for section_name, field_info in TevalConfig.model_fields.items():
        section_model = field_info.annotation
        lines.append(f"SECTION: {section_name}")
        lines.append("-" * 40)
        if section_model.__doc__:
            lines.append(f"{section_model.__doc__}\n")
        for key, prop in section_model.model_fields.items():
            desc = prop.description or "No description provided."
            wrapped = textwrap.fill(
                desc, width=70, initial_indent="    ", subsequent_indent="    "
            )
            lines.append(f"  • {key} (Default: {prop.default})")
            lines.append(wrapped)
            lines.append("")
        lines.append("")
    return "\n".join(lines)