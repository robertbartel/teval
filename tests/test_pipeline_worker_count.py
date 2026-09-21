"""
Every parallel path sizes itself from ``system.cpu``.

An explicit value is used as given.  -1 means the CPUs this job was allocated
-- ``SLURM_CPUS_PER_TASK`` under Slurm, otherwise the cores this process may
run on -- never every core on the node.
"""

from __future__ import annotations

import os

import dask
import pytest

from teval import pipeline, workflow
from teval.config import TevalConfig


@pytest.fixture
def config():
    return TevalConfig()


@pytest.fixture
def no_slurm(monkeypatch):
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)


def test_an_explicit_cpu_wins_over_the_slurm_allocation(config, monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "64")
    config.system.cpu = 8

    assert pipeline.get_worker_count(config) == 8


def test_an_explicit_cpu_may_exceed_the_slurm_allocation(config, monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    config.system.cpu = 8

    assert pipeline.get_worker_count(config) == 8


def test_all_cpus_means_the_slurm_allocation(config, monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "64")

    assert pipeline.get_worker_count(config) == 64


@pytest.mark.skipif(
    not hasattr(os, "sched_getaffinity"), reason="no CPU affinity on this platform"
)
def test_all_cpus_outside_slurm_means_the_cores_this_process_may_use(
    config, monkeypatch, no_slurm
):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2})

    assert pipeline.get_worker_count(config) == 3


def test_dask_is_sized_to_the_worker_count(config, monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "6")

    with dask.config.set(num_workers=None):
        pipeline.configure_dask(config)

        assert dask.config.get("num_workers") == 6


def test_domain_visualizations_get_the_worker_count(config, monkeypatch):
    """Hydrographs and animation render with the run's count, not the node's."""
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "6")
    config.viz.hydrographs.enabled = True
    received = []

    monkeypatch.setattr(
        workflow, "load_domain_data", lambda domain_dict, io, stats: {"formulations": {}}
    )
    monkeypatch.setattr(
        pipeline, "compute_and_write", lambda name, data, domain_dict, config: data
    )
    monkeypatch.setattr(
        workflow,
        "produce_domain_specific_visualizations",
        lambda data, viz, io, stats, n_workers: received.append(n_workers),
    )

    pipeline.run_domain("domain", {}, config)

    assert received == [6]
