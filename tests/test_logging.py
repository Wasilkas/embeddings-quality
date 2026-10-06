import logging

import numpy as np
import pytest
from loguru import logger

from embeddings_quality import AuditConfig, evaluate_embeddings
from embeddings_quality.cli import main


@pytest.fixture
def small_data():
    rng = np.random.default_rng(11)
    labels = np.repeat(np.arange(3), 6)
    embeddings = np.eye(3)[labels] + rng.normal(0, 0.05, (len(labels), 3))
    return embeddings, labels


@pytest.fixture
def log_messages():
    messages = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]),
        level="INFO",
        filter="embeddings_quality",
    )
    logger.enable("embeddings_quality")
    try:
        yield messages
    finally:
        logger.remove(sink)
        logger.disable("embeddings_quality")


def test_api_logs_stages_and_intermediate_geometry_progress(small_data, log_messages, monkeypatch):
    embeddings, labels = small_data
    ticks = iter(range(10000))
    monkeypatch.setattr("embeddings_quality.geometry.perf_counter", lambda: next(ticks))
    report = evaluate_embeddings(
        embeddings,
        labels,
        metadata={"area": np.arange(len(labels)), "constant": np.ones(len(labels))},
        config=AuditConfig(ks=(1,), cv_folds=2),
    )
    messages = log_messages
    for stage in ("Stage 1/4", "Stage 2/4", "Stage 3/4", "Stage 4/4"):
        assert any(stage in message and "started" in message for message in messages)
        assert any(stage in message and "completed in" in message for message in messages)
    progress = [message for message in messages if message.startswith("Geometry progress:")]
    assert any("100.0%" not in message for message in progress)
    assert "18/18 samples (100.0%)" in progress[-1]
    assert any("OOF fold 1/2" in message and "linear" in message for message in messages)
    assert any("OOF fold 1/2" in message and "kNN" in message for message in messages)
    assert any("Pair 3/3" in message and "fold 2/2" in message for message in messages)
    assert any("Metadata 1/2: 'area', regression fold 2/2" in message for message in messages)
    assert any("Metadata 2/2: 'constant': skipped" in message for message in messages)
    assert report.summary["linear_probe"]["oof"]["linear"]["balanced_accuracy"] == 1


def test_skipped_probes_are_visible(log_messages):
    report = evaluate_embeddings(np.eye(2), ["a", "b"], config=AuditConfig(ks=(1,)))
    text = "\n".join(log_messages)
    assert report.summary["linear_probe"]["status"] == "skipped"
    assert "OOF probes skipped:" in text
    assert "Pair 1/1: 'a' vs 'b': skipped" in text
    assert "Nuisance probes skipped: no metadata supplied" in text
    assert "OOF fold" not in text


def test_failed_stage_does_not_log_success(small_data, log_messages, monkeypatch):
    def fail_geometry(*args, **kwargs):
        raise RuntimeError("geometry failure")

    monkeypatch.setattr("embeddings_quality.audit.geometry", fail_geometry)
    with pytest.raises(RuntimeError, match="geometry failure"):
        evaluate_embeddings(*small_data)
    text = "\n".join(log_messages)
    assert "failed after" in text
    assert "Stage 2/4" not in text
    assert "Embedding analysis: completed" not in text


def test_cli_logs_to_stderr_without_duplicates_or_standard_logging_changes(
    small_data, tmp_path, capsys
):
    embeddings, labels = small_data
    path = tmp_path / "data.npz"
    np.savez(path, embeddings=embeddings, labels=labels)
    root_logger = logging.getLogger()
    old_root_level, old_root_handlers = root_logger.level, list(root_logger.handlers)
    args = [str(path), "--output", str(tmp_path / "report"), "--ks", "1", "--cv-folds", "2"]
    for level in (None, "warning", "DEBUG"):
        invocation = args if level is None else [*args, "--log-level", level]
        assert main(invocation) == 0
        captured = capsys.readouterr()
        assert "Report saved" in captured.out
        assert "Stage 1/4" not in captured.out
        if level == "warning":
            assert "Stage 1/4" not in captured.err
        else:
            assert "Stage 1/4" in captured.err
            assert (
                captured.err.count(
                    "Stage 1/4: geometry (purity, silhouette, distance margins): started"
                )
                == 1
            )
            assert "Saving report" in captured.err
        if level == "DEBUG":
            assert "distance block starting at row" in captured.err
            assert "Saving samples.csv" in captured.err
        assert root_logger.level == old_root_level
        assert root_logger.handlers == old_root_handlers
    with pytest.raises(SystemExit):
        main([str(tmp_path / "missing.npz"), "--output", str(tmp_path / "report")])
    # The CLI's temporary sink is removed even on failure.
    logger.enable("embeddings_quality")
    evaluate_embeddings(embeddings, labels, config=AuditConfig(ks=(1,), cv_folds=2))
    captured = capsys.readouterr()
    assert "Embedding analysis: started" not in captured.err
    logger.disable("embeddings_quality")


def test_api_respects_loguru_activation_and_application_sinks(small_data, log_messages):
    logger.disable("embeddings_quality")
    evaluate_embeddings(*small_data, config=AuditConfig(ks=(1,), cv_folds=2))
    assert not log_messages
    logger.enable("embeddings_quality")
    for _ in range(2):
        evaluate_embeddings(*small_data, config=AuditConfig(ks=(1,), cv_folds=2))
    assert (
        log_messages.count("Embedding analysis: started; validating inputs and preprocessing") == 2
    )
    assert all("%s" not in message and "%d" not in message for message in log_messages)
