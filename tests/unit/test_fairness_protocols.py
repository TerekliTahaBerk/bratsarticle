from pathlib import Path

from bratsarticle.experiments.fairness import (
    load_compute_matched_protocol,
    load_convergence_matched_protocol,
)


def test_compute_matched_protocol_is_explicit_and_not_epoch_matched() -> None:
    protocol = load_compute_matched_protocol(
        Path("configs/protocols/compute_matched.yaml")
    )

    assert protocol.gpu_model == "NVIDIA A100-SXM4-80GB"
    assert protocol.maximum_gpu_hours_per_run == 8.0
    assert protocol.maximum_optimizer_steps == 30000
    assert protocol.maximum_tuning_trials_per_family == 4
    assert protocol.input_shape == (4, 240, 240)
    assert protocol.effective_batch_size == (
        protocol.batch_size * protocol.gradient_accumulation_steps
    )
    assert protocol.scheduler.name == "linear_warmup_cosine_decay"
    assert not protocol.epoch_count_is_fairness_criterion
    assert not protocol.test_subset_permitted


def test_convergence_protocol_uses_one_scheduler_and_patient_metric() -> None:
    protocol = load_convergence_matched_protocol(
        Path("configs/protocols/convergence_matched.yaml")
    )

    assert protocol.maximum_optimizer_steps == 50000
    assert protocol.validation_frequency_optimizer_steps == 500
    assert protocol.early_stopping_patience_validation_checks == 12
    assert protocol.minimum_improvement == 0.001
    assert protocol.monitored_metric == "validation_patient_mean_regional_dice"
    assert protocol.scheduler.name == "linear_warmup_cosine_decay"
    assert protocol.scheduler_count == 1
    assert not protocol.epoch_count_is_fairness_criterion
    assert not protocol.test_subset_permitted
