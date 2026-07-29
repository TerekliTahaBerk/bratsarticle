from bratsarticle.experiments.pilot_runner import PatientGroupedSampler


def test_patient_grouped_sampler_is_complete_grouped_and_deterministic() -> None:
    sampler = PatientGroupedSampler(
        patient_count=4,
        samples_per_patient=3,
        seed=17,
    )

    first = list(sampler)
    sampler.set_epoch(1)
    second = list(sampler)
    sampler.set_epoch(0)

    assert sorted(first) == list(range(12))
    assert sorted(second) == list(range(12))
    assert first != second
    assert list(sampler) == first
    for offset in range(0, 12, 3):
        group = first[offset : offset + 3]
        assert group == list(range(group[0], group[0] + 3))
