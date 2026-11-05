"""Formal P2 runs must use disjoint samples and resume without retraining."""

import json

import pytest
import yaml

from scripts import run_formal_noise


def test_disjoint_formal_splits_and_resume(tmp_path, monkeypatch):
    config = tmp_path / 'noise.yaml'
    config.write_text(yaml.safe_dump({'base': {'sensors': 128, 'observations': 32}}))
    calls = []

    def fake_run_mvp(*, methods, output_root, config_path):
        case = yaml.safe_load(config_path.read_text())
        calls.append((methods, case['seed'], case['noise']))
        directory = output_root / methods
        directory.mkdir(parents=True)
        (directory / 'summary.json').write_text(json.dumps({
            'method': methods, 'seed': case['seed'], 'noise': case['noise'], 'status': 'ok'}))

    monkeypatch.setattr(run_formal_noise, 'run_mvp', fake_run_mvp)
    root = tmp_path / 'results'
    run_formal_noise.run(config, root, ('fourier',), split_seeds=(42, 123), sample_count=3)
    assert len(calls) == 12
    assert {seed for _, seed, _ in calls} == {1_042_000, 1_042_001, 1_042_002,
                                               1_123_000, 1_123_001, 1_123_002}
    assert yaml.safe_load((root / 'seed42/noise01/sample000/input.yaml').read_text())['device'] == 'auto'
    run_formal_noise.run(config, root, ('fourier',), split_seeds=(42, 123), sample_count=3)
    assert len(calls) == 12
    with pytest.raises(ValueError, match='protocol differs'):
        run_formal_noise.run(config, root, ('fourier',), split_seeds=(42, 123), sample_count=4)
