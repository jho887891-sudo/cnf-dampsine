from pathlib import Path

from scripts.run_mvp import run_mvp


def test_fourier_mvp_writes_metrics_and_predictions(tmp_path):
    result = run_mvp(methods='fourier', output_root=tmp_path, steps_override=1)
    assert result['fourier']['status'] in ('ok', 'numerical_failure')
    assert 'dense_ode_rms' in result['fourier']
    directory = Path(result['fourier']['run_dir'])
    assert (directory / 'summary.json').is_file()
    assert (directory / 'predictions.npz').is_file()
    assert (directory / 'config.yaml').is_file()
