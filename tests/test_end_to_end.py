from pathlib import Path

from scripts.run_mvp import run_mvp
import yaml


def test_fourier_mvp_writes_metrics_and_predictions(tmp_path):
    result = run_mvp(methods='fourier', output_root=tmp_path, steps_override=1)
    assert result['fourier']['status'] in ('ok', 'numerical_failure')
    assert 'dense_ode_rms' in result['fourier']
    directory = Path(result['fourier']['run_dir'])
    assert (directory / 'summary.json').is_file()
    assert (directory / 'predictions.npz').is_file()
    assert (directory / 'config.yaml').is_file()


def test_rectangular_cnf_uses_fixed_basis_count_and_records_feasibility(tmp_path):
    for method, interior in (('cnf_gaussian', 15), ('cnf_gaussian', 63),
                             ('cnf_siren', 15), ('cnf_siren', 63)):
        config = tmp_path / f'{method}_{interior}.yaml'
        config.write_text(yaml.safe_dump({
            'seed': 42, 'basis_count': 33, 'collocation_interior': interior,
            'rectangular_extension': True, 'dense_points': 64,
            'observations': 16, 'device': 'cpu'}))
        result = run_mvp(methods=method, output_root=tmp_path / f'{method}_{interior}',
                         steps_override=1, config_path=config)[method]
        assert result['status'] in ('ok', 'infeasible', 'numerical_failure')
        assert result['matrix_shape'] == [interior + 2, 33]
        assert isinstance(result['hard_feasible'], bool)
        assert result['solver'] in ('minimum_norm_underdetermined',
                                    'least_squares_overdetermined')
