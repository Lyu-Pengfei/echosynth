"""Check public runbook configuration against the documented analysis map."""
from pathlib import Path
import yaml
from echosynth.utils.config import load_config

ROOT = Path(__file__).resolve().parents[2]

def test_legacy_reference_is_explicit_and_has_historical_grid():
    values = load_config(ROOT / 'configs/experiment/duke_two_step_allview.yaml')
    assert values['generator']['sampling_timesteps'] == 500
    assert values['synthetic_grid'] == dict(start=0, stop=2000, step=100)

def test_patient_workflow_configuration():
    root = ROOT / 'workflows/patient_sensitivity'
    diffusion = yaml.safe_load((root/'configs/diffusion/ca_revision_prcc_full.yaml').read_text())
    experiment = yaml.safe_load((root/'configs/experiment/ca_revision_prcc_full.yaml').read_text())
    assert diffusion['sampling_timesteps'] == 100
    assert diffusion['timesteps'] == 1000
    assert experiment['classifier']['batch_size'] == 64
    assert experiment['synthetic_grid'] == dict(start=500, stop=2000, step=500)
    assert experiment['eval']['primary_unit'] == 'patient_id'
    assert '--steps 10000' in (root/'README.md').read_text()

def test_withdrawn_external_interval_is_not_a_reproduction_target():
    text = (ROOT/'tests/integration/test_reproduction.py').read_text()
    assert '"echonet_lvh_delta":' not in text
