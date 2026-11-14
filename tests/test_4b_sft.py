from pathlib import Path
import yaml
from neuronforge_engine.config import load_config

ROOT=Path(__file__).resolve().parents[1]

def test_sft_config_matches_pretraining_architecture():
    base=load_config(ROOT/"configs"/"4b.yaml")
    sft=load_config(ROOT/"configs"/"4b-sft.yaml")
    assert base.model.parameter_count()==4_064_498_688
    assert sft.model.parameter_count()==base.model.parameter_count()
    assert sft.train.learning_rate<base.train.learning_rate
    assert sft.train.gradient_checkpointing

def test_sft_policy_requires_reviewed_local_data():
    policy=yaml.safe_load((ROOT/"data"/"4b-sft-sources.yaml").read_text(encoding="utf-8"))
    assert "locally reviewed JSONL" in policy["starter_policy"]

def test_sft_scripts_compile():
    for name in ("scripts/prepare_4b_sft.py","scripts/instruction_tune_4b_fsdp.py"):
        source=(ROOT/name).read_text(encoding="utf-8")
        compile(source,name,"exec")


def test_sft_starts_from_model_weights_with_fresh_optimizer():
    source=(ROOT/"scripts"/"instruction_tune_4b_fsdp.py").read_text(encoding="utf-8")
    assert 'dcp.load({"training":ModelOnlyState(model)}' in source
    assert "get_model_state_dict" in source
    assert "optimizer=torch.optim.AdamW" in source
