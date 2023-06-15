from pathlib import Path

base_path = Path(__file__).resolve().parent
project_path = (base_path / ".." / "..").resolve()
data_path = project_path / "data"
results_path = project_path / "results"
plots_path = project_path / "plots"
config_path = project_path / "configs"
general_path = project_path / "general"
test_resources_path = project_path / "tests" / "resources"
