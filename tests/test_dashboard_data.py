from driftguard.dashboard_data import load_alerts, load_batches, short_name
from driftguard.storage import create_tables, get_engine, save_batch


def test_short_name_handles_windows_and_linux_paths():
    assert short_name(r"C:\Users\me\driftguard\data\batches\batch_001.csv") == "batch_001"
    assert short_name("data/batches/batch_002.csv") == "batch_002"


def test_dashboard_reads_what_the_pipeline_saved(tmp_path):
    engine = get_engine(f"sqlite:///{tmp_path / 'test.db'}")
    create_tables(engine)
    batch = {"batch_id": "abc123", "source_file": "data/batches/batch_009.csv", "row_count": 100,
             "valid_count": 90, "rejected_count": 10, "status": "passed_with_rejections"}
    alert = [{"category": "quality", "severity": "warning", "message": "10 rows rejected"}]
    evaluation = {"model_version": "v1", "n_labelled": 90, "accuracy": 0.8, "precision": 0.7,
                  "recall": 0.6, "f1": 0.65, "roc_auc": 0.9, "pr_auc": 0.75}
    save_batch(engine, batch, [], [], alert, evaluation)

    batches = load_batches(engine)
    assert batches.loc[0, "batch"] == "batch_009"
    assert batches.loc[0, "n_alerts"] == 1
    assert batches.loc[0, "rejection_rate"] == 0.1
    assert batches.loc[0, "roc_auc"] == 0.9

    alerts = load_alerts(engine)
    assert alerts.loc[0, "next_step"].startswith("Open data/rejected/")