import pytest
from datasets import Dataset

from fastdetector import utils as utils_module
from fastdetector.frontend.toml_config import ConditionConfig
from fastdetector.utils import (
    apply_filter_conditions,
    load_dataset_all_shards,
    load_dataset_auto_shard,
    shard_config_name,
    upload_readme,
)


@pytest.fixture
def fake_hub(monkeypatch):
    """Replace the HF Hub calls used by the dataset loaders.

    Returns:
        A recorder object; set ``configs`` (or ``configs_error``) and inspect
        ``loads`` for the ``(name, config_name, split)`` of each load.
    """

    class Hub:
        """Stub dataset Hub recorder tracking configs and load calls."""

        configs: list[str] = []
        configs_error: Exception | None = None
        loads: list[tuple] = []
        contents: dict[str, list[dict]] = {}

    hub = Hub()
    hub.loads = []

    def fake_get_config_names(dataset_name):
        if hub.configs_error is not None:
            raise hub.configs_error
        return list(hub.configs)

    def fake_load_dataset(dataset_name, name=None, split="train"):
        hub.loads.append((dataset_name, name, split))
        rows = hub.contents.get(name or "__default__", [{"id": name or "default"}])
        return Dataset.from_list(rows)

    monkeypatch.setattr(utils_module, "get_dataset_config_names", fake_get_config_names)
    monkeypatch.setattr(utils_module, "load_dataset", fake_load_dataset)
    return hub


# --------------------------------------------------------------------------
# shard_config_name
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("index", "expected"), [(0, "shard_0"), (7, "shard_7")])
def test_shard_config_name(index, expected):
    assert shard_config_name(index) == expected


def test_shard_config_name_is_the_single_source_of_truth(repo_root):
    # The numbered gen configs are named after the shard they produce; if the
    # two ever disagree, a run reads one shard and writes another.
    expected_counts = {"train": 14, "val": 6, "test": 2}
    for dataset_kind, count in expected_counts.items():
        names = [p.stem for p in (repo_root / "config" / "gen" / dataset_kind).glob("shard_*.toml")]
        assert sorted(names, key=lambda n: int(n.split("_")[1])) == [
            shard_config_name(index) for index in range(count)
        ]


# --------------------------------------------------------------------------
# load_dataset_auto_shard
# --------------------------------------------------------------------------


def test_auto_shard_resolves_the_matching_config(fake_hub):
    fake_hub.configs = ["shard_0", "shard_1", "shard_2"]
    load_dataset_auto_shard("user/ds", subset_index=1)
    assert fake_hub.loads == [("user/ds", "shard_1", "train")]


def test_auto_shard_honours_the_split(fake_hub):
    fake_hub.configs = ["shard_0"]
    load_dataset_auto_shard("user/ds", split="validation", subset_index=0)
    assert fake_hub.loads[0][2] == "validation"


def test_auto_shard_raises_rather_than_reading_the_wrong_shard(fake_hub):
    # Falling back to some other shard would duplicate work and corrupt data.
    fake_hub.configs = ["shard_0", "shard_1"]
    with pytest.raises(ValueError, match="no config named 'shard_5'"):
        load_dataset_auto_shard("user/ds", subset_index=5)


def test_auto_shard_error_lists_the_available_configs(fake_hub):
    fake_hub.configs = ["shard_0", "other"]
    with pytest.raises(ValueError, match=r"\['other', 'shard_0'\]"):
        load_dataset_auto_shard("user/ds", subset_index=9)


def test_auto_shard_accepts_an_unsharded_dataset_at_index_zero(fake_hub):
    fake_hub.configs = ["default"]
    load_dataset_auto_shard("user/ds", subset_index=0)
    assert fake_hub.loads == [("user/ds", "default", "train")]


def test_a_single_config_does_not_satisfy_a_non_zero_shard(fake_hub):
    fake_hub.configs = ["default"]
    with pytest.raises(ValueError, match="no config named 'shard_1'"):
        load_dataset_auto_shard("user/ds", subset_index=1)


def test_auto_shard_falls_back_to_the_default_config_when_listing_fails(fake_hub):
    fake_hub.configs_error = ConnectionError("hub unreachable")
    load_dataset_auto_shard("user/ds", subset_index=0)
    assert fake_hub.loads == [("user/ds", None, "train")]


def test_auto_shard_with_no_configs_loads_the_default(fake_hub):
    fake_hub.configs = []
    load_dataset_auto_shard("user/ds", subset_index=2)
    assert fake_hub.loads == [("user/ds", None, "train")]


def test_auto_shard_with_a_none_index_skips_resolution_entirely(fake_hub):
    fake_hub.configs = ["shard_0", "shard_1"]
    load_dataset_auto_shard("user/ds", subset_index=None)
    assert fake_hub.loads == [("user/ds", None, "train")]


# --------------------------------------------------------------------------
# load_dataset_all_shards
# --------------------------------------------------------------------------


def test_all_shards_concatenates_every_config(fake_hub):
    fake_hub.configs = ["shard_0", "shard_1"]
    fake_hub.contents = {
        "shard_0": [{"id": 0}, {"id": 1}],
        "shard_1": [{"id": 2}],
    }
    dataset = load_dataset_all_shards("user/ds")
    assert dataset["id"] == [0, 1, 2]


def test_all_shards_with_one_config_returns_it_directly(fake_hub):
    fake_hub.configs = ["only"]
    fake_hub.contents = {"only": [{"id": 5}]}
    assert load_dataset_all_shards("user/ds")["id"] == [5]


def test_all_shards_falls_back_to_the_default_config(fake_hub):
    fake_hub.configs = []
    load_dataset_all_shards("user/ds")
    assert fake_hub.loads == [("user/ds", None, "train")]


def test_all_shards_survives_a_config_listing_failure(fake_hub):
    fake_hub.configs_error = ConnectionError("hub unreachable")
    load_dataset_all_shards("user/ds")
    assert fake_hub.loads == [("user/ds", None, "train")]


# --------------------------------------------------------------------------
# upload_readme
# --------------------------------------------------------------------------


@pytest.fixture
def fake_api(monkeypatch, tmp_path):
    """Stub HfApi and hf_hub_download.

    Returns:
        A recorder with ``uploads`` (path_in_repo -> raw bytes), ``commits``
        (how many commits were made), ``remote`` (repo_id -> README text) and
        a ``fail_upload`` switch.
    """

    class Api:
        """Recorder tracking uploaded files and remote README contents."""

        uploads: dict[str, bytes] = {}
        remote: dict[str, str] = {}
        commits = 0
        fail_upload = False

        @property
        def readme(self) -> str:
            """The uploaded README decoded as text."""
            return self.uploads["README.md"].decode("utf-8")

    api = Api()
    api.uploads = {}
    api.remote = {}

    class FakeHfApi:
        """Mock HfApi client recording the operations of each commit."""

        def create_commit(self, repo_id, repo_type, operations, commit_message=None):
            if api.fail_upload:
                raise OSError("upload failed")
            api.commits += 1
            for operation in operations:
                api.uploads[operation.path_in_repo] = operation.path_or_fileobj

    def fake_download(repo_id, filename, repo_type):
        if repo_id not in api.remote:
            raise FileNotFoundError(f"{repo_id} has no {filename}")
        path = tmp_path / f"{repo_id.replace('/', '_')}_{filename}"
        path.write_text(api.remote[repo_id], encoding="utf-8")
        return str(path)

    monkeypatch.setattr(utils_module, "HfApi", FakeHfApi)
    monkeypatch.setattr(utils_module, "hf_hub_download", fake_download)
    return api


def test_upload_readme_writes_the_content(fake_api):
    upload_readme("user/ds", readme_content="# Title\nbody\n")
    assert fake_api.readme == "# Title\nbody\n"


def test_upload_readme_preserves_the_existing_yaml_header(fake_api):
    # The YAML block carries the dataset's config/split declarations; dropping
    # it un-registers every shard on the Hub.
    fake_api.remote["user/ds"] = "---\nconfigs:\n- config_name: shard_0\n---\n\nold body\n"
    upload_readme("user/ds", readme_content="# New\n")
    uploaded = fake_api.readme
    assert uploaded.startswith("---\nconfigs:\n- config_name: shard_0\n---\n")
    assert uploaded.endswith("# New\n")
    assert "old body" not in uploaded


def test_upload_readme_without_an_existing_readme(fake_api):
    upload_readme("user/ds", readme_content="# New\n")
    assert fake_api.readme == "# New\n"


def test_upload_readme_appends_a_previous_readme_body(fake_api):
    fake_api.remote["user/previous"] = "---\nkey: value\n---\n\n## Previous\ntext"
    upload_readme(
        "user/ds", readme_content="## New\n", append_readme_source="user/previous"
    )
    uploaded = fake_api.readme
    assert "## Previous" in uploaded
    assert uploaded.index("## Previous") < uploaded.index("## New")
    # The source's YAML header must not be carried over into the body.
    assert "key: value" not in uploaded


def test_upload_readme_separates_the_appended_sections(fake_api):
    fake_api.remote["user/previous"] = "## Previous\ntext"
    upload_readme(
        "user/ds", readme_content="## New\n", append_readme_source="user/previous"
    )
    assert "text\n\n## New" in fake_api.readme


def test_upload_readme_survives_a_missing_append_source(fake_api):
    upload_readme(
        "user/ds", readme_content="## New\n", append_readme_source="user/missing"
    )
    assert fake_api.readme == "## New\n"


def test_upload_readme_uploads_extra_files(fake_api):
    upload_readme(
        "user/ds",
        readme_content="# T\n",
        files={"chart.png": b"\x89PNG", "summary.json": b"{}"},
    )
    assert set(fake_api.uploads) == {"README.md", "chart.png", "summary.json"}


def test_upload_readme_uses_one_commit_regardless_of_file_count(fake_api):
    # A commit per file used to trip the Hub's per-repo hourly commit limit
    # (128), aborting an analysis run partway and leaving the README pointing
    # at charts that never uploaded.
    files = {f"chart_{i}.png": b"\x89PNG" for i in range(200)}
    upload_readme("user/ds", readme_content="# T\n", files=files)
    assert fake_api.commits == 1
    assert len(fake_api.uploads) == len(files) + 1


def test_upload_readme_raises_when_the_upload_fails(fake_api):
    # A printed warning plus exit code 0 made lost READMEs look like success.
    fake_api.fail_upload = True
    with pytest.raises(RuntimeError, match="Failed to upload README"):
        upload_readme("user/ds", readme_content="# T\n")


def test_upload_readme_handles_a_yaml_header_with_no_closing_marker(fake_api):
    fake_api.remote["user/ds"] = "---\nunterminated header\n"
    upload_readme("user/ds", readme_content="# New\n")
    assert fake_api.readme == "# New\n"


# --------------------------------------------------------------------------
# apply_filter_conditions
# --------------------------------------------------------------------------


def condition(column: str, operator: str, value) -> ConditionConfig:
    """Build a ConditionConfig."""
    return ConditionConfig(column=column, operator=operator, value=value)


@pytest.fixture
def sample_dataset() -> Dataset:
    """A small dataset covering the types the filters have to handle."""
    return Dataset.from_list(
        [
            {"score": 0.1, "flag": True, "name": "alpha", "maybe": 1.0},
            {"score": 0.5, "flag": False, "name": "beta", "maybe": None},
            {"score": 0.9, "flag": True, "name": "gamma", "maybe": 3.0},
        ]
    )


def test_no_conditions_returns_the_dataset_untouched(sample_dataset):
    assert apply_filter_conditions(sample_dataset, []) is sample_dataset


@pytest.mark.parametrize(
    ("operator", "value", "expected"),
    [
        (">", 0.4, ["beta", "gamma"]),
        (">=", 0.5, ["beta", "gamma"]),
        ("<", 0.5, ["alpha"]),
        ("<=", 0.5, ["alpha", "beta"]),
        ("==", 0.5, ["beta"]),
        ("!=", 0.5, ["alpha", "gamma"]),
    ],
)
def test_numeric_operators(sample_dataset, operator, value, expected):
    filtered = apply_filter_conditions(sample_dataset, [condition("score", operator, value)])
    assert filtered["name"] == expected


def test_equality_on_booleans(sample_dataset):
    filtered = apply_filter_conditions(sample_dataset, [condition("flag", "==", True)])
    assert filtered["name"] == ["alpha", "gamma"]


def test_equality_on_strings(sample_dataset):
    filtered = apply_filter_conditions(sample_dataset, [condition("name", "==", "beta")])
    assert filtered["score"] == [0.5]


def test_and_requires_every_condition(sample_dataset):
    filtered = apply_filter_conditions(
        sample_dataset,
        [condition("score", ">", 0.2), condition("flag", "==", True)],
        filter_type="AND",
    )
    assert filtered["name"] == ["gamma"]


def test_or_requires_any_condition(sample_dataset):
    filtered = apply_filter_conditions(
        sample_dataset,
        [condition("score", ">", 0.8), condition("name", "==", "alpha")],
        filter_type="OR",
    )
    assert filtered["name"] == ["alpha", "gamma"]


def test_filter_type_is_case_insensitive(sample_dataset):
    filtered = apply_filter_conditions(
        sample_dataset,
        [condition("score", ">", 0.8), condition("name", "==", "alpha")],
        filter_type="or",
    )
    assert len(filtered) == 2


def test_unknown_filter_type_is_rejected(sample_dataset):
    with pytest.raises(ValueError, match='must be "AND" or "OR"'):
        apply_filter_conditions(
            sample_dataset, [condition("score", ">", 0.1)], filter_type="XOR"
        )


def test_a_missing_column_filters_the_row_out(sample_dataset):
    filtered = apply_filter_conditions(
        sample_dataset, [condition("not_a_column", "==", 1)]
    )
    assert len(filtered) == 0


def test_a_null_cell_filters_the_row_out(sample_dataset):
    filtered = apply_filter_conditions(sample_dataset, [condition("maybe", ">", 0.0)])
    assert filtered["name"] == ["alpha", "gamma"]


def test_a_non_numeric_comparison_filters_the_row_out(sample_dataset):
    filtered = apply_filter_conditions(sample_dataset, [condition("name", ">", 1)])
    assert len(filtered) == 0


def test_an_unknown_operator_filters_everything_out(sample_dataset):
    filtered = apply_filter_conditions(sample_dataset, [condition("score", "~=", 0.5)])
    assert len(filtered) == 0


def test_numeric_strings_are_coerced_for_numeric_operators():
    dataset = Dataset.from_list([{"n": "10"}, {"n": "2"}])
    filtered = apply_filter_conditions(dataset, [condition("n", ">", 5)])
    assert filtered["n"] == ["10"]


# --------------------------------------------------------------------------
# trashed-data configs must never be mistaken for a shard
# --------------------------------------------------------------------------


def test_a_trashed_config_alongside_a_shard_is_ignored(fake_hub):
    fake_hub.configs = ["shard_0", "shard_0_trashed_data", "shard_1", "shard_1_trashed_data"]
    load_dataset_auto_shard("user/ds", subset_index=1)
    assert fake_hub.loads == [("user/ds", "shard_1", "train")]


def test_the_default_batch_id_still_lands_on_shard_zero(fake_hub):
    # gen.py defaults --batch-id to 0, so the no-shard-specified path is this one.
    fake_hub.configs = ["shard_0", "shard_0_trashed_data"]
    load_dataset_auto_shard("user/ds", subset_index=0)
    assert fake_hub.loads == [("user/ds", "shard_0", "train")]


def test_a_lone_trashed_config_is_not_silently_loaded_as_shard_zero(fake_hub):
    # The single-config fallback must not fire for a trashed-only repo.
    fake_hub.configs = ["shard_0_trashed_data"]
    with pytest.raises(ValueError, match="no config named 'shard_0'"):
        load_dataset_auto_shard("user/ds", subset_index=0)


def test_a_lone_mismatched_shard_config_is_not_loaded_as_shard_zero(fake_hub):
    fake_hub.configs = ["shard_5"]
    with pytest.raises(ValueError, match="no config named 'shard_0'"):
        load_dataset_auto_shard("user/ds", subset_index=0)
