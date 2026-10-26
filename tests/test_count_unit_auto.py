"""bamCoverageR --countUnit auto: read for plain runs, fragment with -e/--collapse."""

import json

import pytest

from deeptoolsr import bamCoverage, config, deeptoolsr_list_tools
from tests.test_count_unit import MAPQ, bam  # noqa: F401  (module fixture)

pytest.importorskip("pysam")

MODELS = {
    "plain": ([], "read"),
    "extend": (["-e", "60"], "fragment"),
    "5prime": (["--collapse", "5prime"], "fragment"),
    "3prime": (["--collapse", "3prime"], "fragment"),
    "center": (["--collapse", "center"], "fragment"),
}
METRICS = ["read-count", "coverage-mean", "CPM", "BPM"]


def outputs(tmp_path, bam, name, *options):  # noqa: F811
    out = tmp_path / name
    out.mkdir()
    bamCoverage.main(["-b", str(bam), "-o", str(out / "cov.bedgraph"), "-of", "bedgraph",
                      "-bs", "10", "--minMappingQuality", str(MAPQ), *options])
    return {p.name: p.read_bytes() for p in sorted(out.iterdir())}


@pytest.mark.parametrize("threads", ["1", "4"])
@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("model", list(MODELS))
def test_auto_equals_resolved_unit(tmp_path, bam, model, metric, split, threads):  # noqa: F811
    flags, unit = MODELS[model]
    common = [*flags, "--normalizeUsing", metric, "-p", threads]
    if split:
        common += ["--strandedness", "reverse", "--filterRNAstrand", "split"]
    expected = outputs(tmp_path, bam, "explicit", *common, "--countUnit", unit)
    assert outputs(tmp_path, bam, "default", *common) == expected
    assert outputs(tmp_path, bam, "auto", *common, "--countUnit", "auto") == expected
    other = "fragment" if unit == "read" else "read"
    assert outputs(tmp_path, bam, "other", *common, "--countUnit", other) != expected


def store(value):
    config.options_path().parent.mkdir(parents=True, exist_ok=True)
    config.options_path().write_text(json.dumps({"count_unit": value}))


@pytest.mark.parametrize("model", ["plain", "extend"])
@pytest.mark.parametrize("stored", ["auto", "read", "fragment"])
def test_stored_count_unit(tmp_path, bam, capsys, stored, model):  # noqa: F811
    flags, auto_unit = MODELS[model]
    deeptoolsr_list_tools.process_args(["options", "--countUnit", stored])
    assert "bamCoverageR count unit: {}".format(stored) in capsys.readouterr().out
    assert json.loads(config.options_path().read_text())["count_unit"] == stored
    unit = auto_unit if stored == "auto" else stored
    expected = outputs(tmp_path, bam, "explicit", *flags, "--countUnit", unit)
    assert outputs(tmp_path, bam, "stored", *flags) == expected
    for cli in ("read", "fragment", "auto"):
        want = auto_unit if cli == "auto" else cli
        assert (outputs(tmp_path, bam, "cli-" + cli, *flags, "--countUnit", cli)
                == outputs(tmp_path, bam, "ref-" + cli, *flags, "--countUnit", want))


def test_default_options_show_auto(capsys):
    deeptoolsr_list_tools.process_args(["options"])
    assert "bamCoverageR count unit: auto" in capsys.readouterr().out


@pytest.mark.parametrize("model", ["plain", "extend"])
def test_invalid_stored_count_unit_falls_back_to_auto(tmp_path, bam, capsys, model):  # noqa: F811
    flags, unit = MODELS[model]
    expected = outputs(tmp_path, bam, "explicit", *flags, "--countUnit", unit)
    store("pairs")
    capsys.readouterr()
    assert outputs(tmp_path, bam, "stored", *flags) == expected
    assert "ignoring invalid count_unit; using auto" in capsys.readouterr().err
