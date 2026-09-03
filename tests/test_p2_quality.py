import json

from src.consistency.quality_report import (
    IdentityQualityReport,
)


def test_quality_report(tmp_path):

    report = IdentityQualityReport()

    report.add(
        score=0.90,
        warning=False,
        severe=False,
    )

    report.add(
        score=0.50,
        warning=True,
        severe=False,
    )

    report.retry_count = 1

    path = (
        tmp_path
        / "identity_quality.json"
    )

    report.save(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        data = json.load(handle)

    assert data["evaluated_frames"] == 2
    assert data["warning_frames"] == 1
    assert data["retry_count"] == 1
