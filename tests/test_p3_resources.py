from src.core.resource_monitor import ResourceMonitor


def test_resource_snapshot():
    snapshot = ResourceMonitor.snapshot(".")
    assert "cpu" in snapshot
    assert "memory" in snapshot
    assert "disk" in snapshot
    assert snapshot["memory"]["total_gb"] > 0
    assert snapshot["disk"]["total_gb"] > 0


def test_resource_healthy():
    snapshot = ResourceMonitor.snapshot(".")
    healthy, errors = ResourceMonitor.healthy(snapshot, minimum_free_disk_gb=0.1)
    assert healthy is True
    assert len(errors) == 0
