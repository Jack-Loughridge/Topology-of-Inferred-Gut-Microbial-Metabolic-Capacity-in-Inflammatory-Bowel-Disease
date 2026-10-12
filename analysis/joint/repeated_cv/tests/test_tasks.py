from joint_repeated_cv.tasks import TASK_ORDER, TASK_SPECS, normalise_label, ordered_classes


def test_requested_task_order_is_locked() -> None:
    assert TASK_ORDER == (
        "IBD_vs_nonIBD",
        "three_way_nonIBD_UC_CD",
        "nonIBD_vs_UC",
        "nonIBD_vs_CD",
        "CD_vs_UC",
    )
    assert TASK_SPECS[-1].report_name == "UC vs CD"


def test_label_normalisation_and_preferred_order() -> None:
    assert normalise_label("non-IBD") == "nonIBD"
    assert normalise_label("ulcerative colitis") == "UC"
    assert normalise_label("Crohn's disease") == "CD"
    classes = ordered_classes(["CD", "non-IBD", "UC"], TASK_SPECS[1])
    assert classes == ("nonIBD", "UC", "CD")
