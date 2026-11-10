"""Canonical label orders for the supported downstream classification tasks."""

PTBXL_SUPER_LABELS = ["HYP", "NORM", "MI", "CD", "STTC"]

PTBXL_SUB_LABELS = [
    "AMI", "LAFB/LPFB", "LVH", "STTC", "IMI", "SEHYP", "CRBBB", "WPW",
    "LAO/LAE", "NORM", "AVB", "RAO/RAE", "ISC", "LMI", "ISCI", "ISCA",
    "NST", "CLBBB", "ILBBB", "IRBBB", "PMI", "RVH", "IVCD",
]

PTBXL_FORM_LABELS = [
    "PAC", "DIG", "HVOLT", "STD", "LPR", "QWAVE", "VCLVH", "NDT", "TAB",
    "LOWT", "ABQRS", "LNGQT", "PRC(S)", "NT", "STE", "NST", "PVC", "INVT",
    "LVOLT",
]

PTBXL_RHYTHM_LABELS = [
    "SVARR", "BIGU", "STACH", "SARRH", "SBRAD", "TRIGU", "AFIB", "SR",
    "SVTAC", "PSVT", "AFLT", "PACE",
]

CPSC2018_LABELS = [
    "AFIB", "VPC", "NORM", "1AVB", "CRBBB", "STE", "PAC", "CLBBB", "STD",
]

CSN_LABELS = [
    "AQW", "UW", "SR", "WPW", "2AVB", "AT", "VB", "ARS", "STTC", "SA",
    "STE", "VPB", "TWO", "STTU", "ALS", "APB", "2AVB1", "PRIE", "CCR",
    "CR", "AF", "AVB", "QTIE", "LBBB", "VEB", "SVT", "RBBB", "1AVB",
    "STDD", "MI", "AFIB", "TWC", "PWC", "ERV", "RVH", "LVH", "ST", "JEB",
]

def expected_label_names(task_key: str):
    table = {
        "ptbxl_super_class": PTBXL_SUPER_LABELS,
        "ptbxl_sub_class": PTBXL_SUB_LABELS,
        "ptbxl_form": PTBXL_FORM_LABELS,
        "ptbxl_rhythm": PTBXL_RHYTHM_LABELS,
        "icbeb": CPSC2018_LABELS,
        "chapman": CSN_LABELS,
    }
    try:
        return list(table[task_key])
    except KeyError as error:
        raise KeyError(
            f"Unknown task {task_key!r}; expected one of {sorted(table)}"
        ) from error
