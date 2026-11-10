"""Cross-domain groups with target truth OR and source score aggregation."""
import json
import numpy as np

def load_scheme2(path):
    with open(path, encoding="utf-8") as f:
        result = json.load(f)
    if result.get("schema_version") != 2:
        raise ValueError(f"{path}: cross-domain mappings require schema_version 2")
    return result

def grouped_truth(frame, groups):
    covered = sorted({x for xs in groups.values() for x in xs})
    kept = frame.loc[frame[covered].sum(axis=1) > 0].copy()
    for label, members in groups.items():
        kept["zs_label__" + label] = kept[members].max(axis=1)
    return kept.reset_index(drop=True)

def aggregate_scores(scores, labels, pair):
    columns = {label: i for i, label in enumerate(labels)}
    outputs = []
    for label in pair["target_truth_groups"]:
        values = scores[:, [columns[x] for x in pair["source_prediction_groups"][label]]]
        method = pair["aggregation"][label]
        if method not in {"mean", "max"}:
            raise ValueError(method)
        outputs.append(values.mean(axis=1) if method == "mean" else values.max(axis=1))
    return np.column_stack(outputs)
