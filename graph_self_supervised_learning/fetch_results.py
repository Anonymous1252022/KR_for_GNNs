import argparse
import pandas as pd
import wandb
from datetime import datetime


def make_hyperlink(url, value):
    return '=HYPERLINK("%s", "%s")' % (url, value)


def build_skeleton():
    DEPTHS = range(0, 6)
    DATASETS = ["reddit2", "ogbn-arxiv", "ogbn-products", "flickr", "ppi", "pubmed", "reddit"]
    LAYERS = ["gcn", "graphconv", "gat", "graphsage"]

    ans = {}
    for depth in DEPTHS:
        for ds in DATASETS:
            for layer in LAYERS:
                for method in ("reconstruction", "supervised"):
                    key = (layer, ds, depth, method)
                    ans[key] = {"Status": "pending",
                                "Depth": depth,
                                "Layer": layer,
                                "Dataset": ds}
    return ans


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-name", type=str, required=True, help="The name of the W&B project (full path: <entity/project-name>) ")
    parser.add_argument("--output-csv", type=str, required=True, help="Where to store the results (CSV path)")

    args = parser.parse_args()
    proj_name = args.project_name
    output_path = args.output_csv
    if not output_path.endswith(".csv"):
        raise ValueError("Output CSV must end with '.csv'")

    api = wandb.Api()

    # Project is specified by <entity/project-name>
    runs = api.runs(proj_name)

    logged_summaries = build_skeleton()
    for run in runs:
        summary = {"Status": run.state}
        run_summary = {k: v for k, v in run.config.items()
                            if not k.startswith('_')}
        if len(run_summary) == 0:
            continue
        run_type = "reconstruction" if "use_self_in_loss" in run_summary["training"] else "supervised"
        layer_name = run_summary["layer"]["name"]
        ds_name = run_summary["dataset"]["name"]
        depth = run_summary["model"]["depth"]
        summary["Depth"] = depth
        summary["Layer"] = layer_name
        summary["Dataset"] = ds_name

        summary[f"URL_{run_type}"] = make_hyperlink(run.url, run.name)

        if run.state in ("finished", "running"):
            run_summary.update({k: v for k, v in run.summary._json_dict.items() if
                           isinstance(v, (int, float)) and not k.startswith("_")})
            try:
                train_acc = run_summary["best_accuracy_train/lvl_final"]
                val_acc = run_summary["best_accuracy_val/lvl_final"]
                test_acc = run_summary["best_accuracy_test/lvl_final"]
            except KeyError:
                print(f"Got index error in run: {run.name}")
                train_acc = val_acc = test_acc = None
        else:
            train_acc = val_acc = test_acc = None

        if train_acc is None:
            run_key = (layer_name, ds_name, depth, run_type)
            if run_key in logged_summaries and logged_summaries[run_key]["Status"] == "pending":
                logged_summaries[run_key]["Status"] = run.state
            continue

        ts = run.summary._json_dict["_timestamp"]
        dt = datetime.utcfromtimestamp(ts)

        if run_type == "reconstruction" and (dt.month != 5 or dt.day < 10) :
            continue

        run_key = (layer_name, ds_name, depth, run_type)
        if run_key not in logged_summaries:
            continue

        train_key = f"{run_type}_train"
        val_key = f"{run_type}_val"
        test_key = f"{run_type}_test"
        if run_key in logged_summaries and test_key in logged_summaries[run_key] and test_acc < logged_summaries[run_key][test_key]:
            continue

        summary[train_key] = train_acc
        summary[val_key] = val_acc
        summary[test_key] = test_acc
        logged_summaries[run_key] = summary

    # Merge all the runs from supervised to no supervised
    merged_logged_summaries = {}
    for k, v in logged_summaries.items():
        new_k = k[:3]
        new_v = v
        if new_k in merged_logged_summaries:
            new_v = merged_logged_summaries[new_k]
            new_v.update({kk: vv for kk, vv in v.items() if kk not in ("Status", "Depth", "Layer", "Dataset")})
        merged_logged_summaries[new_k] = new_v

    df = pd.DataFrame.from_dict(list(merged_logged_summaries.values()))
    df = df.sort_values(["Dataset", "Layer", "Depth"], ascending = (True, True, True))
    df.to_csv(output_path)