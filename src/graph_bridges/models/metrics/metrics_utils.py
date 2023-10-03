import json
from pathlib import Path

def read_metric(config,metric_string_identifier,checkpoint=None):
    obtain_number = lambda x: int(x.name.split("_")[-1].split(".")[0]) if x.name.split("_")[-1].split(".")[0].isdigit() else None

    generic_metric_path_ = config.experiment_files.metrics_file.format(metric_string_identifier + "*")
    generic_metric_path_to_fill = config.experiment_files.metrics_file.format(metric_string_identifier + "_{0}")
    generic_metric_path_ = Path(generic_metric_path_)

    # avaliable numbers
    numbers_available = list(map(obtain_number,generic_metric_path_.parent.glob(generic_metric_path_.name)))

    metrics_ = {}
    #read check point
    if checkpoint is not None:
        if checkpoint in numbers_available:
            metric_path_ = Path(generic_metric_path_to_fill.format(checkpoint))
            if metric_path_.exists():
                metrics_ = json.load(open(metric_path_, "r"))

    if checkpoint is None:
        #read best file
        metric_path_ = Path(generic_metric_path_to_fill.format("best"))
        if metric_path_.exists():
            metrics_ = json.load(open(metric_path_,"r"))

        #read best available chackpoint
        else:
            if len(numbers_available) > 0:
                best_number = max(numbers_available)
                metric_path_ = Path(generic_metric_path_to_fill.format(best_number))
                if metric_path_.exists():
                    metrics_ = json.load(open(metric_path_, "r"))
        return metrics_