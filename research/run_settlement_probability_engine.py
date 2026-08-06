"""Run the Settlement Probability Engine audit."""
import json

from spe_data import build_feature_panel
from spe_models import train_models
from spe_eval import make_report, select_validation_policy


def main() -> None:
    panel = build_feature_panel()
    predictions, probability, chosen_c = train_models(panel)
    winner, executable = select_validation_policy(predictions)
    summary = make_report(panel, probability, chosen_c, winner, executable)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
