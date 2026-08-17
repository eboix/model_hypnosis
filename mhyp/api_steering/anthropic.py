"""Claude steering CLI -- https://api.anthropic.com/v1/messages (REST).

Produces Fig 9 (saturation) and Fig 11 (steer / measure) data for the Anthropic
reasoning models. For claude-sonnet-5, --effort {low,medium,high} sets
output_config.effort under adaptive thinking; for claude-haiku-4-5 (pre-4.6, no
effort param) it scales thinking.budget_tokens (1024 low / 4096 otherwise).
Key via ``get_key('anthropic')`` only.

  *** PAID ENDPOINT. *** Real calls require --confirm-paid; defaults are small.

  # Fig 9: base P(y+) heatmap across the main grid
  python -m mhyp.api_steering.anthropic saturation --confirm-paid \
      --models claude-haiku-4-5 claude-sonnet-5 --ns 24

  # Fig 11: animals_consider x conscious flip prompts on sonnet-5 (item presence)
  python -m mhyp.api_steering.anthropic steer --confirm-paid \
      --model claude-sonnet-5 --family presence --pool animals_consider \
      --effect conscious --effort low --n 2500

  # Fig 11: effort-transfer -- re-measure a low-effort fit's prompts at higher effort
  python -m mhyp.api_steering.anthropic measure --confirm-paid \
      --model claude-sonnet-5 --fit <cell>/fit.json --effect conscious --effort high
"""
from mhyp import config
from mhyp.api_steering import client


def main():
    client.cli_main(
        "anthropic",
        default_models=["claude-haiku-4-5", "claude-sonnet-5"],
        efforts=["low", "medium", "high"],
        default_steer_model="claude-sonnet-5",
        default_measure_model="claude-sonnet-5",
        sat_out=str(config.DATA / "reasoning_saturation_anthropic.json"),
        measure_out=str(config.DATA / "anthropic_measure.json"),
        doc=__doc__)


if __name__ == "__main__":
    main()
