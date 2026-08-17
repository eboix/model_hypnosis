"""GPT-5.6 (sol / terra) steering CLI -- OpenAI chat.completions, reasoning_effort.

Produces Fig 9 (saturation) and Fig 11 (steer / measure) data for the OpenAI
reasoning models. Key via ``get_key('openai')`` only (never a home file).

  *** PAID ENDPOINT. *** Every subcommand bills OpenAI per sampled answer and
  real calls require --confirm-paid; the K/n defaults are deliberately small.
  Raise --n and the K budgets to the Table 1 values for a publication cell.

  # Fig 9: base P(y+) heatmap across the main grid (one sample per prompt)
  python -m mhyp.api_steering.openai_gpt56 saturation --confirm-paid \
      --models gpt-5.6-sol gpt-5.6-terra --ns 24 --effort low

  # Fig 11: verbprimes x trolley_flip flip prompts on sol (item x position)
  python -m mhyp.api_steering.openai_gpt56 steer --confirm-paid \
      --model gpt-5.6-sol --family ip --pool verbprimes --effect trolley_flip \
      --effort medium --n 2500 --k-cand 10

  # Fig 11: gpt-5.6-terra typos x trolley_yn flip prompts (bank family)
  python -m mhyp.api_steering.openai_gpt56 steer --confirm-paid \
      --model gpt-5.6-terra --family bank --bank typos --effect trolley_yn

  # Fig 11: re-measure a saved fit's top/bot prompts @100 fresh held-out
  python -m mhyp.api_steering.openai_gpt56 measure --confirm-paid \
      --model gpt-5.6-terra --fit <cell>/fit.json --effect trolley_yn
"""
from mhyp import config
from mhyp.api_steering import client


def main():
    client.cli_main(
        "openai",
        default_models=["gpt-5.6-sol", "gpt-5.6-terra"],
        efforts=["low", "medium", "high"],
        default_steer_model="gpt-5.6-sol",
        default_measure_model="gpt-5.6-terra",
        sat_out=str(config.DATA / "gpt56_saturation_maingrid.json"),
        measure_out=str(config.DATA / "gpt56_measure.json"),
        doc=__doc__)


if __name__ == "__main__":
    main()
