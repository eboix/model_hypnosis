"""Gemini flash steering CLI -- Google Generative Language generateContent (REST).

Produces Fig 9 (saturation) and Fig 11 (steer / measure) data for the Gemini
reasoning models. The reasoning knob maps --effort {low,high} to thinkingLevel
low/high (gemini-3+) or a thinkingBudget of 512/2048 (gemini-2.5-*). Thought
parts are stripped before parsing. Key via ``get_key('gemini')`` only.

  *** PAID ENDPOINT. *** Real calls require --confirm-paid; defaults are small.

  # Fig 9: base P(y+) heatmap across the main grid
  python -m mhyp.api_steering.gemini saturation --confirm-paid \
      --models gemini-3-flash-preview gemini-2.5-flash --ns 24

  # Fig 11: animals_consider x five7 flip prompts (item x position, high effort)
  python -m mhyp.api_steering.gemini steer --confirm-paid \
      --model gemini-3-flash-preview --family ip --pool animals_consider \
      --effect five7 --effort high --n 2500

  # Fig 11: held-out @100 validation of a saved fit's prompts
  python -m mhyp.api_steering.gemini measure --confirm-paid \
      --model gemini-3-flash-preview --fit <cell>/fit.json --effect five7 --effort high
"""
from mhyp import config
from mhyp.api_steering import client


def main():
    client.cli_main(
        "gemini",
        default_models=["gemini-3-flash-preview", "gemini-2.5-flash"],
        efforts=["low", "high"],
        default_steer_model="gemini-3-flash-preview",
        default_measure_model="gemini-3-flash-preview",
        sat_out=str(config.DATA / "reasoning_saturation_gemini.json"),
        measure_out=str(config.DATA / "gemini_measure.json"),
        doc=__doc__)


if __name__ == "__main__":
    main()
