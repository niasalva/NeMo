"""
The original JSON splits generated from `get_data.py` only contain graphemes inputs. We recommend adding phonemes as
well to obtain better quality of synthesized audios. So you would expect the dataset double sized. This script implements
such idea. Before running, please install dependency on your local machine as shown below. More details are described in
https://github.com/NVIDIA/NeMo/blob/main/tutorials/tts/Fastpitch_Training_GermanTTS.ipynb

$ pip install phonemizer && apt-get install espeak-ng
"""

import argparse
import json
from pathlib import Path

from phonemizer.backend import EspeakBackend

from nemo.utils import logging

backend = EspeakBackend('de')


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Add phonemes in JSON manifests."
    )
    parser.add_argument(
        "--json-manifests",
        nargs="+",
        type=Path,
        help="Specify a full path of a JSON manifest. You could add multiple manifest.",
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    input_manifest_filepaths = args.json_manifests

    for manifest in input_manifest_filepaths:
        logging.info(f"Phonemizing: {manifest}")
        entries = []
        with open(manifest, 'r') as fjson:
            for line in fjson:
                # grapheme
                grapheme_dct = json.loads(line.strip())
                grapheme_dct.update({"is_phoneme": 0})
                # phoneme
                phoneme_dct = grapheme_dct.copy()
                phonemes = backend.phonemize([grapheme_dct["normalized_text"]])
                phoneme_dct["normalized_text"] = phonemes[0]
                phoneme_dct["is_phoneme"] = 1

                entries.append(grapheme_dct)
                entries.append(phoneme_dct)

        output_manifest_filepath = manifest.parent / f"{manifest.stem}_phonemes{manifest.suffix}"
        with open(output_manifest_filepath, "w", encoding="utf-8") as fout:
            for entry in entries:
                fout.write(f"{json.dumps(entry)}\n")
        logging.info(f"Phonemizing is complete: {manifest} --> {output_manifest_filepath}")


if __name__ == "__main__":
    main()
