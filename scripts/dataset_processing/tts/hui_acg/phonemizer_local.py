"""
The original JSON splits generated from `get_data.py` only contain graphemes inputs. We recommend adding phonemes as
well to obtain better quality of synthesized audios. So you would expect the dataset double sized. This script implements
such idea. Before running, please install dependency on your local machine as shown below. More details are described in
https://github.com/NVIDIA/NeMo/blob/main/tutorials/tts/Fastpitch_Training_GermanTTS.ipynb

$ pip install phonemizer && apt-get install espeak-ng

Usage for example:
$ python scripts/dataset_processing/tts/hui_acg/phonemizer_local.py \
    --preserve-punctuation \
    --json-manifests ~/tmp/val_manifest_text_normed.json ~/tmp/test_manifest_text_normed.json
"""

import argparse
import json
from pathlib import Path

from phonemizer.backend import EspeakBackend
from tqdm import tqdm

from nemo.utils import logging


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
    parser.add_argument(
        "--preserve-punctuation",
        default=False,
        action='store_true',
        help="Preserve punctuations if True when converting char into phonemes.",
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    input_manifest_filepaths = args.json_manifests
    preserve_punctuation = args.preserve_punctuation
    backend = EspeakBackend(language='de', preserve_punctuation=preserve_punctuation)
    for manifest in input_manifest_filepaths:
        logging.info(f"Phonemizing: {manifest}")
        entries = []
        with open(manifest, 'r') as fjson:
            for line in tqdm(fjson):
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
