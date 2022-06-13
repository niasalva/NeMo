# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import csv
import json
import random
import shutil
import urllib.request
from pathlib import Path

from joblib import Parallel, delayed
from nemo_text_processing.text_normalization.normalize import Normalizer
from tqdm import tqdm

# full corpus.
URLS_FULL = {
    "Bernd_Ungerer": "https://opendata.iisys.de/systemintegration/Datasets/HUI-Audio-Corpus-German/dataset_full/Bernd_Ungerer.zip",
    "Eva_K": "https://opendata.iisys.de/systemintegration/Datasets/HUI-Audio-Corpus-German/dataset_full/Eva_K.zip",
    "Friedrich": "https://opendata.iisys.de/systemintegration/Datasets/HUI-Audio-Corpus-German/dataset_full/Friedrich.zip",
    "Hokuspokus": "https://opendata.iisys.de/systemintegration/Datasets/HUI-Audio-Corpus-German/dataset_full/Hokuspokus.zip",
    "Karlsson": "https://opendata.iisys.de/systemintegration/Datasets/HUI-Audio-Corpus-German/dataset_full/Karlsson.zip",
    "others": "https://opendata.iisys.de/systemintegration/Datasets/HUI-Audio-Corpus-German/dataset_clean/others.zip",
}
URL_STATS_FULL = "https://opendata.iisys.de/systemintegration/Datasets/HUI-Audio-Corpus-German/datasetStatistic.zip"

# the clean subset of the full corpus.
URLS_CLEAN = {
    "Bernd_Ungerer": "https://opendata.iisys.de/systemintegration/Datasets/HUI-Audio-Corpus-German/dataset_clean/Bernd_Ungerer_Clean.zip",
    "Eva_K": "https://opendata.iisys.de/systemintegration/Datasets/HUI-Audio-Corpus-German/dataset_clean/Eva_K_Clean.zip",
    "Friedrich": "https://opendata.iisys.de/systemintegration/Datasets/HUI-Audio-Corpus-German/dataset_clean/Friedrich_Clean.zip",
    "Hokuspokus": "https://opendata.iisys.de/systemintegration/Datasets/HUI-Audio-Corpus-German/dataset_clean/Hokuspokus_Clean.zip",
    "Karlsson": "https://opendata.iisys.de/systemintegration/Datasets/HUI-Audio-Corpus-German/dataset_clean/Karlsson_Clean.zip",
    "others": "https://opendata.iisys.de/systemintegration/Datasets/HUI-Audio-Corpus-German/dataset_clean/others.zip",
}
URL_STATS_CLEAN = (
    "https://opendata.iisys.de/systemintegration/Datasets/HUI-Audio-Corpus-German/datasetStatisticClean.zip"
)


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Download HUI-Audio-Corpus-German and create manifests with predefined split. "
        "Please check details about the corpus in https://github.com/iisys-hof/HUI-Audio-Corpus-German.",
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--set-type", default="clean", choices=["full", "clean"], type=str)
    parser.add_argument("--min-duration", default=0.1, type=float)
    parser.add_argument("--max-duration", default=15, type=float)
    parser.add_argument(
        "--num-workers",
        default=-1,
        type=int,
        help="Specify the max number of concurrently text normalization Python workers processes. "
        "If -1 all CPUs are used. If 1 no parallel computing is used. This argument is only "
        "triggered when --normalize-text is called.",
    )
    parser.add_argument(
        "--normalize-text",
        default=False,
        action='store_true',
        help="Normalize original text and add a new entry 'normalized_text' to .json file if True.",
    )
    parser.add_argument(
        "--val-num-uttrs-per-speaker",
        default=1,
        type=int,
        help="Specify the number of utterances for each speaker in val split. All speakers are covered.",
    )
    parser.add_argument(
        "--test-num-utters-per-speaker",
        default=1,
        type=int,
        help="Specify the number of utterances for each speaker in test split. All speakers are covered.",
    )
    parser.add_argument(
        "--seed-for-ds-split",
        default=100,
        type=float,
        help="Seed for deterministic split of train/dev/test, NVIDIA's default is 100",
    )
    args = parser.parse_args()
    return args


def __maybe_download_file(source_url, destination_path):
    if not destination_path.exists():
        print(f"Downloading data: {source_url} --> {destination_path}")
        tmp_file_path = destination_path.with_suffix(".tmp")
        urllib.request.urlretrieve(source_url, filename=tmp_file_path)
        tmp_file_path.rename(destination_path)
    else:
        print(f"Skipped downloading data because it exists: {destination_path}")


def __extract_file(filepath, data_dir):
    print(f"Unzipping data: {filepath} --> {data_dir}")
    try:
        shutil.unpack_archive(filepath, data_dir)
    except RuntimeError:
        print(f"Error while extracting {filepath}. Already extracted?")


def __save_json(json_file, dict_list):
    print(f"Saving json split to {json_file}.")
    with open(json_file, "w") as f:
        for d in dict_list:
            f.write(json.dumps(d) + "\n")


def __process_data(
    dataset_path,
    stat_path,
    speaker_id,
    min_duration,
    max_duration,
    val_size,
    test_size,
    seed_for_ds_split,
):
    print(f"Preparing JSON split for speaker {speaker_id}.")
    entries = []
    with open(stat_path, 'r') as f:
        # Let's skip the header
        f.readline()
        for line in tqdm(f):
            file_stem, duration, *_, text = line.strip().split("|")
            duration = float(duration)

            # file_stem -> dir_name (e.g. maerchen_01_f000051 -> maerchen)
            dir_name = "_".join(file_stem.split("_")[:-2])
            audio_path = dataset_path / dir_name / "wavs" / f"{file_stem}.wav"

            if min_duration <= duration <= max_duration:
                entry = {
                    "audio_filepath": str(audio_path),
                    "duration": duration,
                    "text": text,
                    "speaker": speaker_id,
                }
                entries.append(entry)

    random.Random(seed_for_ds_split).shuffle(entries)
    train_size = len(entries) - val_size - test_size
    assert train_size > 0, "Not enough data for train, val and test"
    print(f"Preparing JSON split for speaker {speaker_id} is complete.")

    return entries[:train_size], entries[train_size : train_size + val_size], entries[train_size + val_size :]


def __text_normalization(json_file, num_workers=-1):
    text_normalizer_call_kwargs = {
        "punct_pre_process": True,
        "punct_post_process": True,
    }
    text_normalizer = Normalizer(
        lang="de",
        input_case="cased",
        overwrite_cache=True,
        cache_dir=str(json_file.parent / "cache_dir"),
    )

    def normalizer_call(x):
        return text_normalizer.normalize(x, **text_normalizer_call_kwargs)

    def add_normalized_text(line_dict):
        normalized_text = normalizer_call(line_dict["text"])
        line_dict.update({"normalized_text": normalized_text})
        return line_dict

    print(f"Normalizing text for {json_file}.")
    with open(json_file, 'r', encoding='utf-8') as fjson:
        lines = fjson.readlines()
        dict_list = Parallel(n_jobs=num_workers)(
            delayed(add_normalized_text)(json.loads(line) for line in tqdm(lines))
        )

    json_file_text_normed = json_file.parent / f"{json_file.stem}_text_normed.{json_file.suffix}"
    with open(json_file_text_normed, 'w', encoding="utf-8") as fjson_norm:
        for dct in dict_list:
            fjson_norm.write(json.dumps(dct) + "\n")
    print(f"Normalizing text is complete: {json_file} --> {json_file_text_normed}")


def main():
    args = get_args()
    data_root = args.data_root
    set_type = args.set_type

    dataset_root = data_root / f"HUI-Audio-Corpus-German-{set_type}"
    dataset_root.mkdir(parents=True, exist_ok=True)

    if set_type == "full":
        data_source = URLS_FULL
        stats_source = URL_STATS_FULL
    elif set_type == "clean":
        data_source = URLS_CLEAN
        stats_source = URL_STATS_CLEAN
    else:
        raise ValueError(f"Unknown {set_type}. Please choose either clean or full.")

    # download datasets
    speakers = []
    for speaker in data_source:
        speakers.append(speaker)
        speaker_data_source = data_source[speaker]
        zipped_speaker_data_path = dataset_root / Path(speaker_data_source).name
        zipped_stats_path = dataset_root / Path(stats_source).name

        # TODO (xueyang): parallel downloading datasets to save time.
        __maybe_download_file(speaker_data_source, zipped_speaker_data_path)
        __maybe_download_file(stats_source, zipped_stats_path)

        __extract_file(zipped_speaker_data_path, dataset_root)
        __extract_file(zipped_stats_path, dataset_root)

    # generate json files for train/val/test splits
    stats_path_root = dataset_root / Path(stats_source).stem / "speacker"
    entries_train, entries_val, entries_test = [], [], []
    speaker_to_id = {}
    num_speakers = 0
    for child in stats_path_root.iterdir():
        if child.is_dir():
            speaker = child.name
            num_speakers += 1
            speaker_to_id[speaker] = num_speakers
            speaker_stats_path = stats_path_root / speaker / "overview.csv"

            speaker_data_source = data_source[speaker]
            speaker_data_path = dataset_root / Path(speaker_data_source).stem
            if speaker not in speakers:  # others
                speaker_data_path = speaker_data_path / speaker

            train, val, test = __process_data(
                speaker_data_path,
                speaker_stats_path,
                num_speakers,
                args.min_duration,
                args.max_duration,
                args.val_num_uttrs_per_speaker,
                args.test_num_uttrs_per_speaker,
                args.seed_for_ds_split,
            )

            entries_train.extend(train)
            entries_val.extend(val)
            entries_test.extend(test)

    # shuffle in place across multiple speakers
    random.Random(args.seed_for_ds_split).shuffle(entries_train)
    random.Random(args.seed_for_ds_split).shuffle(entries_val)
    random.Random(args.seed_for_ds_split).shuffle(entries_test)

    # save speaker to id.
    spk2id_file_path = dataset_root / "spk2id.csv"
    with open(spk2id_file_path, 'w') as fspk:
        fspk.write(f"speaker_name,speaker_id\n")
        for key, val in speaker_to_id.items():
            fspk.write(f"{key},{val}\n")

    # save json splits.
    train_json = dataset_root / "train_manifest.json"
    val_json = dataset_root / "val_manifest.json"
    test_json = dataset_root / "test_manifest.json"
    __save_json(train_json, entries_train)
    __save_json(val_json, entries_val)
    __save_json(test_json, entries_test)

    # normalize text if requested. New json file, train_manifest_text_normalized.json, will be generated.
    if args.normalize_text:
        __text_normalization(train_json, args.num_workers)
        __text_normalization(val_json, args.num_workers)
        __text_normalization(test_json, args.num_workers)


if __name__ == "__main__":
    main()
