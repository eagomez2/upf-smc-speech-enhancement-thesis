import os
import glob
import argparse
import pesq
import torch
import time
import numpy as np
import pandas as pd
import soundfile as sf
import utils.evaluation_process as ep
from tqdm import tqdm
from datetime import datetime


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# command line arguments
parser = argparse.ArgumentParser(description="""
        Deep Noise Suppression Model Score Evaluation
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

# positional arguments
parser.add_argument("ref_dir", metavar="reference_dir",
                    type=str,
                    help="""
                    reference directory containing clean speech files
                    """)

parser.add_argument("est_dir", metavar="estimates_dir",
                    type=str,
                    help="""
                    estimates directory containing predicted speech files
                    """)

# optional arguments
parser.add_argument("-b", "--buffer_size",
                    metavar="buffer_size",
                    type=int,
                    default=25,
                    help="""
                    entries are written to a file every buffer_size iterations
                    """)

# Interim solution to avoid cloud computing problems with DNSMOS
parser.add_argument("-f", "--fix",
                    metavar="filepath",
                    dest="file_to_fix",
                    type=str,
                    help="""
                    resumes and completes the computations of a given file
                    """)

parser.add_argument("-m", "--metrics",
                    metavar="metrics_list",
                    nargs="*",
                    default=["stoi", "si-sdr", "pesq", "warpq"],
                    help="""
                    metrics to be computed
                    """)

parser.add_argument("-p", "--prefix",
                    metavar='report_prefix',
                    type=str,
                    default='',
                    help="""
                    prefix to be added to the report files
                    """)

parser.add_argument("-s", "--sample_rate",
                    metavar="sample_rate",
                    type=int,
                    default=16000,
                    help="""
                    assumed sample rate of the evaluated files
                    """)

parser.add_argument("-u", "--sample_duration",
                    metavar="duration",
                    type=int,
                    default=10,
                    help="""
                    assumed duration of each file (it will be cut if longer)
                    """)

args = parser.parse_args()

if __name__ == "__main__":

    if args.file_to_fix is not None and not os.path.isfile(args.file_to_fix):
        raise FileNotFoundError(f"File to fix not found: {args.file_to_fix}")

    # check if directories exist
    if not os.path.isdir(args.ref_dir):
        raise FileNotFoundError(
                f"Reference directory not found: {args.ref_dir}"
                )

    if not os.path.isdir(args.est_dir):
        raise FileNotFoundError(
                f"Estimates directory not found: {args.est_dir}"
                )

    # fill out metrics dict
    metrics_dict = {}

    if args.file_to_fix is not None:
        # extract metrics to be calculated
        metrics_data = pd.read_csv(args.file_to_fix)
        metrics_list = metrics_data.columns.tolist()

        # remove columns that do no corresponds to a metric
        metrics_list.remove("est_speech")
        metrics_list.remove("ref_speech")

        # get report filename
        report_filename = os.path.abspath(args.file_to_fix)
    else:
        # get metrics list
        metrics_list = args.metrics

        # create file with column names
        report_filename = f"{args.prefix}{os.path.basename(args.est_dir)}" \
                           "_eval.csv"

        # full path of filename
        report_file = os.path.abspath(os.path.join("reports", report_filename))

        # check no file is being overwritten
        if os.path.isfile(report_file):
            raise RuntimeError(f"Report file already exists: {report_file}")

        # save empty file just with column names
        df = pd.DataFrame(columns=metrics_list + ["ref_speech", "est_speech"])
        df.to_csv(report_file, index=False, mode="w", header=True)

    for metric in metrics_list:
        metrics_dict[metric] = []

    # placeholder for reference and estimated speech
    metrics_dict["ref_speech"] = []
    metrics_dict["est_speech"] = []

    # get file list for both reference and estimated files
    ref_files = ep.collect_wav_files(args.ref_dir)
    est_files = ep.collect_wav_files(args.est_dir)

    # sanity check
    if len(ref_files) != len(est_files): 
        raise RuntimeError(
                "Reference and estimates directories should have the "
                f"same number of files. Found {len(ref_files)} reference and "
                f"{len(est_files)} estimates"
                )

    # visqol c++ implementation uses stored files to calculate metrics
    # a tmp folder is created for this purpose
    if "visqol" in metrics_dict.keys():
        tmp_dir_name = datetime.now().strftime("_%Y%m%d%H%M%S_tmp")
        tmp_dir = os.path.abspath(os.path.join("eval", tmp_dir_name))

        if not os.path.isdir(tmp_dir):
            os.makedirs(tmp_dir)
    else:
        tmp_dir = None

    # generate pbar
    pbar = tqdm(zip(ref_files, est_files), total=(len(ref_files)),
            desc="Computing metrics", unit="files")

    for idx, (ref_file, est_file) in enumerate(pbar):
        # get absolute path of file
        ref_file = os.path.abspath(ref_file)
        est_file = os.path.abspath(est_file)

        # check if file has been already calculated in case of fixing a file
        if (args.file_to_fix is not None and
            metrics_data["ref_speech"].str.contains(ref_file).any() and
            metrics_data["est_speech"].str.contains(est_file).any()):
            continue

        # add files to metrics dict
        metrics_dict["ref_speech"].append(ref_file)
        metrics_dict["est_speech"].append(est_file)

        # retrieve audio data
        ref_audio, _ = sf.read(ref_file,
                frames=args.sample_rate * args.sample_duration)
        ref_audio = ref_audio.reshape(-1)
        
        est_audio, _ = sf.read(est_file,
                frames=args.sample_rate * args.sample_duration)
        est_audio = est_audio.reshape(-1)

        # compute metrics 
        if "dnsmos" in metrics_dict.keys():
            dnsmos = ep.run_dnsmos(est_audio)
            metrics_dict["dnsmos"].append(dnsmos)
            pbar.set_postfix_str(f"dnsmos: {dnsmos:.4f}")

        if "pesq" in metrics_dict.keys():
            pesq = ep.run_pesq(ref_audio, est_audio, args.sample_rate)
            metrics_dict["pesq"].append(pesq)
            pbar.set_postfix_str(f"pesq: {pesq:.4f}")

        if "stoi" in metrics_dict.keys():
            stoi = ep.run_stoi(ref_audio, est_audio, args.sample_rate)
            metrics_dict["stoi"].append(stoi)
            pbar.set_postfix_str(f"stoi: {stoi:.4f}")

        if "si-sdr" in metrics_dict.keys():
            si_sdr = ep.run_si_sdr(ref_audio, est_audio)
            metrics_dict["si-sdr"].append(si_sdr)
            pbar.set_postfix_str(f"si-sdr: {si_sdr:.4f}")

        if "visqol" in metrics_dict.keys():
            visqol = ep.run_visqol(ref_audio, est_audio, 
                                   args.sample_rate, tmp_dir)
            metrics_dict["visqol"].append(visqol)
            pbar.set_postfix_str(f"visqol: {visqol:.4f}")

        if "warpq" in metrics_dict.keys():
            warpq = ep.run_warpq(ref_audio, est_audio, 
                                 sample_rate=args.sample_rate)
            metrics_dict["warpq"].append(warpq)
            pbar.set_postfix_str(f"warp-q: {warpq:.4f}")

        # write buffer every buffer_size iterations
        if (idx + 1) % args.buffer_size == 0:

            # write buffer by appending if file_to_fix is valid
            ep.save_score_report(
                    [(k, v) for k, v in metrics_dict.items()],
                    report_filename,
                    mode="a", header=False
                    )

            # flush dict
            for k in metrics_dict.keys():
                metrics_dict[k] = []

    # flush buffer of remaining calculations
    ep.save_score_report(
            [(k, v) for k, v in metrics_dict.items()],
            report_filename,
            mode="a", header=False
            )

    # remove temp dir
    if tmp_dir is not None:
        os.rmdir(tmp_dir)
