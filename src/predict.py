import os
import glob
import argparse
import torch
import soundfile
import torch.nn as nn
import utils.evaluation_process as ep
from tqdm import tqdm
from dtln import DTLN
from cruse import CRUSE

# command line arguments
parser = argparse.ArgumentParser(description="""
        Deep Noise Suppression Model Prediction
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

# positionl arguments
parser.add_argument("input_dir", type=str,
                    help="""
                    directory containing noisy speech files
                    """)

parser.add_argument("output_dir", type=str,
                    help="""
                    directory where predicted files are going to be stored
                    """)

parser.add_argument("checkpoint", type=str,
                    metavar="checkpoint_path",
                    help="""
                    checkpoint to be loaded for inference
                    """)

# optional arguments
parser.add_argument("-m", "--model",
                    metavar="model",
                    type=str,
                    default="dtln",
                    help="model to be instantiated " \
                         "(dtln, dtln_gru, dtln_bigru, dtln_bilstm, "\
                         "cruse, crusex4gru)")

args = parser.parse_args()

if __name__ == "__main__":
    # check if input directory exists
    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    # create directory to store output if it does not exist
    if not os.path.isdir(args.output_dir):
        ep.log(f"Creating output directory at {args.output_dir}")
        os.makedirs(args.output_dir)

    # set up model on cpu
    ep.log(f"Instantiating model {args.model}")

    # all models are instantiated with batch_size=1
    device = torch.device("cpu")

    if args.model.lower() == "dtln":
        model = DTLN(batch_size=1)
    elif args.model.lower() == "dtln_gru":
        model = DTLN(batch_size=1, rnn_type=nn.GRU)
    elif args.model.lower() == "dtln_bigru":
        model = DTLN(batch_size=1, rnn_type=nn.GRU, rnn_bidirectional=True)
    elif args.model.lower() == "dtln_bilstm":
        model = DTLN(batch_size=1, rnn_bidirectional=True)
    elif args.model.lower() == "cruse":
        model = CRUSE(batch_size=1)
    elif args.model.lower() == "crusex4gru":
        model = CRUSE(batch_size1, bottleneck_size=4)
    else:
        raise RuntimeError(f"Model not found: '{args.model}'")

    ep.log(f"Loading checkpoint from {args.checkpoint}")

    # check checkpoint exists
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Invalid checkpoint: {args.checkpoint}")

    # load checkpoint to model and set it to eval
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # disable gradients
    torch.set_grad_enabled(False)

    # collect input files
    input_files = ep.collect_wav_files(args.input_dir)
    ep.log(f"{len(input_files)} files found")

    # inferencing
    for input_file in tqdm(input_files,
            desc="Computing predictions", unit="files"):
        # read file and truncate it accordingly if needed
        audio_data, _ = soundfile.read(input_file, dtype="float32",
                frames=16000 * 10)
        audio_data = torch.from_numpy(audio_data.reshape(1, -1))

        # predict output
        pred_audio_data = model.predict(audio_data)

        # write result
        file_name = f"pred_{os.path.basename(input_file)}"
        output_file = os.path.join(args.output_dir, file_name)
        soundfile.write(output_file, pred_audio_data, 16000)
