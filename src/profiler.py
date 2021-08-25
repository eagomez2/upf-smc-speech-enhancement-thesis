import os
import torch
import argparse
import time
import numpy as np
import torch.nn as nn
from dtln import DTLN
from cruse import CRUSE
from utils.model_analyzer import ModelAnalyzer

# command line arguments
parser = argparse.ArgumentParser(description="""
        Deep Noise Suppression Model Profiler
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-d", "--drop_n",
                    metavar="iterations_to_drop",
                    type=int,
                    default=10,
                    help="""
                    iterations to be drop from calculated stats
                    """)

parser.add_argument("-n", "--n_iter",
                    metavar="n_iter",
                    type=int,
                    default=1000,
                    help="""
                    number of iterations to be performed
                    """)

parser.add_argument("-m", "--model",
                    metavar="model",
                    type=str,
                    default="DTLN",
                    help="""
                    model to be instantiated for evaluation (e.g. dtln, cruse)
                    """)

args = parser.parse_args()

if __name__ == "__main__":
    # only cpu is supported for now
    device = torch.device("cpu")

    if args.model.lower() == "dtln":
        model = DTLN(batch_size=1, sample_duration=0.032) # to get 512 samples
        probe = torch.randn([1, 512], dtype=torch.float32)
        batch_correction = 1.0

    elif args.model.lower() == "dtln_gru":
        model = DTLN(batch_size=1, sample_duration=0.032, 
                      rnn_type=nn.GRU)
        probe = torch.randn([1, 512], dtype=torch.float32)
        batch_correction = 1.0

    elif args.model.lower() == "dtln_bigru":
        model = DTLN(batch_size=1, sample_duration=0.032, 
                      rnn_type=nn.GRU, rnn_bidirectional=True)
        probe = torch.randn([1, 512], dtype=torch.float32)
        batch_correction = 1.0

    elif args.model.lower() == "dtln_bilstm":
        model = DTLN(batch_size=1, sample_duration=0.032,
                      rnn_bidirectional=True)
        probe = torch.randn([1, 512], dtype=torch.float32)
        batch_correction = 1.0

    elif args.model.lower() == "cruse":
        model = CRUSE(batch_size=1)
        probe = torch.randn([1, 161, 5], dtype=torch.float32)
        batch_correction = 1.0 / 5.0

    elif args.model.lower() == "crusex4gru":
        model = CRUSE(batch_size=1, bottleneck_size=4)
        probe = torch.randn([1, 161, 5], dtype=torch.float32)
        batch_correction = 1.0 / 5.0

    else:
        raise RuntimeError(f"Model {args.model} not found")

    # instantiate analyzer
    model_analyzer = ModelAnalyzer(module_exceptions=[nn.LeakyReLU])
    model_analyzer.analyze_inference_speed(
            model, probe, batch_correction=batch_correction)
    print("\n")
    model_analyzer.analyze_complexity(model, probe)
