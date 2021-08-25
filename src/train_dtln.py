import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import (
        DataLoader,
        SubsetRandomSampler,
        SequentialSampler
        )
from dns_dataset import DNSDataset
from dtln import DTLN, DTLNTrainingProcess
from utils.criterion import NegativeSNRLoss
from utils.callbacks import (
        ScheduledCheckpointCallback,
        BestCheckpointCallback,
        AudioProcessTrackerCallback
        )

# suppress tensorflow tpu related warnings
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# command line args
parser = argparse.ArgumentParser(
        description="deep noise suppression model trainer from DTLN "
        "Dual-signal Transformation Long Short-term Memory Network",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# positional arguments
parser.add_argument("input_dir", type=str,
                    help="directory containing noisy speech files")

parser.add_argument("output_dir", type=str,
                    help="directory containing clean speech files")

# optional arguments
parser.add_argument("-b", "--batch_size", 
                    type=int, metavar="batch_size",
                    default=32,
                    help="""
                    batch size used to extract data from dataset
                    """)

parser.add_argument("-c", "--checkpoint",
                    type=str, metavar="filepath",
                    default=None,
                    help="""
                    resume training from a given checkpoint
                    """)

parser.add_argument("-d", "--device",
                    type=str, metavar="device_id", default="cpu",
                    help="""
                    device to be used (i.e. cuda:0 for single gpu)
                    """)

parser.add_argument("-e", "--epochs",
                    dest="max_epochs", metavar="max_epochs",
                    type=int, default=1000,
                    help="""
                    maximum number of epochs
                    """)

parser.add_argument("-g", "--grad_norm_clipping",
                    metavar="grad_norm_clipping_value",
                    type=float, default=3.0,
                    help="""
                    gradient norm clipping
                    """)

parser.add_argument("-l", "--learning_rate",
                    metavar="learning_rate", type=float,
                    default=1e-3,
                    help="""
                    initial learning rate
                    """)

parser.add_argument("-L", "--learning_rate_scheduler",
                    metavar="lr_scheduler_start_epoch", type=int,
                    dest="lr_scheduler_start_epoch",
                    default=50,
                    help="""
                    first epoch where the learning rate scheduler becomes active
                    """)

parser.add_argument("-m", "--model",
                    metavar="model_name", type=str,
                    default="dtln",
                    help="""
                    model to be trained: dtln, dtln_gru, dtln_bigru, dtln_bilstm
                    """)

parser.add_argument("-o", "--overfit_single_batch", action="store_true",
                    help="""
                    enable overfit_single_batch mode
                    """)

parser.add_argument("-p", "--prefix",
                    default="",
                    metavar="prefix_str", type=str,
                    help="""
                    prefix appended to run files
                    """)

parser.add_argument("-s", "--sampling_dir",
                    default=os.path.join("dataset", "sampling"),
                    metavar="dir", type=str,
                    help="""
                    directory to sample using AudioProcessTrackerCallback
                    """)

parser.add_argument("-r", "--random_seed",
                    metavar="number", type=str, default=0,
                    help="""
                    random seed used for stochastic processes
                    """)

args = parser.parse_args()

if __name__ == "__main__":
    # device
    device = torch.device(args.device)

    # random state for replicability
    rand_gen = np.random.RandomState(args.random_seed)
    torch.manual_seed(args.random_seed)

    # get dataset
    dataset = DNSDataset(input_dir=args.input_dir,
                         output_dir=args.output_dir,
                         sample_duration=10)

    # prepare dataset samplers
    dataset_idx_list = np.arange(len(dataset))

    # shuffle indices
    rand_gen.shuffle(dataset_idx_list)

    # train validation split of 0.8/0.2
    val_split_idx = int(np.floor(0.2 * len(dataset)))

    # get index list for train and validation split
    train_idx_list, val_idx_list = (dataset_idx_list[val_split_idx:],
                                    dataset_idx_list[:val_split_idx])

    # generate samplers
    train_sampler = SubsetRandomSampler(train_idx_list)
    val_sampler = SequentialSampler(val_idx_list)

    # train and validation dataloaders
    train_dataloader = DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  drop_last=True,
                                  sampler=train_sampler,
                                  num_workers=os.cpu_count())

    val_dataloader = DataLoader(dataset,
                                batch_size=args.batch_size,
                                drop_last=True,
                                sampler=val_sampler,
                                num_workers=os.cpu_count())

    # model
    if args.model.lower() == "dtln":
        model = DTLN(batch_size=args.batch_size)
    elif args.model.lower() == "dtln_gru":
        model = DTLN(batch_size=args.batch_size, rnn_type=nn.GRU)
    elif args.model.lower() == "dtln_bigru":
        model = DTLN(batch_size=args.batch_size, rnn_type=nn.GRU,
                                                 rnn_bidirectional=True)
    elif args.model == "dtln_bilstm":
        model = DTLN(batch_size=args.batch_size, rnn_type=nn.LSTM,
                                                  rnn_bidirectional=True)
    else:
        raise NotImplementedError(f"Model '{args.model}' does not exist")

    # criterion
    criterion = NegativeSNRLoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, min_lr=1e-6,
            patience=10, cooldown=0, verbose=True)

    # callbacks
    scheduled_checkpoint_callback = ScheduledCheckpointCallback(
            start_epoch=0, epoch_interval=10)

    best_checkpoint_callback = BestCheckpointCallback(n_best=3, verbose=True)

    sampling_input_dir = os.path.join(args.sampling_dir, "noisy_speech")
    sampling_output_dir = os.path.join(args.sampling_dir, "clean_speech")

    audio_process_tracker_callback = AudioProcessTrackerCallback(
            input_dir=sampling_input_dir,
            output_dir=sampling_output_dir,
            epoch_interval=20, sample_duration=10, window_size=512,
            hop_size=128, pred_fn=model.predict)

    # train
    training_process = DTLNTrainingProcess(
                        train_dataloader,
                        val_dataloader,
                        device=device,
                        lr_scheduler=lr_scheduler,
                        lr_scheduler_start_epoch=args.lr_scheduler_start_epoch,
                        grad_norm_clipping=args.grad_norm_clipping,
                        max_epochs=args.max_epochs,
                        run_name=model.__class__.__name__,
                        run_name_prefix=args.prefix,
                        resume_from_checkpoint=args.checkpoint,
                        overfit_single_batch=args.overfit_single_batch,
                        callbacks=[
                            scheduled_checkpoint_callback,
                            best_checkpoint_callback,
                            audio_process_tracker_callback
                            ])

    training_process.fit(model, criterion, optimizer)
