# About this repository
This is the accompanying repository of the thesis work **Deep Noise Suppression 
for Real Time Speech Enhancement in a Single Channel Wide Band Scenario** developed
as part of the requirements of the **Master in Sound and Music Computing** at
**Universitat Pompeu Fabra**. It provides the necessary code the train and evaluate
all the models studied throughout the aforementioned research project. In order
use this repository, some external dependencies addressed in the following
sections are needed. Additionally, a selection of pretrained models and audio examples are provided. In case of any inquiries, feel free to contact the author
through his website [https://www.estebangomez.cl](https://www.estebangomez.cl).

# Abstract
Speech enhancement can be regarded as a dual task that addresses two important issues of degraded speech: Speech *quality* and speech *intelligibility*. This work is focused on speech *quality* in a real time context. Algorithms that improve speech quality are sometimes referred to as *noise suppression algorithms*, since they enhance quality by suppressing the background noise of the degraded speech. Real time capable algorithms are especially important for devices with a limited processing power and physical constraints that cannot make use of large architectures, such as hearing aids or wearables. This work uses a deep learning based approach to expand on two previously proposed architectures in the context of the [Deep Noise Suppression Challenge](https://github.com/microsoft/DNS-Challenge) carried out by [Microsoft Corporation](https://www.microsoft.com). This challenge has provided datasets and resources to teams of researchers with the common goal of fostering the research on the aforementioned topic. The outcome of this thesis can be divided into three main contributions: First, an extended comparison between six variants of the two selected models, considering performance, computational complexity and real time efficiency analyses. Secondly, making available an open source implementation of one of the proposed architectures as well as a framework translation of an existing implementation. Finally, proposed variants that outperform the previously defined models in terms of denoising performance, complexity and real time efficiency.

# Structure of this repository
The content of this repository can be summarized as follows:

- `docs`: Folder containing the written thesis report.
- `src`: Folder that contains the actual code used in this project.
- `src/dataset`: Contains an `example_dataset`(only provided as example since it only has 50 clean/noisy speech pairs) and a `sampling` folder containing examples to be predicted in intermediate steps of a model training.
- `src/logs`: Folder were training `tensorboard` logs and checkpoints are saved.
- `src/predicted`: Place to store the folders with predicted audios when using `predict.py`.
- `src/pretrained_models`: As its name implies, it has a selection of pretrained models ready to use.
- `src/reports`: Folder where `.csv` files containing the scores for each preddicted file using the different provided metrics (SI-SDR, SNR, ViSQOL, etc) are stored.
- `src/utils`: Folder containing a collection of utilities used to implement different classes and functions needed throughout the project.
- `src/visqol`: Expected location for ViSQOL's content (see **External dependencies** for more details).
- `src/cruse.py`, `src/dtln.py`: Implementation of the actual model classes.
- `src/train_cruse.py`, `src/train_dtln.py`: Scripts for training each model variant.
- `src/dns_dataset.py`: Dataloader implementation.
- `src/predict.py`: Script to predict clean speech given a noisy speech folder.
- `src/profiler.py`: Script to analyze a given model in terms of real time performance and complexity.
- `src/requirements.txt`: Python dependencies.
- `src/score.py`: Script for obtaining the performance metric scores in `.csv` format of a specified folder containing predicted files.

# External dependencies

## python dependencies
This repository requires `python 3.7.10` or higher. It may work in older versions, although it has not been tested. In order to install `python` dependencies, please run the following command:

```sh
pip install -r requirements.txt
```

## ViSQOL (optional)
ViSQOL (Virtual Speech Quality Objective Listener) is an objective, full-reference metric for perceived audio quality. In this project, it is used to score the predicted audio files. ViSQOL was implemented in C++ by Google. In this project, it is called by `score.py`. This script assumes that ViSQOL is available in a folder called `visqol` inside the `src` folder of the project. The path to the executable would then be `/src/visqol/bazel-bin/visqol`. The following instructions were copied from the **Build** section of the original repository that can be found at [https://github.com/google/visqol](https://github.com/google/visqol).

### Linux/Mac Build Instructions
**1. Install Bazel**
- Bazel can be installed following the instructions for [Linux](https://docs.bazel.build/versions/master/install-ubuntu.html) or [Mac](https://docs.bazel.build/versions/master/install-os-x.html).
- Tested with Bazel version `3.4.1`.

**2. Build ViSQOL**
- Change directory to the root of the ViSQOL project (i.e. where the WORKSPACE file is) and run the following command: `bazel build :visqol -c opt`

### Windows Build Instructions (Experimental, last Tested on Windows 10 x64, 2020 August)
**1. Install Bazel**
- Bazel can be installed for Windows from [here](https://docs.bazel.build/versions/master/windows.html).
- Tested with Bazel version `3.5.0`.

**2. Install git**
- `git` for Windows can be obtained from the [official git website](https://git-scm.com/downloads).
- When installing, select the option that allows `git` to be accessed from the system shells.

**3.Build ViSQOL:**
Change directory to the root of the ViSQOL project (i.e. where the WORKSPACE file is) and run the following command: `bazel build :visqol -c opt`


## DNSMOS (optional)
DNSMOS is a non-intrusive deep learning based metric developed by Microsoft and provided to researchers as a web-API upon request as part of the [Deep Noise Suppression Challenge](https://github.com/microsoft/DNS-Challenge). To use DNSMOS, you need to enter your corresponding `SCORING_URI` and `AUTH_KEY` in the body of the `run_dnsmos()` function inside `/src/utils/evaluation_process.py`. More information about this metric and how to request access to it can be found [here](https://www.microsoft.com/en-us/research/publication/dnsmos-a-non-intrusive-perceptual-objective-speech-quality-metric-to-evaluate-noise-suppressors/).

# Inferencing using pretrained models
The are three provided pretrained models that can be directly used for inferencing. These are inside the `pretrained_models`. In order to use one of these models for prediction, you `cd` inside the `src` folder and issue the following command:

```sh
python predict.py <input_dir> <output_dir> <checkpoint> -m <model>
```

For example, clean the noisy speech found inside `dataset/example_dataset/noisy_speech` using `pretrained_models/DTLN_BiLSTM_500h.tar`, the command would be the following:

```sh
python predict.py dataset/example_dataset/noisy_speech predicted/DTLN_BiLSTM_500h pretrained_models/DTLN_BiLSTM_500h.tar -m dtln_bilstm
```

This will create a `DTLN_BiLSTM_500h` folder inside the `predicted` folder that will contain all the predicted audio files. 

It is always possible to use the command line help by using the `-h` argument. There it is also possible to see the identifiers for each model to instantiate them correctly. If there is a mismatch between the utilized checkpoint and the model instance you will get an error because the structure contained in the checkpoint differs from that found when the model was instantiated.

```sh
python predict.py -h
```

# Scoring inference results
After a set of predictions is computed, these can be evaluating using some of all of the provided metrics (STOI, SI-SDR, PESQ, ViSQOL, DNSMOS, WARP-Q) by using the following command:

```sh
python score.py <reference_dir> <estimates_dir>
```
As an example, let's asume the prediction of `dataset/example_dataset` are store in `predicted/example_predictions`. Then, the command would be:

```sh
python score.py dataset/example_dataset/clean_speech predicted/example_predictions
```

Please note that DNSMOS and ViSQOL are not included by default since these require external resources to run. If you want to include them, you can do it by typing:

```sh
python score.py dataset/example_dataset/clean_speech predicted/example_predictions -m stoi si-sdr pesq visqol dnsmos warpq
```

This will compute all the available metrics. Once the metrics are computed, a `.csv` file will be automatically created inside the `reports` folder. It will contain one column per metric with the respective results as well as two additional columns showing the reference and estimate path used for each prediction.

# Inspecting a model
It is possible to inspect the amount of parameters and FLOPs performed by each layer as well as the inference time on a given machine by issuing the following command:

```sh
python profiler.py -m <model>
```

For example, to see the information about `CRUSEx4GRU` you must issue the following command:

```sh
python profiler.py -m crusex4gru
```

This will print the statistics during the inference if 1000 prediction cycles along with a table showing the parameters on each layer as well as the FLOPs needed to perform each calculation. Again, further options can be displayed by typing:

```sh
python profiler.py -h
```

# Training a model from scratch
Two files are provided to train a model from scratch. `train_dtln.py` and `train_cruse.py` can be used to train a model of their respective classes. The five possible models to be trained are `dtln`, `dtln_gru`, `dtln_bigru`, `dtln_bilstm`, `cruse` and `crusex4gru`. Their description can be found in the thesis written report inside the `docs` folders. The dataset to trained these models is not provided because of its size, but can be synthezised using the scripts provided by the [DNS-Challenge](https://github.com/microsoft/DNS-Challenge). Please refer to their repository for further details. `dns_dataset.py` file already contains a data loader that to handle the DNS-Challenge dataset. The command to train a model is:

```sh
python train_dtln.py <input_dir> <output_dir> -m <model> -d <device> -b <batch_size>
```

or 

```sh
python train_cruse.py <input_dir> <output_dir> -m <model> -d <device> -b <batch_size>
```

For example, to train a `DTLN_BiGRU` on a GPU using the `example_dataset`, the command would be as follows:

```sh
python train_dtln.py dataset/example_dataset/noisy_speech dataset/example_dataset/clean_speech -m dtln_bigru -d cuda:0 -b 10
```
By doing this, the training process will start and information about it will be displayed on the screen along with a progress bar showing the information of the latest epoch.

A few things to consider:
1. Some parameters have already a default value and therefore may not need to be specified depending on your setup, the details can be check using the help command.

```sh
python train_dtln.py -h
```

or 

```sh
python train_cruse.py -h
```

2. Conversely, several other options are available to tweak the training procedure, these can be explored using the same command.

3. Both `train_dtln.py` and `train_cruse.py` have a default batch size as specified in their respective papers referenced in the thesis document. `-b 10` is used as an example because bigger batch sizes could cause an error with the example dataset since it only contains 50 audio files and a train/validation split is by default set to 80/20. With a bigger dataset, bigger batch sizes will be okay as well.

4. During the training process, a folder inside the `logs` folder will be created. It will contain information that can be visualized in the browser using `tensorboard` in order to facilitate the tracking of the training process. Additionally, at intermediate steps, checkpoints will be saved and the best three checkpoints in terms of validation loss are kept. Further modifications of the routines that are called during the training process can be inspected by looking into `train_dtln.py` and `train_cruse.py`. Moreover, the implementation of the callbacks is inside `utils/callbacks.py`. New callbacks can also be added by creating new classes that inherit from `TrainingProcessCallback`. If you have used [`pytorch_lightning`](https://www.pytorchlightning.ai/) you may be already familiar with this concept. This repository is implemented using [`PyTorch`](https://pytorch.org/), although the structured followed is heavily inspired by [`pytorch_lightning`](https://www.pytorchlightning.ai/).

5. To launch `tensorboard` use the following command:

```sh
tensorboard --logdir <logs_dir>
```

This will display a URL that can be clicked or copy/pasted depending on your terminal of choice. As soon as you do it, you will see a website that will locally display information of your training process such as the current epoch, training loss, validation loss, spectrogram and waveform plots, predicted audio examples and histograms of weights and biases of each layer.

# Acknowledgements
I would like to express my depeest gratitude to [Andrés Pérez, PhD](https://scholar.google.es/citations?user=e-s-24YAAAAJ&hl=es) and [Pritish Chandna, PhD](https://scholar.google.com/citations?user=q4lZQbkAAAAJ&hl=en), the supervisor and co-supervisor of this project, respectively. Additionally, I want to thank [Voicemod](https://www.voicemod.net/) for providing the necessary hardware to carry out the experiments needed throghout the project. Last but not least, I would like to thank [Microsoft Corporation](https://www.microsoft.com) for allowing me to use their proprietary tools that are provided to researchers as part of their [Deep Noise Suppression Challenge](https://github.com/microsoft/DNS-Challenge).

# License
This work is licensed under a [Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
