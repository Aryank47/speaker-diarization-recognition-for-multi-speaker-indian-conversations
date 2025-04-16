# Speaker Diarization & Recognition for Multi-Speaker Indian Conversations

## Project Overview

This project aims to develop a speech understanding system for multi-speaker Indian conversations, integrating **Speaker Diarization** and **Speaker Recognition**. The system is designed to handle the complexity of multilingual and code-switched speech, typical in Indian contexts. The goal is to output a time-aligned transcript with speaker labels and, when possible, identify specific speakers.

## Key Features

- **Voice Activity Detection (VAD)**: Detects speech segments and removes non-speech parts of the audio.
- **Speaker Diarization**: Segments audio based on speaker changes using pre-trained speaker embedding models and clustering algorithms.
- **Speaker Recognition**: Identifies known speakers from the audio by comparing speaker embeddings against pre-enrolled templates.

## Datasets Used

1. **AMI Meeting Corpus**: Provides multi-speaker meeting recordings with ground-truth speaker annotations.
2. **custom hinglish audio**: A large-scale speaker recognition dataset for obtaining robust speaker embeddings.

## Evaluation Metrics

- **Diarization Error Rate (DER)**: Measures the accuracy of speaker segmentation.
- **VAD F1-Score**: Evaluates the precision and recall of speech segment detection.

## Directory Structure

```
└── aryank47-speaker-diarization-recognition-for-multi-speaker-indian-conversations/
├── README.md
├── init.py
├── main.ipynb
├── requirements.txt
├── AMI/
│   ├── init.py
│   └── ES2008a/
│       ├── init.py
│       ├── ES2008a.A.segments.xml
│       ├── ES2008a.B.segments.xml
│       ├── ES2008a.C.segments.xml
│       └── ES2008a.D.segments.xml
└── Outputs/
└── ES2008a_transcript.csv
```

## Repo Setup & Installation Instructions

### Step 1: Clone the repo:

```bash
git clone https://github.com/Aryank47/speaker-diarization-recognition-for-multi-speaker-indian-conversations.git
```

### Step 2: Create a Virtual Environment

```bash
pyenv install 3.11.6
pyenv virtualenv 3.11.6 speech-diarization-env
pyenv activate speech-diarization-env
```

### Step 3: Upgrade pip and install dependencies

```bash
pip install --upgrade pip setuptools wheel
```

### Step 4: Install PyTorch for Mac M1/M2/M3

```bash
pip install torch==2.1.2 torchvision torchaudio
```

### Step 5: Install required packages

```bash
pip install numpy pandas scikit-learn scikit-image joblib
pip install pyannote.metrics
pip install openai-whisper
pip install speechbrain
```

### Step 6: Install ffmpeg (needed for Whisper ASR)

```bash
brew install ffmpeg
```

### Step 7: Verify PyTorch and MPS compatibility

```bash
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.backends.mps.is_available())"
```

## Code Structure

1. main.ipynb: Jupyter notebook that demonstrates the pipeline with code and visualizations.
2. requirements.txt: Lists all the Python dependencies needed for the project.
3. AMI/: Contains AMI meeting corpus data and annotations.
4. ES2008a/: Directory for the specific AMI meeting session data.
5. ES2008a.A.segments.xml: Ground truth speaker segmentation file for speaker A.
6. ES2008a.B.segments.xml: Ground truth speaker segmentation file for speaker B.
7. ES2008a.C.segments.xml: Ground truth speaker segmentation file for speaker C.
8. ES2008a.D.segments.xml: Ground truth speaker segmentation file for speaker D.
9. Outputs/: Directory for output results.
10. ES2008a_transcript.csv: The final transcription result with speaker labels and segments.
11. custom_hinglish.audio.wav: Custom hinglish audio file to test the code.

## Running the Pipeline

To run the pipeline on an audio file, use the provided Jupyter notebook (main.ipynb).

1. Open the notebook **main.ipynb** in Jupyter.
2. Modify paths for your data (if needed), such as the AMI audio file and annotation files.
3. Run the notebook cells sequentially to process the audio, perform VAD, diarization, speaker recognition, and transcription.

## Evaluation

The system’s performance can be evaluated using the Diarization Error Rate (DER) and VAD F1-Score. You can use the AMI ground-truth annotations to evaluate the system’s performance.

## Results

• The **VAD F1-Score** and **Diarization Error Rate (DER)** will be printed after evaluation, providing insights into the system’s accuracy.

## Future Work

• Code-switching Recognition: Improve performance for multi-lingual conversations by fine tuning the ASR model on more diverse datasets.
