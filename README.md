# Dataset
This branch stores the training data used in our project, which we retrieved from GRID audiovisual speech corpus with a total of 24 talkers. The corpus has videos of faces, speech audio, and transcripts of what people said.  
The goal is to make sure the model can achieve a certain amount of accuracy even with lower quality data (lower sample rate, normal video quality), which influenced which data we uploaded to intially will try to traint the model with. 

## Description of Data

# Audio 
* **What**: speech recordings of each person saying short, fixed-vocabulary sentences
  * we chose to use single uniform sample rate at 25,000 Hz to balance accuracy and potential latency

# Video
* **What**:face videos of the same utterances (frontal view at ~25fps) that shows the mouth movements while the talker speaks. We chose the normal video quality because we decided to test if lower quality videos would product an acceptable lip/face landmark tracking

# Transcript/Word Alignments
* **What**: our references for what each utterance (audio & video) is saying + the timeing of each word

## Quick Use

```# If large files use Git LFS:
git lfs install

# Get the dataset branch
git checkout <dataset-branch>
git pull

# Unpack examples
tar -xf data/25kHz/s1.tar -C data/25kHz/
tar -xf data/video/s1.tar -C data/video/

# Inspect transcripts/alignments
ls data/transcripts/```
