# Dataset
This branch stores the training data used in our project, which we retrieved from GRID audiovisual speech corpus with a total of 24 talkers. The corpus has videos of faces, speech audio, and transcripts of what people said.  
The goal is to make sure the model can achieve a certain amount of accuracy even with lower quality data (lower sample rate, normal video quality), which influenced which data we uploaded to intially will try to traint the model with. 
Link to data: https://spandh.dcs.shef.ac.uk/gridcorpus/#examples
## Description of Data

# Audio 
* **What**: speech recordings of each person saying short, fixed-vocabulary sentences
  * we chose to use single uniform sample rate at 25,000 Hz to balance accuracy and potential latency

# Video
* **What**:face videos of the same utterances (frontal view at ~25fps) that shows the mouth movements while the talker speaks. We chose the normal video quality because we decided to test if lower quality videos would product an acceptable lip/face landmark tracking

# Transcript/Word Alignments
* **What**: our references for what each utterance (audio & video) is saying + the timeing of each word

# Hack Technology / Project Attempted

## What you built? 

Python Code that makes a Lime green outline of the User's lips on the webcam

<img width="784" height="502" alt="IMG_7043" src="https://github.com/user-attachments/assets/5fee23bc-7d61-487c-8f6e-1114ebde018e" />[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/uB4PX0S_)


## Quick Use

```# If large files use Git LFS:
git lfs install

I helped with writing the code and testing to make sure it ran, debugging and adjusting the libraries in the venv. Jennifer helped with the GIT commits and ensuring the file was uploaded to the correct branch. 


# Get the dataset branch
git checkout <dataset-branch>
git pull


# Unpack examples
tar -xf data/25kHz/s1.tar -C data/25kHz/
tar -xf data/video/s1.tar -C data/video/

# Inspect transcripts/alignments
ls data/transcripts/```

I got to learn about how to properly use command line to make commits to branches in Github, and also how to work with very niche libraries with specific instructions and documentation. 

## Authors

Shahi Dost

## Acknowledgments

ChatGPT, https://github.com/NumesSanguis/FACSvatar/issues/33 
