
# What you built? 
Our project is a Machine Learning model that uses facial recognition to track a person's lips to understand what was said without audio. We used MediaPipe for the facial feature mappings and TensorFlow Keras for the ML aspect.

<img width="503" height="320" alt="image" src="https://github.com/user-attachments/assets/27d66ddc-d46a-42ff-8415-3f61b09932f7" />

## Dataset
The training data used in our project was retrieved from the GRID audiovisual speech corpus with a total of 34 talkers. The corpus has videos of faces, speech audio, and transcripts of what people said.  
The goal is to make sure the model can achieve a certain amount of accuracy even with lower quality data (lower sample rate, normal video quality), which influenced which data we uploaded to intially. The dataset is limited to a very specific sequence structure: [verb] + [color] + [preposition] + [letter] + [number] + [adverb] for ex: Lay Red at a2 please
Link to data: https://spandh.dcs.shef.ac.uk/gridcorpus/#examples

## Description of Data

## Audio 
- Includes speech recordings of each person saying short, fixed-vocabulary sentences, we downloaded the 25 kHz ones to balance accuracy and potential latency. For this project we didn't use any audio data since we wanted to focus on lip tracking.

## Video
- Face videos of the same utterances (frontal view at ~25fps and 360x288 ~1kbit/s) that shows the mouth movements while the talker speaks. We chose the normal video quality because we decided to test if lower quality videos could still produce an acceptable lip/face landmark tracking. It also gives us an easier starting point to build on later.

## Transcript/Word Alignments
- Prodived to us by GRID, every video has a transcript that includes a starting and ending timestamp for the word said in that interval. This is references for what each utterance (audio & video) is saying + the timing of each word.

# Who did what 
Shahi focused a lot on the data processing and cleaning in this project, he wrote most of the functions that make the data understandable to the model. Jennifer focused on the model itself, she wrote a lot of the code to train + test it and also helped teach Shahi Git.

# What we learned
Finding, processing, and organizing data was a challenge, and understanding what model would work best for us was a lot harder than expected. - Jennifer
I got to learn about how to properly use command line to make commits to branches in Github, and also how to work with very niche libraries with specific instructions and documentation. - Shahi

# What didn't work

We initially wanted to have a large corpus to train on, but found that the more words we added the more intense the training requirement became. For this reason we opted for the GRID dataset even though we could have used Oxford LRS3 (TED) dataset. We also had a little trouble trying to understand what type of model we could use since this is a very unique case. Since we're using our tracked video points to make an inference about what is not in the video (audio) common skeleton related computer vision models we would otherwise have used became mostly irrelevant. This narrowed our options extremely.

# What we would add
Our original idea was to have a model that could either read lips to predict what was said and use speech to text to understand inputs. After trying a few different apis and libraries, we ultimately had to sacrifice the speech-to-text to focus on lip tracking. If we had more time we would add the speech to text to make the model more robust and use other clues to fill in the accuracy gaps from lip reading. Since our model reached an accuracy of 47% this would greatly improve its performance.

## Authors
Jennifer Lee
Shahi Dost

change for feedback PR

## Acknowledgments
ChatGPT, https://github.com/NumesSanguis/FACSvatar/issues/33, https://www.youtube.com/watch?v=0_PgWWmauHk
