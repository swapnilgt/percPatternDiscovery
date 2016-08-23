# Introduction

This repository contains the code for pattern discovery from percussion solo recordings. The approach uses a transcription followed a search for a preset of patterns. The algorithm is demonstrated with tabla solo recordings and presented in the paper, 

S. Gupta, A. Srinivasamurthy, M. Kumar, H. Murthy, X. Serra, [Discovery of Syllabic Percussion Patterns in Tabla Solo Recordings](http://ismir2015.uma.es/program_and_papers_ismir2015.html), Proceedings of the 16th International Society for Music Information Retrieval Conference (ISMIR 2015) (pp. 385-391), 2015, Malaga, Spain. See the [companion page](http://compmusic.upf.edu/ismir-2015-tabla) for further information. The Mulgaonkar Tabla Solo (MTS) dataset described in the paper is explained further at [http://compmusic.upf.edu/tabla-solo-dataset](http://compmusic.upf.edu/tabla-solo-dataset). 

The repository is organized into two parts - transcription and search. Transcription code is in transcription/ folder. Pattern search code with RLCS (Rough Longest Common Subsequence) is in the folder rlcs/. The other two folders data/ and results/ are placeholders for default locations for dataset and results, respectively. 

Here we will explain the necessary steps and configurations that you can do to run the code for pattern transcription and search. To understand the method and variables in the code mentioned below, refer to the article listed above. Transcription and search can be used independently of each other if the goal is only one of the tasks. Hence they are organized into separate folders and work independently. 

# Percussion transcription
The percussion transcription code uses the [hidden Markov model toolkit (HTK)](http://htk.eng.cam.ac.uk/) and has a few helper scripts in MATLAB to run complete transcription experiments. Detailed help is provided in the file transcription/exptWrapper.m 

The only dependency to run the transcription experiments is HTK. 

# Pattern Search
The pattern search code is the rlcs/ folder. 
## Dependencies
Following are the python dependencies that you would need to run the code:
  * [matplotlib](http://matplotlib.org/)
  * [numpy](http://www.numpy.org/)
  * [scipy](https://www.scipy.org/)
  
## Code and Data
Here we explain the steps to run the code and the data needed for that.
### config
This file is present in the location `rlcs/config`.This file consists of the information about the basic config that would enable us to run the experiments using the code. This file has the following variables:
  1. **sylbSimFolder**: This is the folder where wee have the `.mat` (file generated using MATLAB) files for the similarity measure between the syllables as mentioned in **sylbSimFile**. These files hold the similarity relation between syllables. The important point to mention here is that the order of **rows** and **columns** in these similarity matrices are same as the order of syllables in **sylbSimFile**. This folder should have the files related all the different similarity metrices for which we want to run the experiments. To mention the list of similarities we set the variable `similarityList` in `run.py`.
  2. **sylbSimFile**: The file which contains the list of the syllables in the dataset. Each syllable is listed on a new line.
  3. **transFolder**: Folder where we have the results for the automatic transcriptions of the data.
  4. **lblDir**: Directories having the files relevant to different audio files in the dataset. These files are with the extension `.lab` and contain the syllables in the composition with each syllable listed on a new line.
  5. **onsDir**: Directories having the files relevant to different audio files in the dataset. These files are with the extension `.csv` and contain the time stamp of each syllable in the composition(corresponding to the file in **lblDir**) with each timeStamp listed on a new line.
  6. **resultDir**: The result where you want to dump the result of the pattern search experiments.

### How to run the code?
There are few variable that you need to set in the `run.py` before running the code. They are as follows:
  * `queryList`: This is the list of the query patterns for which we want to run the RLCS experiment.
  * `queryLenCheck`: This represents the set of lenghts of the different query patterns mentioned in `queryList`. This is just a check to ensure that the length of the pattern is actually what is being mentioned. Although, it is not a robust as we would expect but is just an initial check on the query pattern list.
  * `similarityList`: This is the list of the different similarity types that we want to run experiment with. All the names in this list should have a corresponding `.mat` file present in **sylbSimFolder** (as mentioned earlier).

To run the code, these are the commands:

  * **For baseline**: `python run.py baseline` (from the terminal) 
  * **For complete grid search**: `python run.py` (from the terminal)


