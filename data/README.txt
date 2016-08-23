Data is assumed to be stored in this folder by default, with the following structure

filelist.txt contains the list of files in the dataset
dictionary.txt contains the list of syllables used in the dataset for transcription
dictionary.txt contains the list of syllables, along with !ENTER and !EXIT non-emitting states 
./feat/<feature>/ has all the feature files in HTK format
./lblLang/ folder has all the transcription files and a master label file (MLF), all in HTK label file format
./lblTimedLang/ folder has all the time aligned transcription files and a master label file (MLF), all in HTK label file format
./wav/ folder has all the wav files
./onsets/ folder has the onset time stamps corresponding to each stroke in each file, with a one to one correspondence


