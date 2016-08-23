Folder to store all the transcription experiment setup related parameters, settings and lists for HTK. The following files are needed. 

alphabet.txt: The list of syllables
alphabetExt.txt: The list of syllables with an !ENTER and !EXIT non-emitting syllables
dictionary.txt: Sorted list of syllables
dictionaryExt.txt: Sorted list of syllables, with an !ENTER and !EXIT non-emitting syllables
grammar.txt: A grammar network with the alphabets defined as needed
HCompV.config: Config file to be used for flat start with HCompV
HCopy_0_D_A.config: Config file used for feature extraction from audio files
hmmListExt: List of HMMs to be built (equivalent to the list of syllables, with !ENTER and !EXIT syllables included in ths list)
listFeatFiles_mfcc_0_d_a.txt: List of feature files with full path to the file
mixcmds.hhed: Commands used by HHed to edit the HMM during training (Optional)
proto_mfcc_0_d_a: HMM prototype definition file, that uses a 39 dimensional MFCC_0_D_A feature 
