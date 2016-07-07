import os
import sys
from src import impl as rlcs
import utils as ut
import analysis as anls
import matplotlib.pyplot as plt
import logging
import pickle as pkl
import time

config = ut.loadConfig('config')

sylbSimFolder=config['sylbSimFolder']
transFolder=config['transFolder']
lblDir=config['lblDir']
onsDir=config['onsDir']
resultDir=config['resultDir']

queryList = [['DHE','RE','DHE','RE','KI','TA','TA','KI','NA','TA','TA','KI','TA','TA','KI','NA'],['TA','TA','KI','TA','TA','KI','TA','TA','KI','TA','TA','KI','TA','TA','KI','TA'], ['TA','KI','TA','TA','KI','TA','TA','KI'], ['TA','TA','KI','TA','TA','KI'], ['TA', 'TA','KI', 'TA'],['KI', 'TA', 'TA', 'KI'], ['TA','TA','KI','NA'], ['DHA','GE','TA','TA']]
queryLenCheck = [4,6,8,16]

for query in queryList:
    if len(query) not in queryLenCheck:
        print 'The query is not of correct length!!'
        sys.exit()

masterData = ut.getAllSylbData(tPath = transFolder, lblDir = lblDir, onsDir = onsDir)

res = anls.getPatternsInTransInGTPos(masterData, queryList)
