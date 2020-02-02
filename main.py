from __future__ import division
from train import training
from test import testing

filtbankN = 13

codebooks = training(filtbankN)

testing(codebooks, filtbankN)