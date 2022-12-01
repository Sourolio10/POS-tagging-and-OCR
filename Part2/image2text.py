#!/usr/bin/python
#
# Perform optical character recognition, usage:
#     python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: Dhananjay Srivastava (dsrivast), Subhodeep Jana (subjana), Souryadeep Ghosh (ghoshsou)
# (based on skeleton code by D. Crandall, Oct 2020)
#

from PIL import Image, ImageDraw, ImageFont
import sys

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    print(im.size)
    print(int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH)
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

#####
# main program
if len(sys.argv) != 4:
    raise Exception("Usage: python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png")

(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)

## Below is just some sample code to show you how the functions above work. 
# You can delete this and put your own code here!

import numpy as np

def get_init_and_trans_prob():
    train_data = []
    with open(train_txt_fname,"r") as file:
        for line in file:
            train_data.append(line.split())   
    train_data = [[j for i,j in enumerate(entry) if i%2==0] for entry in train_data]
    train_file = []
    for entry in train_data:
        sent = ""
        for idx,word in enumerate(entry):
            if word[0].isalnum():
                sent+=" "+word
            else:
                sent+=word
        train_file.append(sent.strip())
    init_prob ={}
    trans_prob = {}
    TOTAL_CHAR_CNT = 0

    for sent in train_file:
        for idx,char in enumerate(sent):
            TOTAL_CHAR_CNT+=1
            if char not in init_prob:
                init_prob[char] = 0
            init_prob[char]+=1
            if char not in trans_prob:
                trans_prob[char] = {}
            if idx!=0:
                prev_char = sent[idx-1]
                if prev_char not in trans_prob[char]:
                    trans_prob[char][prev_char] = 0
                trans_prob[char][prev_char]+=1

    for char,d in trans_prob.items():
        for prev_char,cnt in d.items():
            trans_prob[char][prev_char] = cnt/init_prob[prev_char]

    for char,cnt in init_prob.items():
        init_prob[char] = cnt/TOTAL_CHAR_CNT
    
    return init_prob,trans_prob

def nb_classifier(mtx,train_mtx,classes):
    m=0.9
    NUM_CLASSES = len(classes)
    score = {}
    for c in range(NUM_CLASSES):
        log_prob = -1*np.log(1/NUM_CLASSES)+(np.sum(np.equal(mtx[0,:],train_mtx[c,:]))*np.log(m))+(np.sum(np.not_equal(mtx[0,:],X[c,:]))*np.log(1-m))
        score[classes[c]]=log_prob
    return score

def main():
    train_mtx = []
    for k,v in train_letters.items():
        train_mtx.append([i=="*" for i in list("".join(v))])
    train_mtx = np.array(train_mtx).astype(int)

    classes = list(train_letters.keys())

    init_prob,trans_prob = get_init_and_trans_prob() 
    #ASSUMPTION
    init_prob['"'] = init_prob["'"]

    final_simple = []
    final_hmm = []
    final_hmm.append([init_prob[i] for i in classes])
    
    for v in test_letters:
        mtx = [[i=="*" for i in list("".join(v))]]
        mtx = np.array(mtx).astype(int)
        score = nb_classifier(mtx,train_mtx,classes)
        final_simple.append([score[c] for c in classes])
        curr_layer = []
        for curr_class in classes:
            curr_min_val = np.inf
            for prev_class_idx,prev_prob in enumerate(final_hmm[-1]):
                tp = trans_prob[curr_class][classes[prev_class_idx]]
                val = -np.log(tp)-np.log(prev_prob)
                if val<curr_min_val:
                    curr_min_val = val
            curr_layer.append(curr_min_val+score[curr_class])
        final_hmm.append(curr_layer)
    
    final_simple_mtx = np.array(final_simple).T
    final_hmm_mtx = np.array(final_hmm).T
    
    simple = ""
    hmm = ""
    for i in range(final_hmm_mtx.shape[1]-1):
        simple+=classes[np.argmin(final_simple_mtx[:,i])]
        hmm+=classes[np.argmin(final_hmm_mtx[:,i+1])]
    
    print("Simple: "+simple)
    print("   HMM: "+hmm)

main()