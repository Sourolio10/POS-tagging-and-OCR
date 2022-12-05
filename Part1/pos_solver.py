###################################
# CS B551 Fall 2022, Assignment #3
#
# Subhadeep Jana (subjana) | Souradeep Ghosh (ghoshsou) | Dhananjay Srivastava (dsrivast)
#
# (Based on skeleton code by D. Crandall)
#

import random
import math

from gibbs import Gibbs

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#

POS_count = {}
total_count = {}
emission_prob = {}
start_prob = {}
trans_prob = {}
trans_prob_2 = {}
POS = ['adj','adv','adp','conj','det','noun','num','pron','prt','verb','x','.']
gb = Gibbs()

class Solver:
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, label):
        if model == "Simple":
            sum=0
            for x in range(0,len(sentence)):
                if(sentence[x] not in emission_prob):
                    sum = sum-36.841361487904734
                else:
                    if(label[x] not in emission_prob[sentence[x]]):
                        sum = sum-36.841361487904734
                    else:
                        sum = sum+math.log(emission_prob[sentence[x]][label[x]])
            return sum
        elif model == "HMM":
            sum=0
            ans = []
            val = []
            num = len(sentence)
            V = []
            for i in range(12):
                t = []
                for j in range(num):
                    t.append(0)
                V.append(t)
        
            # 0th column of V
            for i in range(12):
                if(sentence[0] not in emission_prob):
                    V[i][0] = start_prob[POS[i]] * 0.0000000000001
                    sum=sum+math.log(start_prob[POS[i]])-36.841361487904734
                else:
                    if(POS[i] not in emission_prob[sentence[0]]):
                      V[i][0] = start_prob[POS[i]] * 0.0000000000001
                      sum=sum+math.log(start_prob[POS[i]])-36.841361487904734
                    else:
                       V[i][0] = start_prob[POS[i]] * emission_prob[sentence[0]][POS[i]]
                       sum=sum+math.log(start_prob[POS[i]])+math.log(emission_prob[sentence[0]][POS[i]])
        
            temp_pos = 0
            temp_max = -1
            for i in range(12):
                if(V[i][0] > temp_max):
                    temp_max = V[i][0]
                    temp_pos = i
        
            val.append(temp_max)
            ans.append(POS[temp_pos])

            for j in range(1,num):
                for i in range(12):
                    if(sentence[j] not in emission_prob):
                          V[i][j] = trans_prob[ans[j-1]][POS[i]] * val[j-1] * 0.0000000000001
                          sum=sum+(0 if trans_prob[ans[j-1]][POS[i]]==0 else math.log(trans_prob[ans[j-1]][POS[i]]))-36.841361487904734
                    else:
                        if(POS[i] not in emission_prob[sentence[j]]):
                            V[i][j] = trans_prob[ans[j-1]][POS[i]] * val[j-1] * 0.0000000000001
                            sum=sum+(0 if trans_prob[ans[j-1]][POS[i]]==0 else math.log(trans_prob[ans[j-1]][POS[i]]))-36.841361487904734
                        else:
                            V[i][j] = trans_prob[ans[j-1]][POS[i]] * val[j-1] * emission_prob[sentence[j]][POS[i]]
                            sum=sum+(0 if emission_prob[sentence[j]][POS[i]]==0 else math.log(emission_prob[sentence[j]][POS[i]]))+(0 if trans_prob[ans[j-1]][POS[i]]==0 else math.log(trans_prob[ans[j-1]][POS[i]]))-36.841361487904734
                prev_pos = 0
                prev_max = -1
                for i in range(12):
                    if(V[i][j]>prev_max):
                        prev_max = V[i][j]
                        prev_pos = i
                val.append(prev_max)
                ans.append(POS[prev_pos])


            return sum
        elif model == "Complex":
            pred_pos,log_probs,map_val_log = gb.run_gibbs(sentence)
            return map_val_log[-1]
        else:
            print("Unknown algo!")

    # Do the training!
    #
    def train(self, data):

        gb.train(data)
        
        # Creating the count dictionary
        for x in range(0,len(data)):
            for word in data[x][0]:
                if(word not in total_count):
                    total_count[word] = 1
                else:
                    total_count[word] += 1
        
        # Creating the emission probabilities dictionary
        for x in range(0,len(data)):
            for y in range(0,len(data[x][0])):
                if(data[x][0][y] not in emission_prob):
                    temp_dict = {}
                    temp_dict[data[x][1][y]] = 1
                    emission_prob[data[x][0][y]] = temp_dict
                else:
                    if(data[x][1][y] not in emission_prob[data[x][0][y]]):
                        emission_prob[data[x][0][y]][data[x][1][y]] = 1
                    else:
                        emission_prob[data[x][0][y]][data[x][1][y]] +=1
        

        for x in emission_prob:
            for y in emission_prob[x]:
                emission_prob[x][y] = emission_prob[x][y]/total_count[x]
        

        # Creating the start probabilities dictionary
        for x in range(0,len(data)):
            if(data[x][1][0] not in start_prob):
                start_prob[data[x][1][0]] = 1
            else:
                start_prob[data[x][1][0]] += 1
        

        for k in POS:
            if(k not in start_prob):
                start_prob[k] = 0
            else:
                start_prob[k] = start_prob[k]/len(data)
        
        # Creating the transition probabilities dictionary

        for k in POS:
            trans_prob[k] = {}
            trans_prob_2[k] = {}
            POS_count[k] = 0
        
        for m in trans_prob:
            for e in POS:
                trans_prob[m][e] = 0
                trans_prob_2[m][e] = 0
        
        for x in range(0,len(data)):
            for y in range(0,len(data[x][0])-1):
                POS_count[data[x][1][y]] += 1
                trans_prob[data[x][1][y]][data[x][1][y+1]] += 1

        for x in range(0,len(data)):
            for y in range(0,len(data[x][0])-2):
                trans_prob[data[x][1][y]][data[x][1][y+2]] += 1

        for m in trans_prob:
            for n in trans_prob[m]:
                trans_prob[m][n] = trans_prob[m][n]/POS_count[m]
        
        for m in trans_prob_2:
            for n in trans_prob_2[m]:
                trans_prob_2[m][n] = trans_prob_2[m][n]/POS_count[m]
        
        
        
        pass

    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        ans = []
        for x in sentence:
            if(x not in emission_prob):
                ans.append('noun')
            else:
                max_val = -1
                pos = ''
                for y in emission_prob[x]:
                    if(emission_prob[x][y]>max_val):
                        max_val = emission_prob[x][y]
                        pos = y
                ans.append(pos)
        return ans

    def hmm_viterbi(self, sentence):
        ans = []
        val = []
        num = len(sentence)
        V = []
        for i in range(12):
            t = []
            for j in range(num):
                t.append(0)
            V.append(t)
        
        # 0th column of V
        for i in range(12):
            if(sentence[0] not in emission_prob):
                V[i][0] = start_prob[POS[i]] * 0.00001
            else:
                if(POS[i] not in emission_prob[sentence[0]]):
                    V[i][0] = start_prob[POS[i]] * 0.00001
                else:
                    V[i][0] = start_prob[POS[i]] * emission_prob[sentence[0]][POS[i]]
        
        temp_pos = 0
        temp_max = -1
        for i in range(12):
            if(V[i][0] > temp_max):
                temp_max = V[i][0]
                temp_pos = i
        
        val.append(temp_max)
        ans.append(POS[temp_pos])

        for j in range(1,num):
            for i in range(12):
                if(sentence[j] not in emission_prob):
                    V[i][j] = trans_prob[ans[j-1]][POS[i]] * val[j-1] * 0.00001
                else:
                    if(POS[i] not in emission_prob[sentence[j]]):
                        V[i][j] = trans_prob[ans[j-1]][POS[i]] * val[j-1] * 0.00001
                    else:
                        V[i][j] = trans_prob[ans[j-1]][POS[i]] * val[j-1] * emission_prob[sentence[j]][POS[i]]
            prev_pos = 0
            prev_max = -1
            for i in range(12):
                if(V[i][j]>prev_max):
                    prev_max = V[i][j]
                    prev_pos = i
            val.append(prev_max)
            ans.append(POS[prev_pos])

             
        return ans
        #return [ "noun" ] * len(sentence)

    def complex_mcmc(self, sentence):
        pred_pos,log_probs,map_val_log = gb.run_gibbs(sentence)
        return pred_pos



    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        else:
            print("Unknown algo!")

