# Part 1: Part-of-speech tagging

## Training
We have used the training corpus provided to us to find the following dictionaries: 
- POS_count : Stores the number of occurences of each POS in the training corpus.
- total_count : Stores the number of occurences of each unique word in the training corpus.
- emission_prob : Stores the emission probabilites of a POS given a particular word.  
- start_prob : Stores the initial starting probabilities of each POS.
- trans_prob : Stores the transition probabilities of each POS given a previous POS.
- trans_prob_2 : Stores the transition probabilities of each POS given a POS 2 steps back.

## Methodology
### Simple
For implementing the simple bayes net, we have made use of the emission_prob dictionary. We are looping over the input sentence to extract each word. Then we are checking if that word is present in emission_prob. If it is present, we are considering the POS which has the highest probability value for that word. If the word is not present, we are naively assuming the word is a noun. 

### HMM_Viterbi 
For implementing the hidden markov model using viterbi algorithm, we have made use of the emission_prob, start_prob and trans_prob dictionaries. For implementing viterbi algorithm, we have first created a table V, where the number of rows is equal to the total number of POS and the number of column is equal to the total words in the input sentence. The structure of table V is given below, filled with dummy values.

|             | Word_1      | Word_2      | ...         | Word_3      |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| adj         | 0.00078     | 0.00078     | 0.00078     | 0.00078     |
| adv         | 0.00078     | 0.00078     | 0.00078     | 0.00078     |
| adp         | 0.00078     | 0.00078     | 0.00078     | 0.00078     |
| conj        | 0.00078     | 0.00078     | 0.00078     | 0.00078     |
| det         | 0.00078     | 0.00078     | 0.00078     | 0.00078     |
| noun        | 0.00078     | 0.00078     | 0.00078     | 0.00078     |
| num         | 0.00078     | 0.00078     | 0.00078     | 0.00078     |
| pron        | 0.00078     | 0.00078     | 0.00078     | 0.00078     |
| prt         | 0.00078     | 0.00078     | 0.00078     | 0.00078     |
| verb        | 0.00078     | 0.00078     | 0.00078     | 0.00078     |
| x           | 0.00078     | 0.00078     | 0.00078     | 0.00078     |
| .           | 0.00078     | 0.00078     | 0.00078     | 0.00078     |

Now we start filling up this table. For the 1st column, we are considering the value of start_prob[current POS] * emission_prob[current word][current POS]. We select the maximum value out of the column and tag that corresponding POS to the word. Now we move over to the second column. For filling up the second column, we are considering the value of max(values of column_1) * trans_prob[POS of prev word][current POS] * emission_prob[current word][current POS]. We keep tagging the words with the POS for which the value is maximum, and then use this POS for the next columns. This implementation takes into account that the current and past POS's affect the future POS's. After the table is filled, we have our list of tagged POS for all the words in the sentence.

### Complex_MCMC
As we know that under Gibbs sampling we start with a propasal distribution and draw subsequent samples which are biased towards the stationary distribution,

The calculations for this part occur in the file gibbs.py

We begin with the computation of emission probabilities by recording the number of times a given word is observed for a particular POS on the training data and divide that by the number of times that POS occurs in the corpus

Similarly we calculate 2 transition matrices one for next POS probability and one for next to next POS probability calculation

The matrices are calculated in train function

Under function run_gibbs the actual gibbs sampling code occurs

We start with a random permutation of the topics. In each iteration and for each word we set the min_val parameter to inf, this is the negative log of the probability of selecting a particular POS for this word

The actual probability calculation given the complex bayes net structure is shown below

```
if w_idx>=2:
    val = -np.log(self.tr_prob[mtx[w_idx-1]][p])-np.log(val_mtx[w_idx-1])-np.log(self.tr_prob2[mtx[w_idx-2]][p])-np.log(val_mtx[w_idx-2])-np.log(emp)-np.log(emp_prev)
elif w_idx==1:
    val = -np.log(self.tr_prob[mtx[w_idx-1]][p])-np.log(val_mtx[w_idx-1])-np.log(emp)-np.log(emp_prev)
elif w_idx==0:
    val = -np.log(emp)
if val < min_val:
    min_val=val
    best_p=p

```

Since we are taking the negative log we can add up all the transition probabilities and actuals from the previous states depending upon our current position as well as add the emission probabilities. We use negative probabilities as the values are too small

We also added a stopping condition wherein for every 50 iterations for the last 20 iterations the standard deviation is less than 1e-5

# Part 2: Reading text
We are performing Optical Character Recognition using Hidden Markov Model in this part.

### Dataset 
We have been provided a training image - courier-train.png and a training file bc.train from Part 1 which generally represents characters from english language. We have also been provided test images to evaluate the results.

### Hidden Markov Model  
HMMs are statistical models to capture hidden information from observable sequential symbols (e.g., a nucleotidic sequence). The MArkov Model consists of hidden states and  observable states, where the system essentially flows through the states across time. The challenge is to determine the hidden parameters from the observable parameters.

### Procedure - 

i) Pre-processing data: 
We are taking a default character width and height of 14 and 25 px respectively. In the supplied code yo load the letters, we map each of the characters in that image sample to their english language characters. We've further flatten out the image to a single array and represent each pixel by a 0(white) or 1 (black) Now we open bc.train file to load the train file and remove the POS tags rejoin the text to form sentences

ii) Training: After cleaning the dataset we calculate the initial and the transition probabilities of each characters. We also load up courier train image file and represent it with 1s and 0s as this will at as our reference for the prediction.

iii)Naive Bayes Classifier: We implement a simple naive bayes classifier as mentioned in the question, we assume a probability of the pixels being correct as 90%. We assume a small prior of 1/num classes and multiply number of correct pixels with m and  number of incorrect pixels with 1-m.

The final implementation is as follows
```
def nb_classifier(mtx,train_mtx,classes):
    m=0.9
    NUM_CLASSES = len(classes)
    score = {}
    for c in range(NUM_CLASSES):
        log_prob = np.log(1/NUM_CLASSES)+(np.sum(np.equal(mtx[0,:],train_mtx[c,:]))*np.log(m))+(np.sum(np.not_equal(mtx[0,:],train_mtx[c,:]))*np.log(1-m))
        score[classes[c]]=-log_prob
    return score
```

iv) Simple and Viterbi HMM: For the simple HMM we just use the raw Naive Bayes Classifier Probabilities on each word and predict the character with max probability.

For Viterbi Decoding we record a matrix of values we start with the initial probabilities for each class, For each subsequent layer for each class we multply the transition probabilty from the previous layer and the actual value at that layer, as per the algorithm we select the maximum such (minimum in implementation since we are taking negative log) and multiply with emission probability (add in implementation)

This will generate the final matrix we select index of minimum value for all the columns and remap that to each character and we have our final results.

