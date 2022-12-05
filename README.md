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

Since we are taking the negative log we can add up all the transition probabilities and actuals from te previous states depending upon our current position as well as add the emission probabilities. We use negative probabilities as the values are too small

We also added a stopping condition wherein for every 50 iterations for the last 20 iterations the standard deviation is less than 1e-5

# Part 2: Reading text
We are performing Optical Character Recognition using Hidden Markov Model in this part.

### Dataset 
We have been provided a training image - courier-train.png and a training file bc.train from Part 1 which generally represents characters from english language. We have also been provided test images to evaluate the results.

### Hidden Markov Model  
HMMs are statistical models to capture hidden information from observable sequential symbols (e.g., a nucleotidic sequence). They have many applications in sequence analysis, in particular to predict exons and introns in genomic DNA, identify functional motifs (domains) in proteins (profile HMM), align two sequences (pair HMM). In a HMM, the system being modelled is assumed to be a Markov process with unknown parameters, and the challenge is to determine the hidden parameters from the observable parameters.

### Procedure - 

i] Pre-processing data: We are taking a default character width and height of 14 and 25 px respectively. To load the letters, we divide the image by character width and and then map each of the characters in that image sample to their english language characters. Now we open bc.train file to load the train file and split them into different lines. Then, we check if each character in the word is alphabet or number and ignore the special characters to create a clean training file.

ii] Training: After cleaning the dataset we calculate the initial and the transition probabilities of each characters. For the train and test data we create matrix with character "*" at positions where characters are found. Now, use Naive Bayes classifier to test the score of our test image by passing the train and test matrix and the classes of the training dataset. 
