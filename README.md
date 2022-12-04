# Part 1: Part-of-speech tagging

## Training
We have used the training corpus provided to us to find the following dictionaries: 
- POS_count : Stores the number of occurences of each POS in the training corpus.
- total_count : Stores the number of occurences of each unique word in the training corpus.
- emission_prob : Stores the emission probabilites of a POS given a particular word.  
- start_prob : Stores the initial starting probabilities of each POS.
- trans_prob : Stores the transition probabilities of each POS given a previous POS.

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


# Part 2: Reading text
