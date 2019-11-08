import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# 2nd Nov 2019
# Mengyan Zhang

class Embedding():
    """Embedding for biological sequences.
    """
    def __init__(self, data):
        """
        Parameters
        --------------------------------------
        data: ndarray 
            (num_seq, ) (each element is a biological sequence)
        """
        self.data = data
        # self.num_seq, self.num_bases = data.shape
        self.num_seq,  = data.shape
        self.num_bases = len(data[0])

    def onehot(self):
        """One-hot embedding.

        Returns
        --------------------------------------------
        embedded_data: ndarray
            {0, 1}^{num_seq x num_bases * 4}
        """
        bases = ['A','C','G','T']
        base_dict = dict(zip(bases,range(4))) # {'A' : 0, 'C' : 1, 'G' : 2, 'T' : 3}

        embedded_data = np.zeros((self.num_seq, self.num_bases * 4))

        # loop through the array of sequences to create a feature array 
        for i in range(self.num_seq):
            seq = self.data[i]
            # loop through each individual sequence, from the 5' to 3' end
            for b in range(self.num_bases):
                embedded_data[i, b * 4 + base_dict[seq[b]]] = 1
        return embedded_data

    def kmer(self, size = 3):
        """k-merization embedding.
           See https://en.wikipedia.org/wiki/K-mer.
           Tutorial (https://www.kaggle.com/thomasnelson/working-with-dna-sequence-data-for-ml#Machine-learning-with-DNA-sequence-data

        Parameters
        --------------------------------------------------------------------
        size: int 
            len of unit of words
        """

        sentences = []

        for i in range(self.num_seq):
            sequence = self.data[i][:8]
            words = [sequence[x:x+size] for x in range(len(sequence) - size + 1)]
            words.append(self.data[i][-1])
            sentence = ' '.join(words)
            sentences.append(sentence)

        cv = CountVectorizer()
        embedded = cv.fit_transform(sentences).toarray()

        return embedded

    def PMF(self):
        """Position weight matrix.
            See https://davetang.org/muse/2013/10/01/position-weight-matrix/
        """

    def spectrum_kernel(self):
        """Spectrum kernel
        """
        
