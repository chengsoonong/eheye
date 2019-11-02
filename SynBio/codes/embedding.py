from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 2nd Nov 2019
# Mengyan Zhang

class Embedding():
    """Embedding for biological sequences.
    """
    def __init__(data):
        """
        Parameters
        --------------------------------------
        data: ndarray 
            {A, G, C, T} ^ {num_seq x num_bases}
        """
        self.data = data
        self.num_seq, self.num_bases = data.shape

        self.alphabet = ['A','C','G','T']
        self.label_encoder.fit(np.array(self.alphabet))

    def onehot_encoder():
        """One-hot embedding.
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

    def kmer():
        """k-merization embedding.
           See https://en.wikipedia.org/wiki/K-mer.
           Tutorial (https://www.kaggle.com/thomasnelson/working-with-dna-sequence-data-for-ml#Machine-learning-with-DNA-sequence-data
        """


    def PMF():
        """Position weight matrix.
            See https://davetang.org/muse/2013/10/01/position-weight-matrix/
        """

    def spectrum_kernel():




