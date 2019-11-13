import numpy as np
from collections import defaultdict, OrderedDict
import operator
from codes.embedding import Embedding

class Rewards_env():
    """Rewards environment for biology sequence.

    Attributes
    -------------------------------------------------
    rewards_dict: dict of list
        keys: string of embedded sequence 
            e.g. '001000100010001000100010'
        values: list of available labels
    labels_dict: dict of list
        keys: string of embedded sequence 
            e.g. '001000100010001000100010'
        values: label (expectation of rewards list)
    """
    def __init__(self, data, embedding_method = 'onehot'):
        """
        Parameters
        ---------------------------------------------
        data: ndarray 
            num_data * 2
            two columns: biology sequence; score (label)
        embedding_method: string
            specify embedding method
            default is 'onehot'
        """
        self.data = data
        self.num_seq, self.num_col = self.data.shape
        assert self.num_col == 2
        self.embedding_method = embedding_method

        self.embedded = self.arm_embedding()

        self.rewards_dict = defaultdict(list)
        self.create_rewards_dict()
        self.labels_dict = self.labels_generator()
        self.arm_features = self.sort_arms()
        
    def sort_arms(self):
        self.labels_dict = OrderedDict(sorted(self.labels_dict.items(), key=operator.itemgetter(1)))
        return list(self.labels_dict.keys())
        
    def arm_embedding(self):
        """Embedding biological sequence.
        """
        # encoder = Embedding(self.data[:, 0].reshape(self.num_seq, 1))
        encoder = Embedding(self.data[:, 0])
        if self.embedding_method == 'label':
            embedded= encoder.label()
        elif self.embedding_method == 'onehot':
            embedded = encoder.onehot()
        elif self.embedding_method == 'kmer':
            embedded = encoder.kmer()
        else:
            print('To be added.')
            embedded = self.data[:, 0]
        return embedded

    def create_rewards_dict(self):
        """Create reward dict based on the embedded features.
        """
        for i, d in enumerate(self.data):
            key = self.to_string(self.embedded[i])
            self.rewards_dict[key].append(d[1])

        # sort the rewards dict by keys
        self.rewards_dict = dict(sorted(self.rewards_dict.items()))

    def labels_generator(self):
        """Generate labels for each arm.

        Return 
        -----------------------------------------
        labels_dict: dict of list
            keys: string of biology sequence
                e.g. 'AGCTAA'
            values: label (expectation of rewards list)
        """
        labels_dict = {}
        for key, value in self.rewards_dict.items():
            labels_dict[key] = np.mean(value)
        return labels_dict

    def sample(self, arm):
        """Sample the arm with idx. 

        Parameters
        ----------------------------------------
        idx: int
            arm index
        arm: arm features
            e.g. '001010001000100000101000'
        
        Returns
        ----------------------------------------
        sample for the selected arm
        """
        # key = list(self.rewards_dict.keys())[idx]
        samples = self.rewards_dict[arm]

        return np.random.choice(samples)

 #----------------------------------------------------------               
    def to_string(self, code):
        """decoding arms: from one-hot encoding to strings
        
        code: list of one hot encoding
        
        Return: string
        """
        return ''.join(str(int(e)) for e in code)   