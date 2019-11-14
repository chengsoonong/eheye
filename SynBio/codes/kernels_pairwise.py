import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

def phi(x, y, l, j_x, j_y, d):
    """Calculate spectrum features for spectrum kernel.

    phi is a mapping of a row of matrix x into a |alphabet|^l
    dimensional feature space. For each sequence in x,
    each dimension corresponds to one of the |alphabet|^l 
    possible strings s of length l and is the count of 
    the number of occurrance of s in x. 

    Paramters
    ---------------------------------------------------
    x : string
        a row of the data matrix
    y : string
        a row of the data matrix
    l : int, default 3
        number of l-mers (length of 'word') 
    j_x : int
        start position of sequence in x
    j_y : int
        start position of sequence in y
    d : int
        the length of analysed sequence
        j + d is end position of sequence 

    Returns
    ----------------------------------------------------
    embedded_x: array
        1 * num_embedded_features
    embedded_y: array
        1 * num_embedded_features
    """

    sentences = []

    sequence_x= x[j_x:j_x + d]
    words_x = [sequence_x[a:a+l] for a in range(len(sequence_x) - l + 1)]
    sentence_x = ' '.join(words_x)
    sentences.append(sentence_x)

    sequence_y= y[j_y:j_y + d]
    words_y = [sequence_y[a:a+l] for a in range(len(sequence_y) - l + 1)]
    sentence_y = ' '.join(words_y)
    sentences.append(sentence_y)
    cv = CountVectorizer(analyzer='word',token_pattern=u"(?u)\\b\\w+\\b")
    #cv = CountVectorizer()
    embedded = cv.fit_transform(sentences).toarray()
    #print(embedded)

    return embedded[0], embedded[1]

def inverse_label(x):
    """convert_to_string
    """
    le = LabelEncoder()
    bases = ['A','C','G','T']
    le.fit(bases)
    
    int_x = []
    for i in x:
        int_x.append(int(i))
    #print(int_x)
    inverse_x = le.inverse_transform(int_x)
    inverse_x = ''.join(e for e in inverse_x)   
    #print(inverse_x)
    
    return inverse_x

def spectrum_kernel_pw(x, y=None, gamma = 1.0, l = 3, j_x = 0, j_y = 0, d = None):
    """
    Compute the spectrum kernel between x and y:
        k_{l}^{spectrum}(x, y) = <phi(x), phi(y)>
    for each pair of rows x in x and y in y.
    when y is None, y is set to be equal to x.

    Parameters
    ----------
    x : string
        a row of the data matrix
    y : string
        a row of the data matrix
    gamma: float, default is 1.
        parameter require by gaussain process kernel.
    l : int, default 3
        number of l-mers (length of 'word')
    j_x : int
        start position of sequence in x
    j_y : int
        start position of sequence in y
    d : int, default None
        if None, set to the length of sequence
        d is the length of analysed sequence
        j + d is end position of sequence 
    Returns
    -------
    kernel_matrix : array of shape (n_samples_x, n_samples_y)
    """
    
    if y is None:
        y = x
    
    x = inverse_label(x)
    y = inverse_label(y)

    if d is None:
        d = len(x) 
    # sequence cannot pass the check 
    # x, y = check_pairwise_arrays(x, y)
    phi_x, phi_y = phi(x, y, l, j_x, j_y, d)
    return phi_x.dot(phi_y.T)

def mixed_spectrum_kernel_pw(x, y=None, gamma = 1.0, l = 3):
    """
    Compute the mixed spectrum kernel between x and y:
        k(x, y) = \sum_{d = 1}^{l} beta_d k_d^{spectrum}(x,y)
    for each pair of rows x in X and y in Y.
    when Y is None, Y is set to be equal to X.

    beta_d = 2 frac{l - d + 1}{l^2 + 1}

    Parameters
    ----------
    X : array of shape (n_samples_X, )
        each row is a sequence (string)
    Y : array of shape (n_samples_Y, )
        each row is a sequence (string)
    gamma: float, default is 1.
        parameter require by gaussain process kernel.
    l : int, default 3
        number of l-mers (length of 'word')
    Returns
    -------
    kernel_matrix : array of shape (n_samples_X, n_samples_Y)
    """
    if y is None:
        y = x

    k = 0

    for d in range(1, l+1):
        #print(d)
        beta = 2 * float(l - d + 1)/float(l ** 2 + 1)
        k += beta * spectrum_kernel_pw(x, y, l = d)
    return k

def WD_kernel_pw(x, y=None, gamma = 1.0, l = 3):
    """Weighted degree kernel.
    Compute the mixed spectrum kernel between x and y:
        k(x, y) = \sum_{d = 1}^{l} \sum_j^{L-d} 
            beta_d k_d^{spectrum}(x[j:j+d],y[j:j+d])
    for each pair of rows x in X and y in Y.
    when Y is None, Y is set to be equal to X.

    beta_d = 2 frac{l - d + 1}{l^2 + 1}

    Parameters
    ----------
    X : array of shape (n_samples_X, )
        each row is a sequence (string)
    Y : array of shape (n_samples_Y, )
        each row is a sequence (string)
    gamma: float, default is 1.
        parameter require by gaussain process kernel.
    l : int, default 3
        number of l-mers (length of 'word')
    Returns
    -------
    kernel_matrix : array of shape (n_samples_X, n_samples_Y)
    """
    if y is None:
        y = x

    k = 0

    # assume all seq has the same total length
    L = len(x)

    for d in range(1, l+1):
        #print(d)
        for j in range(0, L - d + 1):
            beta = 2 * float(l - d + 1)/float(l ** 2 + 1)
            k+= beta * spectrum_kernel_pw(x, y, l = d, j_x = j, j_y = j, d = d)
    return k

def WD_shift_kernel_pw(x, y=None, gamma = 1.0, l = 3, shift_range = 1):
    """Weighted degree kernel with shifts.
    Compute the mixed spectrum kernel between X and Y:
        K(x, y) = \sum_{d = 1}^{l} \sum_j^{L-d} \sum_{s=0 and s+j <= L}
            beta_d * gamma_j * delta_s *
            (k_d^{spectrum}(x[j+s:j+s+d],y[j:j+d]) + k_d^{spectrum}(x[j:j+d],y[j+s:j+s+d]))
    for each pair of rows x in X and y in Y.
    when Y is None, Y is set to be equal to X.

    beta_d = 2 frac{l - d + 1}{l^2 + 1}
    gamma_j = 1
    delta_s = 1/(2(s+1))

    TODO: to confirm why shift useful?
    
    Parameters
    ----------
    X : array of shape (n_samples_X, )
        each row is a sequence (string)
    Y : array of shape (n_samples_Y, )
        each row is a sequence (string)
    gamma: float, default is 1.
        parameter require by gaussain process kernel.
    l : int, default 3
        number of l-mers (length of 'word')
    shift_range: int, default 1
        number of shifting allowed
    Returns
    -------
    kernel_matrix : array of shape (n_samples_X, n_samples_Y)
    """
    if y is None:
        y = x

    k = 0
    
    L = len(x) # assume all seq has the same total length

    for d in range(1, l+1):
        #print(d)
        for j in range(0, L - d + 1):
            for s in range(shift_range+1): # range is right open
                if s + j <= L:
                    beta = 2 * float(l - d + 1)/float(l ** 2 + 1)
                    delta = 1.0/(2 * (s + 1))
                    k += beta * delta * (spectrum_kernel_pw(x, y, l = d, \
                        j_x = j+s, j_y = j,d = d) + \
                        spectrum_kernel_pw(x, y, l = d, j_x = j, j_y = j+s, d= d))
    return k
