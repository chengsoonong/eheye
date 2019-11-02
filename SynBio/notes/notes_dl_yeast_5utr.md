### Some takeaway points from [Deep learning of the regulatory grammar of yeast 5′ untranslated regions from 500,000 random sequences](https://genome.cshlp.org/content/27/12/2015.long)

1. linear regression model with position-dependent 3-mer features (4^3 × 48 = 3072 distinct features; R^2 = 0.42) outperformed models with more complex but position-independent features (e.g., 6-mer model; 4^6 = 4096
features; R^2 = 0.33) (Supplemental Fig. S3A,B)
2. the position-sensitive linear regression model
was still unable to capture more complex features, such as uORFs
or secondary structure. When features capturing this information
were added to the model, the performance was further improved
(R^2 = 0.52) (Supplemental Fig. S3C; Methods; Dvir et al. 2013).
On the other hand, CNNs can capture not only position depen-
dence but also nonlinear interactions between features. Since
they do so in an unsupervised fashion, they can also potentially
draw attention to unappreciated elements.
3. a corresponding decrease in the predictive power of our models as
the training size decreased (Supplemental Fig. S4C)
4. We also found
that inclusion of the entire 50 nt of sequence was necessary for
the high predictive capacity of the model, since a CNN trained using only the 10 nt adjacent to the start codon performed poorly
(R^2 = 0.097) (Supplemental Fig. S4D).
5. a CNN trained
only on this native library performed worse on both the native and
random library data sets, most likely due to the limited size of the
training set (training set = 9492 sequences; R 2 = 0.47 native test set;
R 2 = 0.30 random test set) (Supplemental Fig. S6). As in the case
of our model trained on random sequences, a CNN trained on
the native sequences using only the 10 nt proximal to the start
codon also performed poorly (R 2 = 0.14) (Supplemental Fig. S6).
6.  the first layer of the model learned uAUG motif variants,
while eight learned motifs with stop codons (UAG/UGA/UAA)
(Fig. 2B; Supplemental Fig. S5); ome of
the higher layers combine uAUG and stop codon filters to learn
the concept of a uORF, as evidenced by the model predicting
much lower protein expression for 5 ′ UTR sequences containing a
uORF (see Fig. 2A)

encoding methods:
1. one-hot encoding  
2. k-mer models. e.g. a 3-mer
model, there are 64 possible 3-mer sequences and 48 positions,
leading to 3072 model weights; 3072 = 43 × 48


