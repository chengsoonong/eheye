### Streaming data and quantile estimation

1. Overall intro
   - The field of quantile estimation is big
   - this chapter details some of the work that has been done in the quantile estimation on streaming data, organised as follows:
   - details some of the current quantile estimation methods on the streaming data phenomenon, which are classified wrt algorithm types(?), while section 3.2 and 3.3 classify quantile estimation problems by aims
   - discuss algorithms on single quantile estimation
   - present the relatively new problem: algorithms on multi-quantile estimation

#### Streaming data and quantile estimation
 1. data stream
    - definition
    - **problems**
    <!-- - models & maths? -->
 2. quantile estimations
    1. why quantile/estimation/data stream
    2. quant est
      - intro: trade off
      - our intro to current works
 3. deterministic algos
    - MRL
    - GK
    - q_digest
    - Agarwal: mergeable summaries
 4. randomised algos
    <!-- - Vitter: Random Sampling with a Reservoir -->
    - Manku: MRL99
    - Count-min sketch
    - A Randomized Online Quantile Summary in O((1/ε)log(1/ε)) Words
 5. other algos --> next two subsections
    - Biased estimate
    - Sliding windows: Manku: Approximate counts and quantiles over sliding windows
    - Multi-thread data stream: SPDT

#### Single Quantile estimation with memory limitaion
1. Frugal
2. Yaz

#### Multi-quantile estimation

our algorithm:
    - online learning
    - one pass
    - constant memory usage
    - im tired QAQ

our work:
    - frugal == online learning
    - How it works for distributions/data size/ data sequence(?)/step size(?) 
    - single quantile: step size (based on Yazidi & Hammer's work & Frugal)
    - multi-quantile: step size & relations
    - deal with real data?
    - who to compare with?