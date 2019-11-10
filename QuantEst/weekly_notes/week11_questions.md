### Week 11 - Questions about the project

1. What are the constraints for the quantile SGD esimation problem? e.g.
    - Only one/two units can be used for the algorithm?
    - Does the code need to be super simple/fast?
    - 
    
2. Evaluation of "quantile estimation works": E = |q_batch - q_est|:
    - Is it possible to normalize the E? for example, some datasets has very densed distribution and some don't
    - Why is E = |rank(q_batch) - rank(q_est)| so hard? Quantiles are all about rankings anyway
    
    
3. How to link anomaly detection with quantile estimation?