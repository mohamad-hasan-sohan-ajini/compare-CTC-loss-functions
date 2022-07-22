# compare-CTC-loss-functions
To compare CTC loss functions:

1. Pytorch Native CTC loss
2. CUDNN CTC loss
3. warp-ctc CTC loss

So this repo consists of codes to produce equivalent results using these functions.

# Conclusion

1. WarpCTC filed to compile with new cuda version (and GPU with high computational capability).
2. Using pytorch native CTC loss usually causes the network to converge faster, compared using the CUDNN CTC loss.
3. The `reduction` function of CTC loss has a significant effect on the convergence speed:
    a) Comparing reductions: Best results usually achieved using `sum` reduction. Although using `sum` reduction cause the loss start from enormous values (say xe+4), so may be unstable in some cases.
    b) Using `mean` reduction in CUDNN loss may cause the network not to converge at all!
