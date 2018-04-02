This is a clear version of pointer generator without policy gradient:
1. This code is modified from pointer-generator-pg-func2 by removing the policy gradient part.
2. Attention decoder do one step at once.
3. Attention decoder will store the previous context vector (not need to compute again for next step).
4. This version can get the same redults as See et al. but take less time for training (about 80000 iterations to converge).
