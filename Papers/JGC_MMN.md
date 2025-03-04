# 1. basic intro

1. Complex nature of influencing factors and sparse transaction records

2. Model target: urban subregion house price prediction from city-level to mile-level

3. Learn all-level features and capture spatiotemporal correlations in all-time stages

4. Devise a novel JGC based fusion method to better fuse
   the heterogeneous data of multi- stage models by considering their interactions in temporal dimension

5. Existing networks focusing on prediction individual house price which is highly depend on the data quality.

# 2. main contributions

1. Use densely connected networks to capture the all-level features in order to overcome the sparsity challenge
   and alleviate the corresponding overfitting.

2. Consider more well-selected factors, including current ingredients and future price-growth expectations, as the submodules of prediction.

3. Propose a novel multi-modal framework by fusing multiple learners on the different temporal characteristics
   (i.e., long-term periodicity, recent tendency, current, and future periods) for depicting spatiotemporal dependencies.

4. Design a new method, JGC, to learn the correlations between them automatically by generating joint attention flows
   within various modalities and filtrating noises of multiple similar modalities with the gated function.

# 3. basic definition

1. City-Region definition City can be devided into different suquares with d0 as their side-length.
   Require no less than 10 transaction records in a single area in a specific month(30x30).

2. House price set with specific month $T$ and corresponding region index, price set of a month can represented in a tensor like ixjx1.

3. subregion house price prediction: given a historical dataset from 1 to n, the model can predicted any subregion house price given the month $n+1$
   RMSE is used to measure the accuracy.

# 4. basic stucture

1. Long-term spatiotemporal Densenet: input shape(ixjx5), as long term refers to length of 5 years.

2. Short-term spatiotemporal Densenet: input shape(ixjx12), as short term refers to 12 months.

3. current ingredients layer: a fcnn.

4. Future expectations layer: use the Kalman Filter to model the subjective expectations from the public.

# 5. notations in paper

1. $S^T$ represents the house transaction price set of the entire city of month $T$

2. $m_r$: number of rows, $m_c$: number of columns
