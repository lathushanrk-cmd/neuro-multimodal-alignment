# Orthogonal Procrustes

Orthogonal Procrustes estimates a rotation/reflection of one representation toward another. In NMA, each training fold independently fits scalers and PCA to both modalities, estimates an orthogonal transform, then reports held-out root-mean-square error (RMSE). `improvement_over_baseline` compares this RMSE against predicting zero in *training-centered and standardized* target space. Neither small error nor a recovered rotation establishes biological equivalence. Component signs/rotations depend on the training representation.
