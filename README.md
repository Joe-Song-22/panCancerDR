panCancerDR: A Domain Generalization Framework for Drug Response Prediction
panCancerDR is a novel domain generalization framework designed to predict drug response in out-of-distribution samples, including individual cells and patient data, using only in vitro cancer cell line data for training. By leveraging adversarial domain generalization and innovative feature extraction techniques, panCancerDR addresses the limitations of traditional domain adaptation methods, which are unsuitable for unseen target domains.
Key Features
Latent Independent Projection (LIP): A plug-and-play module that extracts expressive and decorrelated features to mitigate overfitting and improve generalization across diverse datasets.
Asymmetric Adaptive Clustering Constraint: Ensures sensitive samples are tightly clustered, while resistant samples form diverse clusters in the latent space, reflecting real-world biological diversity in drug responses.
Comprehensive Validation: The model has been rigorously evaluated on bulk RNA-seq, single-cell RNA-seq (scRNA-seq), and patient-level datasets, demonstrating superior or comparable performance to current state-of-the-art (SOTA) methods.
Contributions
Novel LIP Module: Encourages the encoder to extract informative, non-redundant features, enabling robust predictions across unseen domains.
Asymmetric Clustering for Generalization: Inspired by the distinct characteristics of sensitive and resistant cells, this approach ensures effective latent space representation for drug response prediction.
Extensive Evaluation: Achieved high performance on 13 cancer types, single-cell data, and patient-level drug response tasks, highlighting the model's versatility and predictive power.
# panCancerDR
