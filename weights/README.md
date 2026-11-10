# External model weights

Place complete local Hugging Face snapshots of `ncbi/MedCPT-Article-Encoder` and `ncbi/MedCPT-Query-Encoder` in `MedCPT-Article-Encoder/` and `MedCPT-Query-Encoder/`, respectively. Include model configuration, model parameters, and tokenizer assets. The source uses local-only loading.

The Article encoder supplies frozen clinical-text features and initializes TGL embeddings. The Query encoder supplies frozen disease-query features. Query features are cached in memory at startup; the query encoder can then be released.

Signal and image encoders initialize from scratch when their YAML `checkpoint_path` values are empty. Stage transitions load the preceding LaUER artifact through `--init-from`.

No pretrained or trained weights are distributed in this directory. Checkpoints produced before the LaUER naming revision require explicit migration of parameter names and configuration metadata.
