# Project conventions

- Use short, simple conventional commit messages.
- Nicole Ma is the sole commit author. Do not add co-author trailers or bot attribution.
- Keep model weights frozen. Separate extraction, validation, and test questions.
- Report negative results, selection rules, and truncated generations.
- Never commit credentials, model weights, or raw activation tensors.
- Do not claim that a text signature proves the model's internal reasoning method.
- The two-H100 node is shared with another project. Use GPU 0 for this repository and run GPU jobs sequentially. Leave GPU 1 available for the other project.
