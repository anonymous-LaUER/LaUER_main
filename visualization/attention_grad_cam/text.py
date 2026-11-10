from scipy.io import loadmat

def sample_text(sample, separator, override=None):
    if sample.target_text or override:
        return sample.target_text or override
    if sample.dataset.lower() != 'meeti' or not sample.signal_path:
        raise ValueError('Long-text alignment requires MEETI MAT or explicit target-text')
    data = loadmat(sample.signal_path, variable_names=['report', 'LLM_Interpretation'])
    parts = [str(data[k][0]).strip() if k in data else ''
             for k in ('report', 'LLM_Interpretation')]
    if not all(parts):
        raise ValueError('Missing sample report or LLM_Interpretation')
    return separator.join(parts)
