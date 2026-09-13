"""Measure released GSM8K prompt lengths without loading model weights."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics


MODEL = 'Qwen/Qwen2.5-Math-7B'
MODEL_REVISION = 'b101308fe89651ea5ce025f25317fea6fc07e96e'
DATA_REVISION = '740312add88f781978c0658806c59bc2815b9866'
PROMPT_HASHES = {
    'original': 'db8c84b10707a16cc2f61e192e1e55452393510470a3d5790db86188a42ae4e0',
    'complex': 'd20a8061b484da2d93d10152727c5c137f7bec76f5f4f9e1c94f17cf33051aee',
}
TRAIN_HASH = 'ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(prompt_dir):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    import transformers
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION, local_files_only=True)
    tokenizer_files = {}
    for name in ['tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt', 'config.json']:
        path = Path(hf_hub_download(MODEL, name, revision=MODEL_REVISION, local_files_only=True))
        tokenizer_files[name] = sha(path)
        if name == 'config.json':
            model_context = json.loads(path.read_text())['max_position_embeddings']
    train_path = Path(hf_hub_download('openai/gsm8k', 'main/train-00000-of-00001.parquet',
                                    repo_type='dataset', revision=DATA_REVISION, local_files_only=True))
    assert sha(train_path) == TRAIN_HASH
    questions = pq.read_table(train_path, columns=['question']).column('question').to_pylist()
    assert len(questions) == 7473
    banks = {'zero': ''}
    for name, expected in PROMPT_HASHES.items():
        path = prompt_dir/f'prompt_{name}.txt'
        assert sha(path) == expected
        banks[name] = path.read_text() + '\n'
    summary = {}
    for name, prefix in banks.items():
        prompts = [prefix + 'Question: ' + q + "\nLet's think step by step\n" for q in questions]
        lengths = [len(ids) for ids in tokenizer(prompts, add_special_tokens=False,
                                               truncation=False)['input_ids']]
        summary[name] = {'questions': len(lengths), 'minimum_prompt_tokens': min(lengths),
                         'median_prompt_tokens': statistics.median(lengths),
                         'maximum_prompt_tokens': max(lengths),
                         'maximum_total_with_1024_answer_tokens': max(lengths)+1024,
                         'exceeds_4096_with_1024_answer_tokens': sum(n+1024 > 4096 for n in lengths),
                         'lengths_sha256': hashlib.sha256(json.dumps(lengths).encode()).hexdigest()}
    return {'status': 'tokenized_without_inference', 'checked_at': datetime.now(timezone.utc).isoformat(),
            'model': MODEL, 'model_revision': MODEL_REVISION, 'model_context_tokens': model_context,
            'transformers_version': transformers.__version__, 'tokenizer_files_sha256': tokenizer_files,
            'dataset_revision': DATA_REVISION, 'train_file_sha256': TRAIN_HASH,
            'prompt_files_sha256': PROMPT_HASHES, 'script_sha256': sha(Path(__file__)),
            'query_template': "Question: {question}\nLet's think step by step\n",
            'separator': 'One additional newline after the unchanged released prompt file.',
            'add_special_tokens': False, 'chat_template_applied': False, 'summary': summary,
            'generation_rows': 0, 'weights_loaded': False, 'sample_selected': False,
            'limitations': 'Length check over all training question texts, including previously used and '
                          'reserved questions. No answer labels read, no inference, no eligible-pool claim, '
                          'and no new experiment declaration. Query framing is a proposed adaptation.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refuse to overwrite a budget inspection')
    result = inspect(args.prompt_dir)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'status': result['status'], 'summary': result['summary']}))
