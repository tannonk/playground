


from transformers import GPT2Tokenizer
from transformers import GPT2Model, GPT2Config,GPT2LMHeadModel
from transformers import TextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from nlp import load_dataset

model_dir = "/srv/scratch6/kew/lm_data/rrgen_de/huggingface/"
tokenizer_path = model_dir+"tokenizer-de_rrgen.json"

breakpoint()
tokenizer = GPT2Tokenizer.from_pretrained(model_dir, max_len=512)

# Initializing a GPT2 configuration
configuration = GPT2Config(vocab_size=tokenizer.vocab_size)
model = GPT2LMHeadModel(config=configuration)


dset = load_dataset("text", data_files="/srv/scratch6/kew/lm_data/rrgen_de/raw/train.rev_resp")["train"]


# dataset = TextDataset(
#     tokenizer=tokenizer,
#     file_path="./deu-de_web-public_2019_1M-sentences.txt",
#     block_size=128,
# )

# data_collator = DataCollatorForLanguageModeling(
#     tokenizer=tokenizer, mlm=False,
# )


# training_args = TrainingArguments(
#     output_dir="./output",
#     overwrite_output_dir=True,
#     num_train_epochs=1,
#     per_gpu_train_batch_size=64,
#     save_steps=10_000,
#     save_total_limit=2,
# )
