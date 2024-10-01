# Make it more memory efficient by monkey patching the LLaMA model with xformers attention.

# Need to call this before importing transformers.
from physbench_eval.models.qa_model.model_library.cambrian.train.llama_xformers_attn_monkey_patch import (
    replace_llama_attn_with_xformers_attn,
)

replace_llama_attn_with_xformers_attn()

from physbench_eval.models.qa_model.model_library.cambrian.train.train_fsdp import train

if __name__ == "__main__":
    train()
