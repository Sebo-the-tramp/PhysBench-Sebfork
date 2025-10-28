from eval.models.qa_model.model_library.cambrian.train.train_fsdp import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
