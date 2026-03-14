from pathlib import Path
import yaml

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Sequence, NFD, Lowercase, StripAccents
from tokenizers.decoders import BPEDecoder


CONFIG_PATH = Path("configs/base_config.yaml")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    raw_text_path = Path(config["data"]["raw_text_path"])
    tokenizer_dir = Path(config["data"]["tokenizer_dir"])
    vocab_size = config["tokenizer"]["vocab_size"]
    min_frequency = config["tokenizer"]["min_frequency"]
    special_tokens = config["tokenizer"]["special_tokens"]

    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    if not raw_text_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {raw_text_path}")

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    tokenizer.normalizer = Sequence([
        NFD(),
        Lowercase(),
        StripAccents()
    ])

    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.decoder = BPEDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens
    )

    tokenizer.train([str(raw_text_path)], trainer)

    tokenizer_json_path = tokenizer_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_json_path))

    print(f"Tokenizer saved to: {tokenizer_json_path}")
    print(f"Vocab size learned: {tokenizer.get_vocab_size()}")

    sample_text = "Transformers are powerful models for language understanding."
    encoded = tokenizer.encode(sample_text)

    print("\n=== Tokenizer Test ===")
    print("Original text:", sample_text)
    print("Tokens:", encoded.tokens)
    print("IDs:", encoded.ids)

    decoded = tokenizer.decode(encoded.ids)
    print("Decoded text:", decoded)


if __name__ == "__main__":
    main()