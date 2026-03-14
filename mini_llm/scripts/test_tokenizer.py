from pathlib import Path
from tokenizers import Tokenizer

TOKENIZER_PATH = Path("tokenizer/tokenizer.json")


def main():
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_PATH}")

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    text = "Large language models learn patterns from text."
    encoded = tokenizer.encode(text)

    print("Input text:", text)
    print("Tokens:", encoded.tokens)
    print("Token IDs:", encoded.ids)

    decoded = tokenizer.decode(encoded.ids)
    print("Decoded:", decoded)


if __name__ == "__main__":
    main()