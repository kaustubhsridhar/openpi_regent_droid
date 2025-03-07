import sentencepiece
import openpi.shared.download as download
import json

path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
with path.open("rb") as f:
    paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

# Print vocabulary size
vocab_size = paligemma_tokenizer.get_piece_size()
print(f"Vocabulary size: {vocab_size}")

# Print all tokens in the vocabulary
print("\nVocabulary:")
vocabulary = {}
for i in range(vocab_size):
    piece = paligemma_tokenizer.id_to_piece(i)
    # Get the score (log probability) of the piece
    score = paligemma_tokenizer.get_score(i)
    # Print token ID, token text, and score
    vocabulary[i] = piece

with open("scripts/print_vocab.json", "w") as f:
    json.dump(vocabulary, f, indent=4)
