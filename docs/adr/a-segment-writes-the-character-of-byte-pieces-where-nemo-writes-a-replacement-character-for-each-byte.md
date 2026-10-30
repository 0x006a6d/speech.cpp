# A segment writes the character of byte pieces, where NeMo writes a replacement character for each byte

## Context

A FastConformer model whose SentencePiece tokenizer has byte fallback, Haas-ja-v1.0, writes a character outside its
vocabulary as byte pieces: 癌 as `<0xE7>`, `<0x99>` and `<0x8C>`. NeMo decodes the transcript's ids together, which gives
the character, but builds a segment's text from words (`get_words_offsets()` in timestamp_utils.py, NeMo 2.7.3): a token
whose piece differs from its text decoded alone starts a word, so each byte piece is a word of its own, decoded alone to
U+FFFD, and the words are joined with spaces. Its segment then reads 「細胞死や � � �の原因」 where its transcript reads
「細胞死や癌の原因」.

## Decision

**A segment's text is the text its tokens make, as the transcript writes it**: the bytes of consecutive byte pieces are
read as UTF-8 and the character goes to the token of its last byte. The tokens' times, the word starts and the segments'
spans and times stay NeMo's: a byte piece starts a word as NeMo's rule makes it. The times check takes a segment whose
text differs from NeMo's only in byte pieces each written alone.

The alternative was turned down:

- NeMo's segment text, U+FFFD for each byte joined with spaces. It would give a caller of timestamps a text unlike the
  transcript's, and a replacement character where the model recognized a character.

## Consequences

On Haas-ja's dumps of 5 Japanese FLEURS utterances, the 2 with byte pieces give segments whose texts differ from NeMo's
in those characters alone, with every time equal to NeMo's.
