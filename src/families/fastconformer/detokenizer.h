#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "model-file.h"

namespace fastconformer {

/**
 * The text of SentencePiece ids as NeMo's decoding writes it: SentencePieceProcessor.DecodeIds(), then one
 * whitespace removed before each punctuation mark of the vocabulary
 * (AbstractRNNTDecoding.decode_tokens_to_str_with_strip_punctuation()). With byte fallback, the bytes of consecutive
 * byte pieces are read as UTF-8, each byte that begins no whole character standing for U+FFFD, as DecodeIds() reads them.
 */
class Detokenizer {
public:
    explicit Detokenizer(const ModelFile & m);

    std::string text(const std::vector<int32_t> & ids) const;

    /**
     * The text of each token as it stands in text(ids), which they make when joined: a whitespace the decoding
     * removes before a punctuation mark is taken from the token that holds it, and a character of byte pieces belongs
     * to the token of its last byte, so that each token's text is whole UTF-8.
     */
    std::vector<std::string> token_texts(const std::vector<int32_t> & ids) const;

    /**
     * Whether the token, decoded alone, is one of the vocabulary's punctuation marks, as compute_rnnt_timestamps()
     * tests it: decode_ids_to_str([id]) in supported_punctuation.
     */
    bool punctuation(int32_t id) const;

    /**
     * Whether each token begins a word as get_words_offsets() tells it for a SentencePiece vocabulary: its piece
     * differs from its text decoded alone, as a piece with a leading "▁" and the unknown piece do, and it is not a
     * punctuation mark, which NeMo adds to the word before it.
     */
    std::vector<bool> word_starts(const std::vector<int32_t> & ids) const;

private:
    /** The piece of `id` as DecodeIds() writes it when the text before it is `empty` or not. */
    std::string piece_text(int32_t id, bool empty) const;

    std::vector<std::string> pieces_;
    /** The byte each piece stands for, or -1 for a piece that is no byte piece. */
    std::vector<int> bytes_;
    std::vector<std::string> punctuation_;
    int unknown_id_;
    std::string unknown_surface_;
    bool strip_leading_space_;
};

}  // namespace fastconformer
