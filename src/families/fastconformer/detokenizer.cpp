#include "detokenizer.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace fastconformer {

namespace {

const std::string kSpaceSymbol = "\xe2\x96\x81";

const std::string kReplacement = "\xef\xbf\xbd";

/** The bytes of a whole UTF-8 character at the start of `b`, or 0 where it begins none. */
size_t utf8_character(const std::string & b) {
    const auto byte = [&](size_t i) { return (unsigned char) b[i]; };
    const auto continuation = [&](size_t i) { return i < b.size() && (byte(i) & 0xc0) == 0x80; };
    const unsigned char c = byte(0);
    if (c < 0x80) return 1;
    // The shortest form alone, and no surrogate or code point past U+10FFFF, as a UTF-8 decoder accepts.
    if (c >= 0xc2 && c <= 0xdf) return continuation(1) ? 2 : 0;
    if (c >= 0xe0 && c <= 0xef) {
        if (!continuation(1) || !continuation(2)) return 0;
        if ((c == 0xe0 && byte(1) < 0xa0) || (c == 0xed && byte(1) >= 0xa0)) return 0;
        return 3;
    }
    if (c >= 0xf0 && c <= 0xf4) {
        if (!continuation(1) || !continuation(2) || !continuation(3)) return 0;
        if ((c == 0xf0 && byte(1) < 0x90) || (c == 0xf4 && byte(1) >= 0x90)) return 0;
        return 4;
    }
    return 0;
}

std::string replace_all(std::string s, const std::string & from, const std::string & to) {
    for (size_t at = s.find(from); at != std::string::npos; at = s.find(from, at + to.size())) s.replace(at, from.size(), to);
    return s;
}

}  // namespace

Detokenizer::Detokenizer(const ModelFile & m)
    : pieces_(m.str_array("fastconformer.tokenizer.tokens")),
      punctuation_(m.str_array("fastconformer.tokenizer.punctuation")),
      unknown_id_((int) m.u32("fastconformer.tokenizer.unknown_id")),
      unknown_surface_(m.str("fastconformer.tokenizer.unknown_surface")),
      strip_leading_space_(m.boolean("fastconformer.tokenizer.strip_leading_space")) {
    bytes_.assign(pieces_.size(), -1);
    if (!m.boolean("fastconformer.tokenizer.byte_fallback")) return;
    // The converter checks that the byte pieces are <0x00> to <0xFF>, in that order.
    for (size_t id = 0; id < pieces_.size(); id++) {
        const std::string & p = pieces_[id];
        if (p.size() == 6 && p.compare(0, 3, "<0x") == 0 && p[5] == '>' && std::isxdigit((unsigned char) p[3]) && std::isxdigit((unsigned char) p[4])) {
            bytes_[id] = std::stoi(p.substr(3, 2), nullptr, 16);
        }
    }
}

std::string Detokenizer::piece_text(int32_t id, bool empty) const {
    if (id < 0 || (size_t) id >= pieces_.size()) throw std::runtime_error("token id " + std::to_string(id) + " is not in the vocabulary");
    if (id == unknown_id_) return unknown_surface_;
    // A byte piece decoded alone is its byte where that is a character, ASCII, and U+FFFD otherwise, as DecodeIds()
    // gives it for one id.
    if (const int byte = bytes_[(size_t) id]; byte >= 0) return byte < 0x80 ? std::string(1, (char) byte) : kReplacement;
    std::string piece = pieces_[(size_t) id];
    // SentencePiece drops the leading "▁" of each piece until the text is no longer empty.
    if (strip_leading_space_ && empty && piece.compare(0, kSpaceSymbol.size(), kSpaceSymbol) == 0) piece.erase(0, kSpaceSymbol.size());
    return replace_all(piece, kSpaceSymbol, " ");
}

std::vector<std::string> Detokenizer::token_texts(const std::vector<int32_t> & ids) const {
    std::vector<std::string> pieces;
    std::string text;
    for (size_t k = 0; k < ids.size();) {
        if (ids[k] < 0 || (size_t) ids[k] >= pieces_.size() || bytes_[(size_t) ids[k]] < 0) {
            pieces.push_back(piece_text(ids[k++], text.empty()));
            text += pieces.back();
            continue;
        }
        // A run of byte pieces: each whole character goes to the piece of its last byte, and each byte that begins
        // none stands for U+FFFD.
        std::string run;
        const size_t first = k;
        for (; k < ids.size() && (size_t) ids[k] < pieces_.size() && bytes_[(size_t) ids[k]] >= 0; k++) run += (char) bytes_[(size_t) ids[k]];
        pieces.resize(k);
        for (size_t at = 0; at < run.size();) {
            const size_t n = utf8_character(run.substr(at));
            pieces[first + at + (n ? n : 1) - 1] += n ? run.substr(at, n) : kReplacement;
            at += n ? n : 1;
        }
        for (size_t i = first; i < k; i++) text += pieces[i];
    }
    // The converter checks that no piece holds whitespace other than "▁" and that the unknown surface holds
    // only spaces, so the space is the one whitespace character the official pattern's \s can meet here.
    std::vector<std::string> out(pieces.size());
    size_t i = 0;
    for (size_t k = 0; k < pieces.size(); k++) {
        for (char c : pieces[k]) {
            bool before_mark = false;
            if (c == ' ') {
                for (const std::string & p : punctuation_) before_mark = before_mark || text.compare(i + 1, p.size(), p) == 0;
            }
            if (!before_mark) out[k] += c;
            i++;
        }
    }
    return out;
}

std::string Detokenizer::text(const std::vector<int32_t> & ids) const {
    std::string text;
    for (const std::string & t : token_texts(ids)) text += t;
    return text;
}

bool Detokenizer::punctuation(int32_t id) const {
    const std::string alone = piece_text(id, true);
    return std::find(punctuation_.begin(), punctuation_.end(), alone) != punctuation_.end();
}

std::vector<bool> Detokenizer::word_starts(const std::vector<int32_t> & ids) const {
    std::vector<bool> starts;
    for (int32_t id : ids) {
        const std::string alone = piece_text(id, true);
        starts.push_back(alone != pieces_[(size_t) id] && !punctuation(id));
    }
    return starts;
}

}  // namespace fastconformer
