#![feature(portable_simd)]

use std::simd::{Simd, StdFloat}; // Changed from SimdFloat to StdFloat

pub use crate::weights::{Lang, LANGUAGES};

#[allow(clippy::all)]
mod weights;

const NUM_LANGUAGES: usize = LANGUAGES.len();
pub const DIMENSION: usize = 1 << 12;
const BIGRAM_MASK: u32 = (1 << 16) - 1;
const TRIGRAM_MASK: u32 = (1 << 24) - 1;
const CHUNK_SIZE: usize = 16; // Increased SIMD width

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Feature {
    AsciiNGram(u32),
    Unicode(char),
    UnicodeClass(char),
}

const SEED: u32 = 3_242_157_231u32;

#[inline(always)]
fn murmurhash2(mut k: u32, seed: u32) -> u32 {
    const M: u32 = 0x5bd1_e995;
    let mut h = seed;
    k = k.wrapping_mul(M);
    k ^= k >> 24;
    k = k.wrapping_mul(M);
    h = h.wrapping_mul(M);
    h ^= k;
    h ^= h >> 13;
    h = h.wrapping_mul(M);
    h ^ (h >> 15)
}

impl Feature {
    #[inline(always)]
    pub fn to_hash(&self) -> u32 {
        match *self {
            Feature::AsciiNGram(ngram) => murmurhash2(ngram, SEED),
            Feature::Unicode(chr) => murmurhash2((chr as u32) >> 7, SEED ^ 2),
            Feature::UnicodeClass(chr) => murmurhash2(classify_codepoint(chr), SEED ^ 4),
        }
    }
}

#[inline(always)]
fn update_scores(scores: &mut [f32], weight: &[f32]) {
    let simd_len = scores.len() - (scores.len() % CHUNK_SIZE);
    
    for i in (0..simd_len).step_by(CHUNK_SIZE) {
        let s: Simd<f32, CHUNK_SIZE> = Simd::from_slice(&scores[i..]);
        let w: Simd<f32, CHUNK_SIZE> = Simd::from_slice(&weight[i..]);
        (s + w).copy_to_slice(&mut scores[i..]);
    }

    // Process remaining elements with manual loop unrolling
    let mut i = simd_len;
    while i + 3 < scores.len() {
        scores[i] += weight[i];
        scores[i+1] += weight[i+1];
        scores[i+2] += weight[i+2];
        scores[i+3] += weight[i+3];
        i += 4;
    }
    while i < scores.len() {
        scores[i] += weight[i];
        i += 1;
    }
}

#[inline(always)]
fn apply_transform(scores: &mut [f32], intercepts: &[f32], sqrt_inv: f32) {
    let sqrt_inv_simd: Simd<f32, CHUNK_SIZE> = Simd::splat(sqrt_inv);
    let simd_len = scores.len() - (scores.len() % CHUNK_SIZE);

    for i in (0..simd_len).step_by(CHUNK_SIZE) {
        let s: Simd<f32, CHUNK_SIZE> = Simd::from_slice(&scores[i..]);
        let inter: Simd<f32, CHUNK_SIZE> = Simd::from_slice(&intercepts[i..]);
        s.mul_add(sqrt_inv_simd, inter).copy_to_slice(&mut scores[i..]);
    }

    // Process remaining elements with FMA
    for i in simd_len..scores.len() {
        scores[i] = scores[i].mul_add(sqrt_inv, intercepts[i]);
    }
}

pub fn detect_language(text: &str) -> Lang {
    if text.is_empty() {
        return Lang::Eng;
    }

    let mut scores = [0.0; NUM_LANGUAGES];
    let mut num_features = 0u32;
    
    emit_tokens(text, |token| {
        num_features += 1;
        let idx = (token.to_hash() as usize % DIMENSION) * NUM_LANGUAGES;
        update_scores(
            &mut scores, 
            &weights::WEIGHTS[idx..idx + NUM_LANGUAGES]
        );
    });
    
    if num_features == 0 {
        return Lang::Eng;
    }
    
    let sqrt_inv = (num_features as f32).sqrt().recip();
    apply_transform(
        &mut scores, 
        &weights::INTERCEPTS[..NUM_LANGUAGES], 
        sqrt_inv
    );
    
    // Find max using SIMD-friendly reduction
    let mut max_idx = 0;
    let mut max_val = scores[0];
    for i in 1..scores.len() {
        if scores[i] > max_val {
            max_val = scores[i];
            max_idx = i;
        }
    }
    
    LANGUAGES[max_idx]
}

/// Extract language features from text
#[doc(hidden)]
pub fn emit_tokens(text: &str, mut listener: impl FnMut(Feature)) {
    if text.is_ascii() {
        process_ascii_text(text.as_bytes(), &mut listener);
    } else {
        process_mixed_text(text, &mut listener);
    }
}

/// Process ASCII-only text for n-grams
#[inline(always)]
fn process_ascii_text(bytes: &[u8], listener: &mut impl FnMut(Feature)) {
    let mut prev = b' ' as u32;
    let mut num_prev_ascii = 1;
    
    for &b in bytes {
        let code = (b as char).to_ascii_lowercase() as u32;
        prev = (prev << 8) | code;
        process_ascii_char(prev, &mut num_prev_ascii, listener);
        if !(b as char).is_alphanumeric() {
            prev = b' ' as u32;
            num_prev_ascii = 1;
        }
    }
}

/// Process mixed ASCII/non-ASCII text
#[inline(always)]
fn process_mixed_text(text: &str, listener: &mut impl FnMut(Feature)) {
    let mut prev = b' ' as u32;
    let mut num_prev_ascii = 1;
    
    for chr in text.chars() {
        if !chr.is_ascii() {
            listener(Feature::Unicode(chr));
            listener(Feature::UnicodeClass(chr));
            num_prev_ascii = 0;
        } else {
            let code = chr.to_ascii_lowercase() as u32;
            prev = (prev << 8) | code;
            process_ascii_char(prev, &mut num_prev_ascii, listener);
            if !chr.is_alphanumeric() {
                prev = b' ' as u32;
                num_prev_ascii = 1;
            }
        }
    }
}

/// Process ASCII character n-grams
#[inline(always)]
fn process_ascii_char(prev: u32, num_prev_ascii: &mut u8, listener: &mut impl FnMut(Feature)) {
    match *num_prev_ascii {
        0 => *num_prev_ascii = 1,
        1 => {
            listener(Feature::AsciiNGram(prev & BIGRAM_MASK));
            *num_prev_ascii = 2;
        }
        2 => {
            listener(Feature::AsciiNGram(prev & BIGRAM_MASK));
            listener(Feature::AsciiNGram(prev & TRIGRAM_MASK));
            *num_prev_ascii = 3;
        }
        _ => {
            listener(Feature::AsciiNGram(prev & BIGRAM_MASK));
            listener(Feature::AsciiNGram(prev & TRIGRAM_MASK));
            listener(Feature::AsciiNGram(prev));
        }
    }
}

// Japanese and Chinese character ranges
const JP_PUNCT_START: u32 = 0x3000;
const JP_PUNCT_END: u32 = 0x303f;
const JP_HIRAGANA_START: u32 = 0x3040;
const JP_HIRAGANA_END: u32 = 0x309f;
const JP_KATAKANA_START: u32 = 0x30a0;
const JP_KATAKANA_END: u32 = 0x30ff;
const CJK_KANJI_START: u32 = 0x4e00;
const CJK_KANJI_END: u32 = 0x9faf;
const JP_HALFWIDTH_KATAKANA_START: u32 = 0xff61;
const JP_HALFWIDTH_KATAKANA_END: u32 = 0xff90;

static CLASSIFICATION_POINTS: [u32; 52] = [
    160, 161, 171, 172, 173, 174, 187, 192, 196, 199, 200, 201, 202, 205, 214, 220, 223, 224, 225,
    226, 227, 228, 231, 232, 233, 234, 235, 236, 237, 238, 239, 242, 243, 244, 245, 246, 249, 250,
    251, 252, 333, 339, JP_PUNCT_START, JP_PUNCT_END, JP_HIRAGANA_START, JP_HIRAGANA_END,
    JP_KATAKANA_START, JP_KATAKANA_END, CJK_KANJI_START, CJK_KANJI_END,
    JP_HALFWIDTH_KATAKANA_START, JP_HALFWIDTH_KATAKANA_END,
];

/// Classify a Unicode character into a specific category
#[inline(always)]
fn classify_codepoint(chr: char) -> u32 {
    CLASSIFICATION_POINTS.binary_search(&(chr as u32)).unwrap_or_else(|pos| pos) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ascii_ngram_feature(text: &str) -> Feature {
        let mut bytes = [0; 4];
        bytes[4 - text.len()..].copy_from_slice(text.as_bytes());
        Feature::AsciiNGram(u32::from_be_bytes(bytes))
    }

    #[test]
    fn test_emit_tokens() {
        let mut tokens = Vec::new();
        emit_tokens("hello　こん！", |token| tokens.push(token));
        assert_eq!(
            tokens,
            vec![
                ascii_ngram_feature(" h"),
                ascii_ngram_feature("he"),
                ascii_ngram_feature(" he"),
                ascii_ngram_feature("el"),
                ascii_ngram_feature("hel"),
                ascii_ngram_feature(" hel"),
                ascii_ngram_feature("ll"),
                ascii_ngram_feature("ell"),
                ascii_ngram_feature("hell"),
                ascii_ngram_feature("lo"),
                ascii_ngram_feature("llo"),
                ascii_ngram_feature("ello"),
                Feature::Unicode('　'),
                Feature::UnicodeClass('　'),
                Feature::Unicode('こ'),
                Feature::UnicodeClass('こ'),
                Feature::Unicode('ん'),
                Feature::UnicodeClass('ん'),
                Feature::Unicode('！'),
                Feature::UnicodeClass('！'),
            ]
        );
    }

    #[test]
    fn test_empty_str() {
        assert_eq!(detect_language(""), Lang::Eng);
    }

    #[test]
    fn test_detect_language() {
        assert_eq!(detect_language("Hello, happy tax payer"), Lang::Eng);
        assert_eq!(detect_language("Bonjour joyeux contribuable"), Lang::Fra);
        assert_eq!(detect_language("Hallo glücklicher Steuerzahler"), Lang::Deu);
        assert_eq!(detect_language("こんにちは幸せな税金納め"), Lang::Jpn);
        assert_eq!(detect_language("你好幸福的纳税人"), Lang::Cmn);
        assert_eq!(detect_language("Merhaba, mutlu vergi mükellefi"), Lang::Tur);
        assert_eq!(detect_language("Hallo, blije belastingbetaler"), Lang::Nld);
        assert_eq!(detect_language("안녕하세요 행복한 납세자입니다"), Lang::Kor);
        assert_eq!(detect_language("Ciao, felice contribuente!"), Lang::Ita);
        assert_eq!(detect_language("Hola feliz contribuyente"), Lang::Spa);
        assert_eq!(detect_language("¡Hola!"), Lang::Spa);
        assert_eq!(detect_language("Olá feliz contribuinte"), Lang::Por);
    }
}