//! Minimal, dependency-free SHA-256 implementation used to verify package
//! cache content. Only the small API the cache layer needs is exposed.

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

#[derive(Clone)]
pub(crate) struct Sha256 {
    state: [u32; 8],
    buffer: [u8; 64],
    buffer_len: usize,
    total_len: u64,
}

impl Sha256 {
    pub(crate) fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            buffer: [0; 64],
            buffer_len: 0,
            total_len: 0,
        }
    }

    pub(crate) fn update(&mut self, mut data: &[u8]) {
        self.total_len = self.total_len.wrapping_add(data.len() as u64);
        if self.buffer_len > 0 {
            let needed = 64 - self.buffer_len;
            if data.len() < needed {
                self.buffer[self.buffer_len..self.buffer_len + data.len()].copy_from_slice(data);
                self.buffer_len += data.len();
                return;
            }
            self.buffer[self.buffer_len..].copy_from_slice(&data[..needed]);
            data = &data[needed..];
            self.buffer_len = 0;
            self.compress(self.buffer);
        }
        while data.len() >= 64 {
            let block = &data[..64];
            self.compress(block.try_into().unwrap());
            data = &data[64..];
        }
        if !data.is_empty() {
            self.buffer[..data.len()].copy_from_slice(data);
            self.buffer_len = data.len();
        }
    }

    pub(crate) fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.total_len.wrapping_mul(8);
        self.buffer[self.buffer_len] = 0x80;
        for byte in &mut self.buffer[self.buffer_len + 1..] {
            *byte = 0;
        }
        if self.buffer_len + 1 > 56 {
            self.compress(self.buffer);
            self.buffer = [0; 64];
        }
        self.buffer[56..].copy_from_slice(&bit_len.to_be_bytes());
        self.compress(self.buffer);
        let mut digest = [0u8; 32];
        for (word, chunk) in self.state.iter().zip(digest.chunks_mut(4)) {
            chunk.copy_from_slice(&word.to_be_bytes());
        }
        digest
    }

    fn compress(&mut self, block: [u8; 64]) {
        let mut words = [0u32; 64];
        for (word, chunk) in words[..16].iter_mut().zip(block.chunks(4)) {
            *word = u32::from_be_bytes(chunk.try_into().unwrap());
        }
        for index in 16..64 {
            let small = words[index - 15].rotate_right(7) ^ words[index - 15].rotate_right(18)
                ^ (words[index - 15] >> 3);
            let large = words[index - 2].rotate_right(17) ^ words[index - 2].rotate_right(19)
                ^ (words[index - 2] >> 10);
            words[index] = words[index - 16]
                .wrapping_add(small)
                .wrapping_add(words[index - 7])
                .wrapping_add(large);
        }
        let mut working = self.state;
        for (index, word) in words.iter().enumerate() {
            let choice = (working[4] & working[5]) ^ ((!working[4]) & working[6]);
            let majority = (working[0] & working[1]) ^ (working[0] & working[2])
                ^ (working[1] & working[2]);
            let sigma1 =
                working[4].rotate_right(6) ^ working[4].rotate_right(11) ^ working[4].rotate_right(25);
            let sigma0 =
                working[0].rotate_right(2) ^ working[0].rotate_right(13) ^ working[0].rotate_right(22);
            let temp1 = working[7]
                .wrapping_add(choice)
                .wrapping_add(sigma1)
                .wrapping_add(K[index])
                .wrapping_add(*word);
            let temp2 = sigma0.wrapping_add(majority);
            working[7] = working[6];
            working[6] = working[5];
            working[5] = working[4];
            working[4] = working[3].wrapping_add(temp1);
            working[3] = working[2];
            working[2] = working[1];
            working[1] = working[0];
            working[0] = temp1.wrapping_add(temp2);
        }
        for (state, value) in self.state.iter_mut().zip(working) {
            *state = state.wrapping_add(value);
        }
    }
}

pub(crate) fn digest_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        encoded.push_str(&format!("{byte:02x}"));
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::digest_hex;

    #[test]
    fn matches_known_sha256_vectors() {
        assert_eq!(
            digest_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            digest_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            digest_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn handles_chunks_across_block_boundaries() {
        let input = b"pklr package cache content verification".repeat(10);
        let full = digest_hex(&input);
        let mut hasher = super::Sha256::new();
        for (index, byte) in input.iter().enumerate() {
            hasher.update(&[*byte]);
            assert_eq!(index + 1 > 0, true);
        }
        let mut one_by_one = String::new();
        for byte in hasher.finalize() {
            one_by_one.push_str(&format!("{byte:02x}"));
        }
        assert_eq!(full, one_by_one);
    }
}
