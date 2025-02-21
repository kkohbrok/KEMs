#![cfg_attr(not(test), no_std)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![doc = include_str!("../README.md")]
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/RustCrypto/meta/master/logo.svg",
    html_favicon_url = "https://raw.githubusercontent.com/RustCrypto/meta/master/logo.svg"
)]
#![deny(missing_docs)]
#![warn(clippy::pedantic)]

//! # Usage
//!
//! This crate implements the X-Wing Key Encapsulation Method (X-Wing-KEM) algorithm.
//! X-Wing-KEM is a KEM in the sense that it creates an (decapsulation key, encapsulation key) pair,
//! such that anyone can use the encapsulation key to establish a shared key with the holder of the
//! decapsulation key. X-Wing-KEM is a general-purpose hybrid post-quantum KEM, combining p384 and ML-KEM-1024.
//!
//! ```
//! use kem::{Decapsulate, Encapsulate};
//!
//! let mut rng = &mut rand::rngs::OsRng;
//! let (sk, pk) = x_wing::generate_key_pair(rng);
//! let (ct, ss_sender) = pk.encapsulate(rng).unwrap();
//! let ss_receiver = sk.decapsulate(&ct).unwrap();
//! assert_eq!(ss_sender, ss_receiver);
//! ```

use core::convert::Infallible;

use kem::{Decapsulate, Encapsulate};
use ml_kem::array::ArrayN;
use ml_kem::{
    kem, EncapsulateDeterministic, EncodedSizeUser, KemCore, MlKem1024, MlKem1024Params, B32,
};
use p384::elliptic_curve::sec1::{FromEncodedPoint, ToEncodedPoint};
use p384::elliptic_curve::{NonZeroScalar, PublicKey};
use p384::{AffinePoint, NistP384, ProjectivePoint, U384};
use rand_core::CryptoRngCore;
#[cfg(feature = "getrandom")]
use rand_core::OsRng;
use sha3::digest::core_api::XofReaderCoreWrapper;
use sha3::digest::{ExtendableOutput, XofReader};
use sha3::{Sha3_384, Shake256, Shake256ReaderCore};
#[cfg(feature = "zeroize")]
use zeroize::{Zeroize, ZeroizeOnDrop};

type MlKem1024DecapsulationKey = kem::DecapsulationKey<MlKem1024Params>;
type MlKem1024EncapsulationKey = kem::EncapsulationKey<MlKem1024Params>;

const X_WING_LABEL: &[u8; 6] = br"\.//^\";

const MLKEM_ENCAP_KEY_SIZE: usize = 1568;
const MLKEM_CIPHERTEXT_SIZE: usize = MLKEM_ENCAP_KEY_SIZE;
const P384_PK_KEY_SIZE: usize = 97;
const P384_SK_KEY_SIZE: usize = 48;
/// Size in bytes of the `EncapsulationKey`.
pub const ENCAPSULATION_KEY_SIZE: usize = MLKEM_ENCAP_KEY_SIZE + P384_PK_KEY_SIZE;
/// Size in bytes of the `DecapsulationKey`.
pub const DECAPSULATION_KEY_SIZE: usize = 64;
/// Size in bytes of the `Ciphertext`.
pub const CIPHERTEXT_SIZE: usize = MLKEM_CIPHERTEXT_SIZE + P384_PK_KEY_SIZE;

const MLKEM_ENCAP_RANDOMNESS_SIZE: usize = 32;
const P384_ENCAP_RANDOMNESS_SIZE: usize = 48;
/// Size of the random bytes required for encapsulation.
pub const ENCAP_RANDOMNESS_SIZE: usize = MLKEM_ENCAP_RANDOMNESS_SIZE + P384_ENCAP_RANDOMNESS_SIZE;

/// Shared secret key.
pub type SharedSecret = [u8; 48];

// The naming convention of variables matches the RFC.
// ss -> Shared Secret
// ct -> Cipher Text
// ek -> Ephemeral Key
// pk -> Public Key
// sk -> Secret Key
// Postfixes:
// _m -> ML-Kem related key
// _x -> x25519 related key

/// X-Wing encapsulation or public key.
#[derive(Clone, PartialEq)]
pub struct EncapsulationKey {
    pk_m: MlKem1024EncapsulationKey,
    pk_x: p384::PublicKey,
}

impl EncapsulationKey {
    /// Encapsulate using the given randomness.
    pub fn encapsulate_derand(
        &self,
        randomness: [u8; ENCAP_RANDOMNESS_SIZE],
    ) -> Result<(Ciphertext, SharedSecret), Infallible> {
        let ml_kem_randomness = randomness[0..MLKEM_ENCAP_RANDOMNESS_SIZE]
            .try_into()
            .unwrap();
        let p384_randomness: [u8; P384_ENCAP_RANDOMNESS_SIZE] = randomness
            [MLKEM_ENCAP_RANDOMNESS_SIZE..ENCAP_RANDOMNESS_SIZE]
            .try_into()
            .unwrap();

        let (ct_m, ss_m) = self.pk_m.encapsulate_deterministic(ml_kem_randomness)?;
        let ek_x = p384_randomness;
        self.encapsulate_internal(ct_m, ss_m, ek_x)
    }

    fn encapsulate_internal(
        &self,
        ct_m: ArrayN<u8, MLKEM_CIPHERTEXT_SIZE>,
        ss_m: B32,
        ek_x: SharedSecret,
    ) -> Result<(Ciphertext, SharedSecret), Infallible> {
        let ek_x_scalar = derive_p384_scalar(&ek_x);

        let ct_x_point = ProjectivePoint::GENERATOR * *ek_x_scalar;
        let ct_x_affine = ct_x_point.to_affine();
        let ct_x = ct_x_affine.to_encoded_point(false);

        assert!(ct_x.as_bytes().len() == P384_PK_KEY_SIZE);

        let ct_x = <[u8; P384_PK_KEY_SIZE]>::try_from(ct_x.as_bytes()).unwrap();

        let decoded =
            AffinePoint::from_encoded_point(&p384::EncodedPoint::from_bytes(&ct_x).unwrap())
                .into_option();
        assert!(decoded.is_some(), "Failed to decode ct_x");

        let pk_x_affine = AffinePoint::from(*self.pk_x.as_affine());
        let ss_x_point = pk_x_affine * *ek_x_scalar;
        let ss_x_affine = ss_x_point.to_affine();
        let ss_x = ss_x_affine.to_encoded_point(false);

        let ss = combiner(&ss_m, ss_x.as_bytes(), &ct_x, &self.pk_x);

        #[cfg(feature = "zeroize")]
        {
            let mut ss_x = ss_x;
            ss_x.zeroize();
        }

        let ct = Ciphertext { ct_m, ct_x };
        Ok((ct, ss))
    }
}

impl Encapsulate<Ciphertext, SharedSecret> for EncapsulationKey {
    type Error = Infallible;

    fn encapsulate(
        &self,
        rng: &mut impl CryptoRngCore,
    ) -> Result<(Ciphertext, SharedSecret), Self::Error> {
        // Swapped order of operations compared to RFC, so that usage of the rng matches the RFC
        let (ct_m, ss_m) = self.pk_m.encapsulate(rng)?;

        let ek_x: SharedSecret = generate(rng);
        self.encapsulate_internal(ct_m, ss_m, ek_x)
    }
}

impl EncapsulationKey {
    /// Convert the key to the following format:
    /// ML-KEM-1024 public key(1184 bytes) | X25519 public key(32 bytes).
    #[must_use]
    pub fn as_bytes(&self) -> [u8; ENCAPSULATION_KEY_SIZE] {
        let mut buffer = [0u8; ENCAPSULATION_KEY_SIZE];
        buffer[0..MLKEM_ENCAP_KEY_SIZE].copy_from_slice(&self.pk_m.as_bytes());
        buffer[MLKEM_ENCAP_KEY_SIZE..ENCAPSULATION_KEY_SIZE]
            .copy_from_slice(self.pk_x.to_encoded_point(false).as_bytes());
        buffer
    }
}

impl From<&[u8; ENCAPSULATION_KEY_SIZE]> for EncapsulationKey {
    fn from(value: &[u8; ENCAPSULATION_KEY_SIZE]) -> Self {
        let mut pk_m = [0; MLKEM_ENCAP_KEY_SIZE];
        pk_m.copy_from_slice(&value[0..MLKEM_ENCAP_KEY_SIZE]);
        let pk_m = MlKem1024EncapsulationKey::from_bytes(&pk_m.into());

        let mut pk_x = [0; P384_PK_KEY_SIZE];
        pk_x.copy_from_slice(&value[MLKEM_ENCAP_KEY_SIZE..]);
        let pk_x = PublicKey::from_sec1_bytes(&pk_x).unwrap();
        EncapsulationKey { pk_m, pk_x }
    }
}

/// X-Wing decapsulation key or private key.
#[derive(Clone)]
#[cfg_attr(feature = "zeroize", derive(Zeroize, ZeroizeOnDrop))]
#[cfg_attr(test, derive(PartialEq, Eq))]
pub struct DecapsulationKey {
    sk: [u8; DECAPSULATION_KEY_SIZE],
}

impl Decapsulate<Ciphertext, SharedSecret> for DecapsulationKey {
    type Error = Infallible;

    #[allow(clippy::similar_names)] // So we can use the names as in the RFC
    fn decapsulate(&self, ct: &Ciphertext) -> Result<SharedSecret, Self::Error> {
        let (sk_m, sk_x, _pk_m, pk_x) = self.expand_key();
        let ss_m = sk_m.decapsulate(&ct.ct_m)?;

        let received_ct_x =
            AffinePoint::from_encoded_point(&p384::EncodedPoint::from_bytes(&ct.ct_x).unwrap())
                .unwrap();

        let sk_x_scalar = sk_x.to_nonzero_scalar();

        let ss_x_point = received_ct_x * *sk_x_scalar;
        let ss_x_affine = ss_x_point.to_affine();

        // Step 4: Extract x-coordinate (32 bytes) as the shared secret
        let ss_x = ss_x_affine.to_encoded_point(false);

        // Step 5: Combine secrets
        let ss = combiner(&ss_m, ss_x.as_bytes(), &ct.ct_x, &pk_x);

        #[cfg(feature = "zeroize")]
        {
            let mut ss_x = ss_x;
            ss_x.zeroize();
        }

        Ok(ss)
    }
}

impl DecapsulationKey {
    /// Generate a new `DecapsulationKey` using `OsRng`.
    #[cfg(feature = "getrandom")]
    pub fn generate_from_os_rng() -> DecapsulationKey {
        Self::generate(&mut OsRng)
    }

    /// Generate a new `DecapsulationKey` using the provided RNG.
    pub fn generate(rng: &mut impl CryptoRngCore) -> DecapsulationKey {
        let sk = generate(rng);
        DecapsulationKey { sk }
    }

    /// Provide the matching `EncapsulationKey`.
    #[must_use]
    pub fn encapsulation_key(&self) -> EncapsulationKey {
        let (_sk_m, _sk_x, pk_m, pk_x) = self.expand_key();
        EncapsulationKey { pk_m, pk_x }
    }

    fn expand_key(
        &self,
    ) -> (
        MlKem1024DecapsulationKey,
        p384::SecretKey,
        MlKem1024EncapsulationKey,
        p384::PublicKey,
    ) {
        use sha3::digest::Update;
        let mut shaker = Shake256::default();
        shaker.update(&self.sk);
        let mut expanded = shaker.finalize_xof();

        let d = read_from(&mut expanded).into();
        let z = read_from(&mut expanded).into();
        let (sk_m, pk_m) = MlKem1024::generate_deterministic(&d, &z);

        let sk_x: [u8; P384_SK_KEY_SIZE] = read_from(&mut expanded);
        let sk_x = NonZeroScalar::<NistP384>::from_uint(U384::from_be_slice(&sk_x)).unwrap();
        let sk_x = p384::SecretKey::from(sk_x);
        let pk_x = sk_x.public_key();

        #[cfg(test)]
        {
            println!("sk_x (expand_key): {:?}", sk_x.to_bytes());
            println!(
                "pk_x (expand_key): {:?}",
                pk_x.to_encoded_point(false).as_bytes()
            );
        }

        (sk_m, sk_x, pk_m, pk_x)
    }

    /// Private key as bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; DECAPSULATION_KEY_SIZE] {
        &self.sk
    }
}

impl From<[u8; DECAPSULATION_KEY_SIZE]> for DecapsulationKey {
    fn from(sk: [u8; DECAPSULATION_KEY_SIZE]) -> Self {
        DecapsulationKey { sk }
    }
}

/// X-Wing ciphertext.
#[derive(Clone, PartialEq, Eq)]
#[cfg_attr(feature = "zeroize", derive(Zeroize, ZeroizeOnDrop))]
pub struct Ciphertext {
    ct_m: ArrayN<u8, MLKEM_CIPHERTEXT_SIZE>,
    ct_x: [u8; P384_PK_KEY_SIZE],
}

impl Ciphertext {
    /// Convert the ciphertext to the following format:
    /// ML-KEM-1024 ciphertext(1568 bytes) | p384 ciphertext(97 bytes).
    #[must_use]
    pub fn as_bytes(&self) -> [u8; CIPHERTEXT_SIZE] {
        let mut buffer = [0; CIPHERTEXT_SIZE];
        buffer[0..MLKEM_CIPHERTEXT_SIZE].copy_from_slice(&self.ct_m);
        buffer[MLKEM_CIPHERTEXT_SIZE..].copy_from_slice(&self.ct_x);
        buffer
    }
}

impl From<&[u8; CIPHERTEXT_SIZE]> for Ciphertext {
    fn from(value: &[u8; CIPHERTEXT_SIZE]) -> Self {
        let mut ct_m = [0; MLKEM_CIPHERTEXT_SIZE];
        ct_m.copy_from_slice(&value[0..MLKEM_CIPHERTEXT_SIZE]);
        let mut ct_x = [0; P384_PK_KEY_SIZE];
        ct_x.copy_from_slice(&value[MLKEM_CIPHERTEXT_SIZE..]);

        Ciphertext {
            ct_m: ct_m.into(),
            ct_x,
        }
    }
}

/// Generate a X-Wing key pair using `OsRng`.
#[cfg(feature = "getrandom")]
pub fn generate_key_pair_from_os_rng() -> (DecapsulationKey, EncapsulationKey) {
    generate_key_pair(&mut OsRng)
}

/// Generate a X-Wing key pair using the provided rng.
pub fn generate_key_pair(rng: &mut impl CryptoRngCore) -> (DecapsulationKey, EncapsulationKey) {
    let sk = DecapsulationKey::generate(rng);
    let pk = sk.encapsulation_key();
    (sk, pk)
}

/// Generate a X-Wing key pair using the provided random bytes.
pub fn generate_key_pair_derand(
    randomness: [u8; DECAPSULATION_KEY_SIZE],
) -> (DecapsulationKey, EncapsulationKey) {
    let sk = DecapsulationKey { sk: randomness };
    let pk = sk.encapsulation_key();
    (sk, pk)
}

fn derive_p384_scalar(ek_x: &[u8; 48]) -> NonZeroScalar<NistP384> {
    use sha3::digest::Update;
    let mut hasher = Shake256::default();
    hasher.update(ek_x);
    let mut reader = hasher.finalize_xof();

    let mut expanded = [0u8; 48];
    reader.read(&mut expanded);

    let u_384 = U384::from_be_slice(&expanded);
    NonZeroScalar::<NistP384>::from_uint(u_384).unwrap()
}

fn combiner(ss_m: &[u8], ss_x: &[u8], ct_x: &[u8], pk_x: &p384::PublicKey) -> SharedSecret {
    use sha3::Digest;

    let mut hasher = Sha3_384::new();
    hasher.update(ss_m);
    hasher.update(ss_x);
    hasher.update(ct_x);
    hasher.update(pk_x.to_encoded_point(false).as_bytes());
    hasher.update(X_WING_LABEL);
    hasher.finalize().into()
}

fn read_from<const N: usize>(reader: &mut XofReaderCoreWrapper<Shake256ReaderCore>) -> [u8; N] {
    let mut data = [0; N];
    reader.read(&mut data);
    data
}

fn generate<const N: usize>(rng: &mut impl CryptoRngCore) -> [u8; N] {
    let mut random = [0; N];
    rng.fill_bytes(&mut random);
    random
}

#[cfg(test)]
mod tests {
    use rand_core::OsRng;

    use super::*;

    #[test]
    fn ciphertext_serialize() {
        let mut rng = OsRng;

        let ct_a = Ciphertext {
            ct_m: generate(&mut rng).into(),
            ct_x: generate(&mut rng),
        };

        let bytes = ct_a.as_bytes();

        let ct_b = Ciphertext::from(&bytes);

        assert!(ct_a == ct_b);
    }

    #[test]
    fn key_serialize() {
        let sk = DecapsulationKey::generate(&mut OsRng);
        let pk = sk.encapsulation_key();

        let sk_bytes = sk.as_bytes();
        let pk_bytes = pk.as_bytes();

        let sk_b = DecapsulationKey::from(*sk_bytes);
        let pk_b = EncapsulationKey::from(&pk_bytes.clone());

        assert!(sk == sk_b);
        assert!(pk == pk_b);
    }

    #[test]
    fn encap_decap() {
        let mut rng = OsRng;

        let (sk, pk) = generate_key_pair(&mut rng);
        let (ct, ss_sender) = pk.encapsulate(&mut rng).unwrap();
        let ss_receiver = sk.decapsulate(&ct).unwrap();
        assert_eq!(ss_sender, ss_receiver);
    }
}
