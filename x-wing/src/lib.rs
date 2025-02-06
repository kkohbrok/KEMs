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
//! decapsulation key. X-Wing-KEM is a general-purpose hybrid post-quantum KEM, combining x25519 and ML-KEM-768.
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
    kem, EncapsulateDeterministic, EncodedSizeUser, KemCore, MlKem768, MlKem768Params, B32,
};
use p256::elliptic_curve::sec1::{FromEncodedPoint, ToEncodedPoint};
use p256::elliptic_curve::{NonZeroScalar, PublicKey};
use p256::{AffinePoint, NistP256, ProjectivePoint, U256};
use rand_core::CryptoRngCore;
#[cfg(feature = "getrandom")]
use rand_core::OsRng;
use sha3::digest::core_api::XofReaderCoreWrapper;
use sha3::digest::{ExtendableOutput, XofReader};
use sha3::{Sha3_256, Shake256, Shake256ReaderCore};
#[cfg(feature = "zeroize")]
use zeroize::{Zeroize, ZeroizeOnDrop};

type MlKem768DecapsulationKey = kem::DecapsulationKey<MlKem768Params>;
type MlKem768EncapsulationKey = kem::EncapsulationKey<MlKem768Params>;

const X_WING_LABEL: &[u8; 6] = br"\.//^\";

/// Size in bytes of the `EncapsulationKey`.
pub const ENCAPSULATION_KEY_SIZE: usize = 1184 + 65;
/// Size in bytes of the `DecapsulationKey`.
pub const DECAPSULATION_KEY_SIZE: usize = 32;
/// Size in bytes of the `Ciphertext`.
pub const CIPHERTEXT_SIZE: usize = 1088 + 65;

/// Shared secret key.
pub type SharedSecret = [u8; 32];

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
    pk_m: MlKem768EncapsulationKey,
    pk_x: p256::PublicKey,
}

impl EncapsulationKey {
    /// Encapsulate using the given randomness.
    pub fn encapsulate_derand(
        &self,
        randomness: [u8; 64],
    ) -> Result<(Ciphertext, SharedSecret), Infallible> {
        let ml_kem_randomness = randomness[0..32].try_into().unwrap();
        let p256_randomness = randomness[32..64].try_into().unwrap();

        let (ct_m, ss_m) = self.pk_m.encapsulate_deterministic(ml_kem_randomness)?;
        let ek_x = p256_randomness;
        self.encapsulate_internal(ct_m, ss_m, ek_x)
    }

    fn encapsulate_internal(
        &self,
        ct_m: ArrayN<u8, 1088>,
        ss_m: B32,
        ek_x: SharedSecret,
    ) -> Result<(Ciphertext, SharedSecret), Infallible> {
        let u_256 = U256::from_be_slice(&ek_x);
        let ek_x_scalar = NonZeroScalar::<NistP256>::from_uint(u_256).unwrap();

        let ct_x_point = ProjectivePoint::GENERATOR * *ek_x_scalar;
        let ct_x_affine = ct_x_point.to_affine();
        let ct_x = ct_x_affine.to_encoded_point(false);

        assert!(ct_x.as_bytes().len() == 65);

        let ct_x = <[u8; 65]>::try_from(ct_x.as_bytes()).unwrap();

        let decoded =
            AffinePoint::from_encoded_point(&p256::EncodedPoint::from_bytes(&ct_x).unwrap())
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
    /// ML-KEM-768 public key(1184 bytes) | X25519 public key(32 bytes).
    #[must_use]
    pub fn as_bytes(&self) -> [u8; ENCAPSULATION_KEY_SIZE] {
        let mut buffer = [0u8; ENCAPSULATION_KEY_SIZE];
        buffer[0..1184].copy_from_slice(&self.pk_m.as_bytes());
        buffer[1184..1216].copy_from_slice(self.pk_x.to_encoded_point(false).as_bytes());
        buffer
    }
}

impl From<&[u8; ENCAPSULATION_KEY_SIZE]> for EncapsulationKey {
    fn from(value: &[u8; ENCAPSULATION_KEY_SIZE]) -> Self {
        let mut pk_m = [0; 1184];
        pk_m.copy_from_slice(&value[0..1184]);
        let pk_m = MlKem768EncapsulationKey::from_bytes(&pk_m.into());

        let mut pk_x = [0; 65];
        pk_x.copy_from_slice(&value[1184..]);
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
            AffinePoint::from_encoded_point(&p256::EncodedPoint::from_bytes(&ct.ct_x).unwrap())
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
        MlKem768DecapsulationKey,
        p256::SecretKey,
        MlKem768EncapsulationKey,
        p256::PublicKey,
    ) {
        use sha3::digest::Update;
        let mut shaker = Shake256::default();
        shaker.update(&self.sk);
        let mut expanded = shaker.finalize_xof();

        let d = read_from(&mut expanded).into();
        let z = read_from(&mut expanded).into();
        let (sk_m, pk_m) = MlKem768::generate_deterministic(&d, &z);

        let sk_x: [u8; 32] = read_from(&mut expanded);
        let sk_x = NonZeroScalar::<NistP256>::from_uint(U256::from_be_slice(&sk_x)).unwrap();
        let sk_x = p256::SecretKey::from(sk_x);
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
    ct_m: ArrayN<u8, 1088>,
    ct_x: [u8; 65],
}

impl Ciphertext {
    /// Convert the ciphertext to the following format:
    /// ML-KEM-768 ciphertext(1088 bytes) | p256 ciphertext(32 bytes).
    #[must_use]
    pub fn as_bytes(&self) -> [u8; CIPHERTEXT_SIZE] {
        let mut buffer = [0; CIPHERTEXT_SIZE];
        buffer[0..1088].copy_from_slice(&self.ct_m);
        buffer[1088..].copy_from_slice(&self.ct_x);
        buffer
    }
}

impl From<&[u8; CIPHERTEXT_SIZE]> for Ciphertext {
    fn from(value: &[u8; CIPHERTEXT_SIZE]) -> Self {
        let mut ct_m = [0; 1088];
        ct_m.copy_from_slice(&value[0..1088]);
        let mut ct_x = [0; 65];
        ct_x.copy_from_slice(&value[1088..]);

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

fn combiner(ss_m: &[u8], ss_x: &[u8], ct_x: &[u8], pk_x: &p256::PublicKey) -> SharedSecret {
    use sha3::Digest;

    let mut hasher = Sha3_256::new();
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
