use clap::Parser;
use rayon::prelude::*;
use curve25519_dalek::scalar::Scalar;
use curve25519_dalek::edwards::CompressedEdwardsY;
use curve25519_dalek::constants::ED25519_BASEPOINT_POINT;
use rand::rngs::OsRng;
use rand::RngCore;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use sha2::{Sha512, Digest};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "")] prefix: String,
    #[arg(long, default_value = "")] suffix: String,
    #[arg(long, default_value = "false")] case_sensitive: String,
}

fn main() {
    let args = Args::parse();
    let prefix = args.prefix;
    let suffix = args.suffix;
    let case = args.case_sensitive.to_lowercase() == "true";
    let found = Arc::new(AtomicBool::new(false));

    let cpu_count = num_cpus::get();

    rayon::scope(|s| {
        for _ in 0..cpu_count {
            let found_clone = found.clone();
            let prefix_clone = prefix.clone();
            let suffix_clone = suffix.clone();

            s.spawn(move |_| {
                let mut rng = OsRng::default();
                let mut attempts = 0;

                loop {
                    if found_clone.load(Ordering::Relaxed) {
                        break;
                    }

                    // 1. Generate a fresh seed
                    let mut seed = [0u8; 32];
                    rng.fill_bytes(&mut seed);

                    // 2. Expand Seed manually to get Scalar (Ref10 logic)
                    // h = SHA512(seed)
                    let h = Sha512::digest(&seed);

                    // s = h[0..32], clamped
                    let mut scalar_bytes = [0u8; 32];
                    scalar_bytes.copy_from_slice(&h[0..32]);
                    scalar_bytes[0] &= 248;
                    scalar_bytes[31] &= 127;
                    scalar_bytes[31] |= 64;

                    let mut current_scalar = Scalar::from_bits(scalar_bytes);

                    // 3. Compute Initial Point from Scalar
                    // P = a * B
                    let mut current_point = &current_scalar * &ED25519_BASEPOINT_POINT;

                    // 4. Offset Loop (Grinding)
                    for _ in 0..10000 {
                         // Increment Scalar and Point
                         // P' = P + B
                         // s' = s + 1
                        current_point = current_point + ED25519_BASEPOINT_POINT;
                        current_scalar = current_scalar + Scalar::one();

                        // Compress point to bytes
                        let pk_bytes = current_point.compress().to_bytes();

                        // Encode to Base58
                        let pk_b58 = bs58::encode(pk_bytes).into_string();

                        let p_match = if case { pk_b58.starts_with(&prefix_clone) } else { pk_b58.to_lowercase().starts_with(&prefix_clone.to_lowercase()) };

                        if p_match {
                            let s_match = if case { pk_b58.ends_with(&suffix_clone) } else { pk_b58.to_lowercase().ends_with(&suffix_clone.to_lowercase()) };

                            if s_match {
                                if !found_clone.swap(true, Ordering::Relaxed) {
                                    // Found a match!
                                    // Construct 64-byte keypair: [scalar(32), public(32)]
                                    let mut final_keypair_bytes = [0u8; 64];
                                    final_keypair_bytes[0..32].copy_from_slice(current_scalar.as_bytes());
                                    final_keypair_bytes[32..64].copy_from_slice(&pk_bytes);

                                    let final_b58 = bs58::encode(final_keypair_bytes).into_string();

                                    // MATCH:<PUBKEY>:<FULL_KEYPAIR>
                                    println!("MATCH:{}:{}", pk_b58, final_b58);
                                }
                                return;
                            }
                        }
                    }

                    attempts += 1;
                    if attempts % 100 == 0 {
                        if found_clone.load(Ordering::Relaxed) { break; }
                    }
                }
            });
        }
    });
}
