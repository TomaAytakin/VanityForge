use clap::Parser;
use rayon::prelude::*;
use ed25519_dalek::{Keypair, SecretKey, PublicKey};
use rand::rngs::OsRng;
use rand::RngCore;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

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

    // Calculate approximate threads to maximize CPU
    let cpu_count = num_cpus::get();

    rayon::scope(|s| {
        for _ in 0..cpu_count {
            let found_clone = found.clone();
            let prefix_clone = prefix.clone();
            let suffix_clone = suffix.clone();

            s.spawn(move |_| {
                // Initialize a random seed per thread
                let mut seed = [0u8; 32];
                let mut rng = OsRng::default();
                rng.fill_bytes(&mut seed);

                let mut attempts = 0;

                loop {
                    if found_clone.load(Ordering::Relaxed) {
                        break;
                    }

                    // Increment seed (Big Endian or Little Endian doesn't matter for randomness)
                    // We increment the first few bytes for speed
                    seed[0] = seed[0].wrapping_add(1);
                    if seed[0] == 0 {
                        seed[1] = seed[1].wrapping_add(1);
                        if seed[1] == 0 {
                            // Re-seed occasionally to avoid getting stuck in a bad loop
                            rng.fill_bytes(&mut seed);
                        }
                    }

                    // Efficient Key Generation from seed using ed25519-dalek v1
                    // This performs SHA512 + Curve25519 mul
                    if let Ok(secret) = SecretKey::from_bytes(&seed) {
                        let public: PublicKey = (&secret).into();

                        // Encode to Base58
                        let pk_b58 = bs58::encode(public.as_bytes()).into_string();

                        let p_match = if case { pk_b58.starts_with(&prefix_clone) } else { pk_b58.to_lowercase().starts_with(&prefix_clone.to_lowercase()) };

                        if p_match {
                            let s_match = if case { pk_b58.ends_with(&suffix_clone) } else { pk_b58.to_lowercase().ends_with(&suffix_clone.to_lowercase()) };

                            if s_match {
                                if !found_clone.swap(true, Ordering::Relaxed) {
                                    // Construct the final keypair bytes (64 bytes: 32 priv + 32 pub)
                                    // Note: ed25519-dalek v1 SecretKey is 32 bytes (seed).
                                    // Solana expects 64 bytes (seed + pubkey).
                                    let mut final_keypair_bytes = [0u8; 64];
                                    final_keypair_bytes[0..32].copy_from_slice(secret.as_bytes());
                                    final_keypair_bytes[32..64].copy_from_slice(public.as_bytes());

                                    let final_b58 = bs58::encode(final_keypair_bytes).into_string();
                                    println!("MATCH_FOUND:{}", final_b58);
                                }
                                break;
                            }
                        }
                    }

                    attempts += 1;
                    if attempts % 1000 == 0 {
                        // Yield occasionally
                        if found_clone.load(Ordering::Relaxed) { break; }
                    }
                }
            });
        }
    });
}
