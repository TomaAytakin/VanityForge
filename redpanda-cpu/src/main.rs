use clap::Parser;
use ed25519_dalek::{SigningKey, Signer};
use rand::rngs::OsRng;
use rayon::prelude::*;
use serde::Serialize;
use std::process::exit;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    prefix: Option<String>,

    #[arg(long)]
    suffix: Option<String>,

    #[arg(long, default_value_t = false)]
    case_sensitive: bool,
}

#[derive(Serialize)]
struct Output {
    public_key: String,
    secret_key: String,
}

fn starts_with_ignore_case(s: &str, prefix_lower: &str) -> bool {
    if s.len() < prefix_lower.len() {
        return false;
    }
    s.chars()
        .take(prefix_lower.chars().count())
        .zip(prefix_lower.chars())
        .all(|(c1, c2)| c1.to_ascii_lowercase() == c2)
}

fn ends_with_ignore_case(s: &str, suffix_lower: &str) -> bool {
    if s.len() < suffix_lower.len() {
        return false;
    }
    // Efficiently check suffix without full lowercase allocation
    let s_bytes = s.as_bytes();
    let suffix_bytes = suffix_lower.as_bytes();

    if s_bytes.len() < suffix_bytes.len() {
        return false;
    }

    let start_idx = s_bytes.len() - suffix_bytes.len();
    let s_suffix = &s_bytes[start_idx..];

    s_suffix.iter().zip(suffix_bytes.iter()).all(|(b1, b2)| {
        b1.to_ascii_lowercase() == *b2
    })
}

fn main() {
    let args = Args::parse();

    let case_sensitive = args.case_sensitive;
    let prefix_input = args.prefix.unwrap_or_default();
    let suffix_input = args.suffix.unwrap_or_default();

    // Prepare targets
    let (target_prefix, target_suffix) = if case_sensitive {
        (prefix_input, suffix_input)
    } else {
        (prefix_input.to_lowercase(), suffix_input.to_lowercase())
    };

    let found = Arc::new(AtomicBool::new(false));

    // Use Rayon to parallelize the search
    // We iterate over a massive range effectively acting as an infinite loop distributed across cores
    (0..u64::MAX).into_par_iter().find_any(|_| {
        // Early exit if found by another thread
        if found.load(Ordering::Relaxed) {
            return true;
        }

        let mut csprng = OsRng;
        let signing_key = SigningKey::generate(&mut csprng);

        // We need the string representation of the public key
        let public_key_bytes = signing_key.verifying_key().to_bytes();
        let public_key_str = bs58::encode(public_key_bytes).into_string();

        let matches = if case_sensitive {
            let p_match = if target_prefix.is_empty() { true } else { public_key_str.starts_with(&target_prefix) };
            let s_match = if target_suffix.is_empty() { true } else { public_key_str.ends_with(&target_suffix) };
            p_match && s_match
        } else {
            let p_match = if target_prefix.is_empty() { true } else { starts_with_ignore_case(&public_key_str, &target_prefix) };
            let s_match = if target_suffix.is_empty() { true } else { ends_with_ignore_case(&public_key_str, &target_suffix) };
            p_match && s_match
        };

        if matches {
            // Atomic swap to ensure only one thread prints and exits
            if !found.swap(true, Ordering::Relaxed) {
                let secret_key_bytes = signing_key.to_bytes(); // 32 bytes seed? or 64?
                // ed25519-dalek 2.0: SigningKey.to_bytes() returns 32-byte seed usually, OR the private key.
                // Wait.
                // In ed25519-dalek 2.0:
                // SigningKey::to_bytes() returns [u8; 32] (the seed).
                // SigningKey::to_keypair_bytes() returns [u8; 64] (seed || pubkey).

                // The Python code `base58.b58encode(bytes(kp))` where kp is from solders/solana-py usually encodes the 64-byte keypair.
                // Solders Keypair.to_bytes() returns 64 bytes.

                let keypair_bytes = signing_key.to_keypair_bytes();
                let secret_key_b58 = bs58::encode(keypair_bytes).into_string();

                let output = Output {
                    public_key: public_key_str,
                    secret_key: secret_key_b58,
                };

                println!("{}", serde_json::to_string(&output).unwrap());
                exit(0);
            }
            return true;
        }
        false
    });
}
