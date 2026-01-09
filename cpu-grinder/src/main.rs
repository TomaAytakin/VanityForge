use clap::Parser;
use rayon::prelude::*;
use ed25519_dalek::{Keypair, Signer};
use rand::rngs::OsRng;
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

    let cpu_count = num_cpus::get();

    rayon::scope(|s| {
        for _ in 0..cpu_count {
            let found_clone = found.clone();
            let prefix_clone = prefix.clone();
            let suffix_clone = suffix.clone();

            s.spawn(move |_| {
                let mut csprng = OsRng{};
                let mut attempts = 0;

                loop {
                    if found_clone.load(Ordering::Relaxed) {
                        break;
                    }

                    // Standard Generation: Seed -> Pair
                    let keypair = Keypair::generate(&mut csprng);
                    let pk_b58 = bs58::encode(keypair.public.to_bytes()).into_string();

                    let p_match = if case { pk_b58.starts_with(&prefix_clone) } else { pk_b58.to_lowercase().starts_with(&prefix_clone.to_lowercase()) };

                    if p_match {
                        let s_match = if case { pk_b58.ends_with(&suffix_clone) } else { pk_b58.to_lowercase().ends_with(&suffix_clone.to_lowercase()) };

                        if s_match {
                            if !found_clone.swap(true, Ordering::Relaxed) {
                                // Match!
                                let final_b58 = bs58::encode(keypair.to_bytes()).into_string();
                                println!("MATCH:{}:{}", pk_b58, final_b58);
                            }
                            return;
                        }
                    }

                    attempts += 1;
                    if attempts % 1000 == 0 {
                         if found_clone.load(Ordering::Relaxed) { break; }
                    }
                }
            });
        }
    });
}
