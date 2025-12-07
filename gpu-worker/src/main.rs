use clap::Parser;
use rayon::prelude::*;
use serde::Serialize;
use solana_sdk::signer::keypair::Keypair;
use solana_sdk::signer::Signer;
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

    #[arg(long)]
    ignore_case: bool,
}

#[derive(Serialize)]
struct Output {
    public_key: String,
    secret_key: String,
}

fn main() {
    let args = Args::parse();
    let prefix = args.prefix.unwrap_or_default();
    let suffix = args.suffix.unwrap_or_default();
    let ignore_case = args.ignore_case;

    let found = Arc::new(AtomicBool::new(false));

    // rayon uses the number of logical cores by default
    rayon::iter::repeat(()).for_each(|_| {
        if found.load(Ordering::Relaxed) {
            return;
        }

        let keypair = Keypair::new();
        let pubkey_str = keypair.pubkey().to_string();

        let matches = if ignore_case {
            let p_lower = prefix.to_lowercase();
            let s_lower = suffix.to_lowercase();
            let pk_lower = pubkey_str.to_lowercase();
            pk_lower.starts_with(&p_lower) && pk_lower.ends_with(&s_lower)
        } else {
            pubkey_str.starts_with(&prefix) && pubkey_str.ends_with(&suffix)
        };

        if matches {
            if !found.swap(true, Ordering::Relaxed) {
                let secret_bytes = keypair.to_bytes(); // 64 bytes
                let secret_string = bs58::encode(secret_bytes).into_string();

                let output = Output {
                    public_key: pubkey_str,
                    secret_key: secret_string,
                };

                println!("{}", serde_json::to_string(&output).unwrap());
                exit(0);
            }
        }
    });
}
