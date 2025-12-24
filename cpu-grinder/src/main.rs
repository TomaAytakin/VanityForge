use clap::Parser;
use rayon::prelude::*;
use solana_sdk::signer::keypair::Keypair;
use solana_sdk::signer::Signer;
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

    rayon::iter::repeat(()).any(|_| {
        if found.load(Ordering::Relaxed) { return true; }
        let kp = Keypair::new();
        let pk = kp.pubkey().to_string();
        
        let p_match = if case { pk.starts_with(&prefix) } else { pk.to_lowercase().starts_with(&prefix.to_lowercase()) };
        let s_match = if case { pk.ends_with(&suffix) } else { pk.to_lowercase().ends_with(&suffix.to_lowercase()) };

        if p_match && s_match {
            found.store(true, Ordering::Relaxed);
            let final_b58 = bs58::encode(kp.to_bytes()).into_string();
            println!("MATCH_FOUND:{}", final_b58);
            return true;
        }
        false
    });
}
