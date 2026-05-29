//! Minimal reproduction for the Qwen3 hang.
use llama_gguf::engine::{Engine, EngineConfig};

fn main() {
    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "models/Qwen3-0.6B-Q3_K_S.gguf".to_string());

    eprintln!("Loading model from: {}", model_path);
    let engine = Engine::load(EngineConfig {
        model_path: model_path.clone(),
        max_context_len: Some(2048),
        max_tokens: 10,
        ..Default::default()
    })
    .expect("Failed to load model");

    eprintln!("Model loaded. Running blocking generation...");
    let start = std::time::Instant::now();
    let result = engine.generate("5+5", 10);
    eprintln!("Blocking generation took: {:?}", start.elapsed());
    match result {
        Ok(text) => eprintln!("Result: {}", text),
        Err(e) => eprintln!("Error: {:?}", e),
    }

    eprintln!("\nRunning streaming generation...");
    let start = std::time::Instant::now();
    let mut stream = engine.generate_streaming("5+5", 10);
    let mut count = 0;
    while let Some(Ok(token)) = stream.next() {
        eprint!("{}", token);
        count += 1;
        if count >= 10 { break; }
    }
    eprintln!("\nStreaming generation took: {:?}", start.elapsed());
}
