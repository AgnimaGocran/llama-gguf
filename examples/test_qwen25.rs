//! Test Qwen2.5 model with proper ChatML formatting.
use llama_gguf::engine::{Engine, EngineConfig};

fn main() {
    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "models/Qwen2.5-0.5B-Instruct-Q8_0.gguf".to_string());

    eprintln!("Loading model from: {}", model_path);
    let engine = Engine::load(EngineConfig {
        model_path: model_path.clone(),
        max_context_len: Some(2048),
        max_tokens: 50,
        ..Default::default()
    })
    .expect("Failed to load model");

    // Test 1: Simple prompt (engine will apply detected chat template)
    eprintln!("\n=== Test 1: Simple prompt ===");
    let start = std::time::Instant::now();
    let result = engine.generate("What is 2+2? Answer with just the number.", 10);
    eprintln!("Generation took: {:?}", start.elapsed());
    match result {
        Ok(text) => eprintln!("Result: '{}'", text.trim()),
        Err(e) => eprintln!("Error: {:?}", e),
    }

    // Test 2: Manual ChatML format for Qwen2.5
    eprintln!("\n=== Test 2: Manual ChatML format ===");
    let chatml_prompt = "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n<|im_start|>user\nWhat is the capital of France? Answer in one word.\n<|im_end|>\n<|im_start|>assistant\n";
    
    let start = std::time::Instant::now();
    let result = engine.generate(chatml_prompt, 10);
    eprintln!("Generation took: {:?}", start.elapsed());
    match result {
        Ok(text) => eprintln!("Result: '{}'", text.trim()),
        Err(e) => eprintln!("Error: {:?}", e),
    }

    // Test 3: Check chat template detection
    eprintln!("\n=== Test 3: Chat template info ===");
    eprintln!("Detected template: {:?}", engine.chat_template());

    // Test 4: Streaming generation
    eprintln!("\n=== Test 4: Streaming generation ===");
    let start = std::time::Instant::now();
    let mut stream = engine.generate_streaming("Say 'hello'", 10);
    let mut count = 0;
    eprint!("Stream: '");
    while let Some(Ok(token)) = stream.next() {
        eprint!("{}", token);
        count += 1;
        if count >= 10 { break; }
    }
    eprintln!("'");
    eprintln!("Streaming took: {:?}", start.elapsed());
    
    eprintln!("\n=== All tests complete ===");
}
