//! `el-chat` — an interactive test client that holds a multi-turn chat with a
//! small **local** LLM (Qwen2.5-0.5B-Instruct, GGUF) running entirely on-device.
//!
//! It exists to exercise the SDK end-to-end: every reply flows through
//! [`el_engine_candle::QwenChatProvider`] → [`el_core::LlmProvider`] →
//! `el_runtime::InferenceSession` (provenance gate → prefill → decode loop).
//! The client itself depends only on SDK crates and contains no inference,
//! model, or tokenizer code of its own.
//!
//! ```text
//! cargo run -p el-chat                          # interactive REPL, ./models defaults
//! cargo run -p el-chat -- --prompt "hi" --once  # one-shot, non-interactive
//! ```
//!
//! REPL commands: `/reset`, `/system <text>`, `/help`, `/exit`.
//!
//! Decoding is the SDK runtime's deterministic greedy argmax, so replies are
//! reproducible (the local path does not sample on temperature).

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::time::Instant;

use el_core::{ChatMessage, ChatRequest, ChatToken, LlmProvider, SafetyMode};
use el_engine_candle::QwenChatProvider;

const DEFAULT_MODEL: &str = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
const DEFAULT_TOKENIZER: &str = "models/qwen2.5-0.5b-instruct.tokenizer.json";
const DEFAULT_SYSTEM: &str = "You are a helpful, concise assistant running locally on-device.";

struct Args {
    model: PathBuf,
    tokenizer: PathBuf,
    system: String,
    max_tokens: u32,
    once: Option<String>,
    safety: SafetyMode,
    guard_words: Vec<String>,
    expert_model: Option<PathBuf>,
    steer_alpha: i32,
}

fn parse_args() -> Result<Args, String> {
    let mut model = PathBuf::from(DEFAULT_MODEL);
    let mut tokenizer = PathBuf::from(DEFAULT_TOKENIZER);
    let mut system = DEFAULT_SYSTEM.to_string();
    let mut max_tokens = 512u32;
    let mut once = None;
    let mut safety = SafetyMode::Lightweight;
    let mut guard_words: Vec<String> = Vec::new();
    let mut expert_model: Option<PathBuf> = None;
    let mut steer_alpha = 1000i32;

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut next = |name: &str| it.next().ok_or_else(|| format!("{name} needs a value"));
        match arg.as_str() {
            "--model" | "-m" => model = PathBuf::from(next("--model")?),
            "--tokenizer" | "-t" => tokenizer = PathBuf::from(next("--tokenizer")?),
            "--system" | "-s" => system = next("--system")?,
            "--prompt" | "-p" => once = Some(next("--prompt")?),
            "--once" => once = once.or(Some(String::new())),
            "--max-tokens" => {
                max_tokens = next("--max-tokens")?
                    .parse()
                    .map_err(|_| "bad --max-tokens")?
            }
            "--safety" => {
                safety = match next("--safety")?.to_ascii_lowercase().as_str() {
                    "off" | "none" => SafetyMode::Off,
                    "lightweight" | "light" | "on" => SafetyMode::Lightweight,
                    other => return Err(format!("bad --safety '{other}' (use off|lightweight)")),
                }
            }
            "--guard-word" => guard_words.push(next("--guard-word")?),
            "--expert-model" => expert_model = Some(PathBuf::from(next("--expert-model")?)),
            "--steer-alpha" => {
                let a: i32 = next("--steer-alpha")?
                    .parse()
                    .map_err(|_| "bad --steer-alpha")?;
                // Negative would reverse the safety direction; cap the upper
                // bound so steering stays sane and the milli math never wraps.
                if !(0..=4000).contains(&a) {
                    return Err("--steer-alpha must be 0..=4000 (x1000; 1000 = 1.0x)".to_string());
                }
                steer_alpha = a;
            }
            "--help" | "-h" => return Err("help".to_string()),
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    Ok(Args {
        model,
        tokenizer,
        system,
        max_tokens,
        once,
        safety,
        guard_words,
        expert_model,
        steer_alpha,
    })
}

fn usage() {
    eprintln!(
        "el-chat — local LLM chat test client (exercises the SDK)\n\n\
         USAGE:\n  el-chat [OPTIONS]\n\n\
         OPTIONS:\n\
         \x20 -m, --model <PATH>        GGUF model file [default: {DEFAULT_MODEL}]\n\
         \x20 -t, --tokenizer <PATH>    tokenizer.json  [default: {DEFAULT_TOKENIZER}]\n\
         \x20 -s, --system <TEXT>       system prompt\n\
         \x20 -p, --prompt <TEXT>       send one message, print the reply, exit\n\
         \x20     --once                read one line from stdin, reply, exit\n\
         \x20     --max-tokens <N>      max generated tokens per reply [default: 512]\n\
         \x20     --safety <MODE>       on-device safety: off | lightweight [default: lightweight]\n\
         \x20     --guard-word <WORD>   add a chunk-guard trip word (repeatable; demo/test hook)\n\
         \x20     --expert-model <PATH> safety expert GGUF → model-backed contrastive steering (ADR-013)\n\
         \x20     --steer-alpha <MILLI> contrastive steering strength x1000 [default: 1000]\n\
         \x20 -h, --help               show this help\n\n\
         REPL COMMANDS: /reset  /system <text>  /help  /exit"
    );
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            if e != "help" {
                eprintln!("error: {e}\n");
            }
            usage();
            std::process::exit(if e == "help" { 0 } else { 2 });
        }
    };

    if !args.model.exists() {
        eprintln!(
            "error: model file not found: {}\n\nFetch a small instruct model, e.g.:\n  \
             curl -sSL -o {DEFAULT_MODEL} \\\n    \
             https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf\n  \
             curl -sSL -o {DEFAULT_TOKENIZER} \\\n    \
             https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json",
            args.model.display()
        );
        std::process::exit(1);
    }

    eprint!("loading {} ... ", args.model.display());
    let _ = std::io::stderr().flush();
    let load_start = Instant::now();
    let provider = match QwenChatProvider::from_paths(&args.model, &args.tokenizer) {
        Ok(p) => {
            let mut p = p
                .with_safety(args.safety)
                .with_extra_guard_words(args.guard_words.iter());
            if let Some(ref expert) = args.expert_model {
                p = p
                    .with_expert_model(expert)
                    .with_steer_alpha(args.steer_alpha);
            }
            p
        }
        Err(e) => {
            eprintln!("\nerror: failed to load model: {e}");
            std::process::exit(1);
        }
    };
    eprintln!("ready ({:.1}s)", load_start.elapsed().as_secs_f64());

    // Show the active on-device safety posture (ADR-005 tier + ADR-012 loop).
    let safety_desc = match args.safety {
        SafetyMode::Off => "off".to_string(),
        _ => {
            let mut d =
                "lightweight — ADR-012 decode-time guard + checkpointed rollback".to_string();
            if let Some(ref expert) = args.expert_model {
                d.push_str(&format!(
                    "; ADR-013 contrastive steer (expert: {}, alpha {})",
                    expert.display(),
                    args.steer_alpha
                ));
            }
            if !args.guard_words.is_empty() {
                d.push_str(&format!("; guard words: {}", args.guard_words.join(", ")));
            }
            d
        }
    };
    eprintln!("safety: {safety_desc}");

    let mut history: Vec<ChatMessage> = vec![ChatMessage::system(&args.system)];

    // One-shot mode: --prompt "..." or --once (read a single stdin line).
    if let Some(p) = args.once {
        let text = if p.is_empty() {
            let mut line = String::new();
            let _ = std::io::stdin().lock().read_line(&mut line);
            line.trim().to_string()
        } else {
            p
        };
        if !text.is_empty() {
            history.push(ChatMessage::user(text));
            let req = ChatRequest::new("local", history.clone()).with_max_tokens(args.max_tokens);
            let _ = run_turn(&provider, &req);
            println!();
        }
        return;
    }

    eprintln!(
        "\nLocal chat ready. Type a message; '/help' for commands, '/exit' to quit.\n\
         (system: {})\n",
        args.system
    );

    let stdin = std::io::stdin();
    loop {
        print!("\x1b[1;34myou>\x1b[0m ");
        let _ = std::io::stdout().flush();

        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => break, // EOF
            Ok(_) => {}
            Err(e) => {
                eprintln!("input error: {e}");
                break;
            }
        }
        let input = line.trim();
        if input.is_empty() {
            continue;
        }

        if let Some(rest) = input.strip_prefix('/') {
            let mut parts = rest.splitn(2, ' ');
            match parts.next().unwrap_or("") {
                "exit" | "quit" | "q" => break,
                "help" | "h" => {
                    usage();
                    continue;
                }
                "reset" => {
                    history = vec![ChatMessage::system(&args.system)];
                    eprintln!("(conversation reset)");
                    continue;
                }
                "system" => {
                    let new_sys = parts.next().unwrap_or("").trim();
                    if new_sys.is_empty() {
                        eprintln!("(usage: /system <text>)");
                    } else {
                        history = vec![ChatMessage::system(new_sys)];
                        eprintln!("(system prompt updated; conversation reset)");
                    }
                    continue;
                }
                other => {
                    eprintln!("(unknown command '/{other}'; try /help)");
                    continue;
                }
            }
        }

        history.push(ChatMessage::user(input.to_string()));
        let req = ChatRequest::new("local", history.clone()).with_max_tokens(args.max_tokens);

        match run_turn(&provider, &req) {
            Ok(reply) => history.push(ChatMessage::assistant(reply)),
            Err(e) => {
                eprintln!("\n(generation error: {e}; conversation reset)");
                history = vec![ChatMessage::system(&args.system)];
            }
        }
    }
    eprintln!("bye.");
}

/// Stream one assistant reply to stdout via the SDK's `LlmProvider::chat_stream`;
/// returns the accumulated text so the caller can append it to history.
fn run_turn(provider: &QwenChatProvider, req: &ChatRequest) -> el_core::Result<String> {
    print!("\x1b[1;32mbot>\x1b[0m ");
    let _ = std::io::stdout().flush();

    let start = Instant::now();
    let mut reply = String::new();
    provider.chat_stream(req, &mut |t: ChatToken| {
        if t.is_final || t.text.is_empty() {
            return;
        }
        reply.push_str(&t.text);
        print!("{}", t.text);
        let _ = std::io::stdout().flush();
    })?;

    let secs = start.elapsed().as_secs_f64();
    eprintln!("\n\x1b[2m[{secs:.1}s]\x1b[0m");
    Ok(reply)
}
