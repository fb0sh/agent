//! `exec` — run a shell command once, capture its output, kill it if it hangs.

use std::process::Stdio;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::Command;

use super::{CodingTools, arg_str};
use crate::{ToolDefinition, truncate};

pub fn definition(timeout: Duration) -> ToolDefinition {
    ToolDefinition {
        name: "exec".into(),
        description: format!(
            "Run a shell command in the working directory and return its exit code, \
             stdout and stderr. Non-zero exit codes are normal output. The command is \
             killed after {} seconds and long output is truncated.",
            timeout.as_secs()
        ),
        parameters: json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Command line, run by the platform shell (`sh -c` on Unix, `cmd /C` on Windows)."
                }
            },
            "required": ["command"],
            "additionalProperties": false
        }),
    }
}

pub async fn run(tools: &CodingTools, args: &Value) -> Result<String> {
    run_with(tools, args, tools.exec_timeout).await
}

async fn run_with(tools: &CodingTools, args: &Value, timeout: Duration) -> Result<String> {
    let command_line = arg_str(args, "command")?;
    let limit = tools.max_output;

    let mut command = if cfg!(windows) {
        let mut command = Command::new("cmd");
        command.arg("/C").arg(command_line);
        command
    } else {
        let mut command = Command::new("sh");
        command.arg("-c").arg(command_line);
        command
    };
    command
        .current_dir(&tools.cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    let mut child = command
        .spawn()
        .with_context(|| format!("failed to run {command_line:?}"))?;

    // Read both pipes while the child runs: a full pipe would otherwise block it.
    let stdout = child.stdout.take().context("no stdout pipe")?;
    let stderr = child.stderr.take().context("no stderr pipe")?;
    let read_stdout = tokio::spawn(drain(stdout, limit));
    let read_stderr = tokio::spawn(drain(stderr, limit));

    let status = match tokio::time::timeout(timeout, child.wait()).await {
        Ok(status) => status.context("failed to wait for the command")?,
        Err(_) => {
            let _ = child.kill().await;
            bail!(
                "command timed out after {}s: {command_line}",
                timeout.as_secs()
            );
        }
    };

    let (stdout, stdout_cut) = read_stdout.await.context("stdout reader panicked")??;
    let (stderr, stderr_cut) = read_stderr.await.context("stderr reader panicked")??;

    let code = match status.code() {
        Some(code) => code.to_string(),
        None => "killed by signal".to_string(),
    };
    let mut output = format!("exit code: {code}");
    // `limit` bounds the whole result, not each stream, so the two are cut to fit
    // together. Markers are added after the cuts, so they survive them.
    for (name, bytes, cut) in [
        ("stdout", &stdout, stdout_cut),
        ("stderr", &stderr, stderr_cut),
    ] {
        if bytes.is_empty() {
            continue;
        }
        let text = String::from_utf8_lossy(bytes);
        let room = limit.saturating_sub(output.len() + name.len() + 3);
        if room == 0 {
            output.push_str(&format!("\n[{name} truncated]"));
            continue;
        }
        output.push_str(&format!("\n{name}:\n"));
        let kept = truncate(&text, room);
        output.push_str(kept);
        if cut || kept.len() < text.len() {
            output.push_str(&format!("\n[{name} truncated]"));
        }
    }
    Ok(output)
}

/// Keep the first `limit` bytes and go on reading to EOF.
///
/// Stopping the read instead would leave the writer blocked on a full pipe and
/// turn a chatty command into a timeout, so the rest is read and discarded.
async fn drain(reader: impl AsyncRead + Unpin, limit: usize) -> std::io::Result<(Vec<u8>, bool)> {
    let mut reader = reader;
    let mut saved = Vec::new();
    let mut buffer = vec![0u8; 8192];
    let mut truncated = false;
    loop {
        let read = reader.read(&mut buffer).await?;
        if read == 0 {
            return Ok((saved, truncated));
        }
        let room = limit.saturating_sub(saved.len());
        if room > 0 {
            saved.extend_from_slice(&buffer[..read.min(room)]);
        }
        if read > room {
            truncated = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::temp_dir;

    #[tokio::test]
    async fn runs_a_command_in_the_working_directory() {
        let dir = temp_dir("exec-ok");
        tokio::fs::write(dir.join("marker.txt"), "hi")
            .await
            .unwrap();
        let tools = CodingTools::new(dir.clone());

        let output = run(&tools, &json!({"command": "cat marker.txt && pwd"}))
            .await
            .unwrap();

        assert!(output.contains("exit code: 0"), "{output}");
        assert!(output.contains("hi"), "{output}");
        assert!(output.contains(&dir.display().to_string()), "{output}");
    }

    #[tokio::test]
    async fn returns_non_zero_exit_codes_as_output() {
        let tools = CodingTools::new(temp_dir("exec-fail"));

        let output = run(&tools, &json!({"command": "echo boom >&2; exit 3"}))
            .await
            .unwrap();

        assert!(output.contains("exit code: 3"), "{output}");
        assert!(output.contains("boom"), "{output}");
    }

    #[tokio::test]
    async fn honours_the_configured_timeout() {
        let tools =
            CodingTools::new(temp_dir("exec-timeout")).exec_timeout(Duration::from_millis(200));

        let error = run(&tools, &json!({"command": "sleep 30"}))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("timed out"), "{error}");
    }

    /// stdout and stderr share one budget, so the total stays near `max_output`
    /// even when both flood the pipe.
    #[tokio::test]
    async fn bounds_the_whole_result_across_both_streams() {
        let tools = CodingTools::new(temp_dir("exec-budget"))
            .max_output(2048)
            .exec_timeout(Duration::from_secs(10));

        let output = run(&tools, &json!({"command": "seq 1 4000; seq 1 4000 >&2"}))
            .await
            .unwrap();

        assert!(output.contains("exit code: 0"), "{output}");
        assert!(output.contains("[stdout truncated]"), "{output}");
        assert!(output.contains("[stderr truncated]"), "{output}");
        assert!(output.len() < 4096, "not bounded: {} bytes", output.len());
    }

    /// A command that floods its pipe must be drained, not blocked: it should
    /// exit normally with truncated output instead of hitting the timeout.
    #[tokio::test]
    async fn drains_past_the_output_limit_without_blocking() {
        let tools = CodingTools::new(temp_dir("exec-drain"))
            .max_output(1024)
            .exec_timeout(Duration::from_secs(10));

        let output = run(
            &tools,
            &json!({"command": "i=0; while [ $i -lt 20000 ]; do echo aaaaaaaaaaaaaaaaaaaa; i=$((i+1)); done"}),
        )
        .await
        .unwrap();

        assert!(output.contains("exit code: 0"), "{output}");
        assert!(output.contains("[stdout truncated]"), "{output}");
        assert!(
            output.len() < 2048,
            "output not clipped: {} bytes",
            output.len()
        );
    }
}
