//! `exec` — run a shell command once, capture its output, kill it if it hangs.

use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::Command;

use super::{ToolDefinition, arg_str, clip};

const TIMEOUT: Duration = Duration::from_secs(120);
/// Per stream, so a runaway process cannot grow the result without bound.
const MAX_STREAM: usize = 100 * 1024;
const MAX_OUTPUT: usize = 100 * 1024;

pub fn definition() -> ToolDefinition {
    ToolDefinition {
        name: "exec",
        description: "Run a shell command in the working directory and return its exit code, \
                      stdout and stderr. Non-zero exit codes are normal output. \
                      The command is killed after 120 seconds and long output is truncated.",
        parameters: json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Command line, run with `sh -c`."
                }
            },
            "required": ["command"]
        }),
    }
}

pub async fn run(cwd: &Path, args: &Value) -> Result<String> {
    run_with_timeout(cwd, args, TIMEOUT).await
}

async fn run_with_timeout(cwd: &Path, args: &Value, timeout: Duration) -> Result<String> {
    let command_line = arg_str(args, "command")?;

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
        .current_dir(cwd)
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
    let read_stdout = tokio::spawn(read_capped(stdout));
    let read_stderr = tokio::spawn(read_capped(stderr));

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

    let stdout = read_stdout.await.context("stdout reader panicked")??;
    let stderr = read_stderr.await.context("stderr reader panicked")??;

    let code = match status.code() {
        Some(code) => code.to_string(),
        None => "killed by signal".to_string(),
    };
    let mut output = format!("exit code: {code}");
    if !stdout.is_empty() {
        output.push_str("\nstdout:\n");
        output.push_str(&String::from_utf8_lossy(&stdout));
    }
    if !stderr.is_empty() {
        output.push_str("\nstderr:\n");
        output.push_str(&String::from_utf8_lossy(&stderr));
    }
    Ok(clip(&output, MAX_OUTPUT))
}

async fn read_capped(reader: impl AsyncRead + Unpin) -> std::io::Result<Vec<u8>> {
    let mut buffer = Vec::new();
    reader
        .take((MAX_STREAM + 1) as u64)
        .read_to_end(&mut buffer)
        .await?;
    Ok(buffer)
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

        let output = run(&dir, &json!({"command": "cat marker.txt && pwd"}))
            .await
            .unwrap();

        assert!(output.contains("exit code: 0"), "{output}");
        assert!(output.contains("hi"), "{output}");
        assert!(
            output.contains(&dir.display().to_string()),
            "cwd not used: {output}"
        );
    }

    #[tokio::test]
    async fn returns_non_zero_exit_codes_as_output() {
        let dir = temp_dir("exec-fail");

        let output = run(&dir, &json!({"command": "echo boom >&2; exit 3"}))
            .await
            .unwrap();

        assert!(output.contains("exit code: 3"), "{output}");
        assert!(output.contains("boom"), "{output}");
    }

    #[tokio::test]
    async fn times_out_instead_of_hanging() {
        let dir = temp_dir("exec-timeout");

        let error = run_with_timeout(
            &dir,
            &json!({"command": "sleep 30"}),
            Duration::from_millis(200),
        )
        .await
        .unwrap_err();

        assert!(error.to_string().contains("timed out"), "{error}");
    }
}
