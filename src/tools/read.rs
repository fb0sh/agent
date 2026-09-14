//! `read` — return a window of a UTF-8 text file.

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use tokio::io::AsyncReadExt;

use super::{CodingTools, PATH_DESCRIPTION, arg_str, resolve};
use crate::ToolDefinition;

/// Bytes scanned per read.
const CHUNK: usize = 16 * 1024;

pub fn definition(limit: usize) -> ToolDefinition {
    ToolDefinition {
        name: "read".into(),
        description: format!(
            "Read a UTF-8 text file. Returns `limit` lines starting at line `offset` \
             (1-based, {limit} lines by default). Reading a large file without \
             offset/limit returns its head only."
        ),
        parameters: json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": PATH_DESCRIPTION},
                "offset": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "First line to return, 1-based. Defaults to 1."
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum number of lines to return."
                }
            },
            "required": ["path"],
            "additionalProperties": false
        }),
    }
}

pub async fn run(tools: &CodingTools, args: &Value) -> Result<String> {
    let path = resolve(&tools.cwd, arg_str(args, "path")?)?;
    let offset = count_arg(args, "offset")?.unwrap_or(1);
    let limit = count_arg(args, "limit")?.unwrap_or(tools.read_limit);
    let max = tools.max_output;

    let mut file = tokio::fs::File::open(&path)
        .await
        .with_context(|| format!("failed to open {}", path.display()))?;

    // Scan fixed-size chunks and keep only the bytes that will be returned, so a
    // file with one enormous line costs one chunk of memory, not the whole line.
    let mut chunk = vec![0u8; CHUNK];
    let mut selected = Vec::new();
    let mut line = 1usize;
    let mut kept = 0usize;
    let mut through = 1usize;
    let mut truncated = false;
    let mut empty = true;
    let mut ends_with_newline = false;

    'reading: loop {
        let read = file
            .read(&mut chunk)
            .await
            .with_context(|| format!("failed to read {}", path.display()))?;
        if read == 0 {
            break;
        }
        empty = false;
        for &byte in &chunk[..read] {
            ends_with_newline = byte == b'\n';
            let keeping = line >= offset && kept < limit;
            if keeping {
                if selected.len() == max {
                    truncated = true;
                    break 'reading;
                }
                selected.push(byte);
                through = line;
            } else if line >= offset {
                // Past the requested window: there is more in the file.
                truncated = true;
                break 'reading;
            }
            if byte == b'\n' {
                if keeping {
                    kept += 1;
                }
                line += 1;
            }
        }
    }

    // An empty file is simply empty; only a window past the end is an error.
    if empty {
        if offset <= 1 {
            return Ok(String::new());
        }
        bail!(
            "nothing to read at offset {offset}: {} is empty",
            path.display()
        );
    }
    if selected.is_empty() {
        let total = if ends_with_newline { line - 1 } else { line };
        bail!(
            "nothing to read at offset {offset}: {} has {total} lines",
            path.display()
        );
    }

    let mut text = String::from_utf8(selected)
        .with_context(|| format!("{} is not valid UTF-8 text", path.display()))?;
    if truncated {
        text = format!("[truncated: lines {offset}-{through}]\n{text}");
    }
    Ok(text)
}

fn count_arg(args: &Value, key: &str) -> Result<Option<usize>> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_u64()
            .filter(|n| *n >= 1)
            .map(|n| Some(n as usize))
            .with_context(|| format!("`{key}` must be an integer >= 1, got {value}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::temp_dir;

    #[tokio::test]
    async fn reads_a_file() {
        let dir = temp_dir("read-basic");
        tokio::fs::write(dir.join("a.txt"), "one\ntwo\nthree\n")
            .await
            .unwrap();
        let tools = CodingTools::new(dir);

        let output = run(&tools, &json!({"path": "a.txt"})).await.unwrap();
        assert_eq!(output, "one\ntwo\nthree\n");
    }

    #[tokio::test]
    async fn honours_offset_and_limit() {
        let dir = temp_dir("read-window");
        tokio::fs::write(dir.join("a.txt"), "1\n2\n3\n4\n5\n")
            .await
            .unwrap();
        let tools = CodingTools::new(dir);

        let output = run(&tools, &json!({"path": "a.txt", "offset": 2, "limit": 2}))
            .await
            .unwrap();
        assert!(output.starts_with("[truncated: lines 2-3]"), "{output}");
        assert!(output.ends_with("2\n3\n"), "{output}");

        let tail = run(&tools, &json!({"path": "a.txt", "offset": 4}))
            .await
            .unwrap();
        assert_eq!(tail, "4\n5\n");
    }

    /// The window can end without a trailing newline when the file does.
    #[tokio::test]
    async fn reads_a_file_that_does_not_end_with_a_newline() {
        let dir = temp_dir("read-no-trailing");
        tokio::fs::write(dir.join("a.txt"), "one\ntwo")
            .await
            .unwrap();
        let tools = CodingTools::new(dir);

        assert_eq!(
            run(&tools, &json!({"path": "a.txt"})).await.unwrap(),
            "one\ntwo"
        );
        assert_eq!(
            run(&tools, &json!({"path": "a.txt", "offset": 2}))
                .await
                .unwrap(),
            "two"
        );
    }

    #[tokio::test]
    async fn uses_the_configured_default_limit() {
        let dir = temp_dir("read-config-limit");
        tokio::fs::write(dir.join("a.txt"), "1\n2\n3\n")
            .await
            .unwrap();
        let tools = CodingTools::new(dir).read_limit(2);

        let output = run(&tools, &json!({"path": "a.txt"})).await.unwrap();
        assert!(output.starts_with("[truncated: lines 1-2]"), "{output}");
        assert!(definition(2).description.contains("2 lines by default"));
    }

    /// A single line larger than the output limit must not be buffered whole.
    #[tokio::test]
    async fn bounds_a_giant_single_line() {
        let dir = temp_dir("read-giant-line");
        tokio::fs::write(dir.join("a.txt"), "a".repeat(4 * 1024 * 1024))
            .await
            .unwrap();
        let tools = CodingTools::new(dir).max_output(1024);

        let output = run(&tools, &json!({"path": "a.txt"})).await.unwrap();
        assert!(output.len() < 2048, "not bounded: {} bytes", output.len());
        assert!(output.starts_with("[truncated: lines 1-1]"), "{output}");
    }

    #[tokio::test]
    async fn reads_an_empty_file_as_empty() {
        let dir = temp_dir("read-empty");
        tokio::fs::write(dir.join("a.txt"), "").await.unwrap();
        let tools = CodingTools::new(dir);

        assert_eq!(run(&tools, &json!({"path": "a.txt"})).await.unwrap(), "");

        let error = run(&tools, &json!({"path": "a.txt", "offset": 2}))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("is empty"), "{error}");
    }

    #[tokio::test]
    async fn rejects_bad_offsets_and_reports_missing_files() {
        let dir = temp_dir("read-errors");
        let tools = CodingTools::new(dir.clone());

        let zero = run(&tools, &json!({"path": "a.txt", "offset": 0}))
            .await
            .unwrap_err();
        assert!(zero.to_string().contains("integer >= 1"), "{zero}");

        let missing = run(&tools, &json!({"path": "nope.txt"})).await.unwrap_err();
        assert!(missing.to_string().contains("failed to open"), "{missing}");

        tokio::fs::write(dir.join("a.txt"), "only\n").await.unwrap();
        let past_end = run(&tools, &json!({"path": "a.txt", "offset": 9}))
            .await
            .unwrap_err();
        assert!(past_end.to_string().contains("has 1 lines"), "{past_end}");
    }

    #[tokio::test]
    async fn reports_missing_arguments() {
        let dir = temp_dir("read-args");
        let error = run(&CodingTools::new(dir), &json!({})).await.unwrap_err();
        assert!(error.to_string().contains("`path`"), "{error}");
    }
}
