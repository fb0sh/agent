//! `read` — return a window of a UTF-8 text file.

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, BufReader};

use super::{CodingTools, PATH_DESCRIPTION, arg_str, resolve};
use crate::{ToolDefinition, clip};

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

    let file = tokio::fs::File::open(&path)
        .await
        .with_context(|| format!("failed to open {}", path.display()))?;
    let mut lines = BufReader::new(file).lines();

    // Streamed, so an absurdly large file costs one window, not the whole file.
    let mut selected = Vec::new();
    let mut number = 0usize;
    let mut truncated = false;
    while let Some(line) = lines
        .next_line()
        .await
        .with_context(|| format!("failed to read {} as UTF-8 text", path.display()))?
    {
        number += 1;
        if number < offset {
            continue;
        }
        if selected.len() == limit {
            truncated = true;
            break;
        }
        selected.push(line);
    }

    if selected.is_empty() {
        bail!(
            "nothing to read at offset {offset}: {} has {number} lines",
            path.display()
        );
    }

    let mut text = selected.join("\n");
    text.push('\n');
    if truncated {
        text = format!(
            "[truncated: lines {offset}-{}]\n{text}",
            offset + selected.len() - 1
        );
    }
    Ok(clip(&text, tools.max_output))
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

    #[tokio::test]
    async fn clips_to_the_configured_output_limit() {
        let dir = temp_dir("read-clip");
        tokio::fs::write(dir.join("a.txt"), "x".repeat(500))
            .await
            .unwrap();
        let tools = CodingTools::new(dir).max_output(100);

        let output = run(&tools, &json!({"path": "a.txt"})).await.unwrap();
        assert!(output.ends_with("[output truncated]"), "{output}");
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
