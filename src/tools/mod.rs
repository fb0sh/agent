//! The four tools the agent can call.
//!
//! There is no registry and no trait: a tool is a `match` arm, and adding one
//! means adding a file, its `definition()` and its arm in [`Tools::execute`].

mod edit;
mod exec;
mod read;
mod write;

use std::path::{Component, Path, PathBuf};

use anyhow::{Result, anyhow, bail};
use serde_json::Value;

use crate::llm::{ToolCall, ToolResult};

/// A tool as every supported API wants to see it.
pub struct ToolDefinition {
    pub name: &'static str,
    pub description: &'static str,
    pub parameters: Value,
}

/// The fixed tool set, bound to one working directory.
pub struct Tools {
    pub cwd: PathBuf,
}

impl Tools {
    pub fn new(cwd: PathBuf) -> Self {
        Self { cwd }
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        vec![
            read::definition(),
            write::definition(),
            edit::definition(),
            exec::definition(),
        ]
    }

    /// Run one tool call. A failure is not an error for the caller: it becomes
    /// `error: ...` output so the model can read it and correct itself instead
    /// of the agent aborting the whole run.
    pub async fn execute(&self, call: &ToolCall) -> ToolResult {
        let outcome = match call.name.as_str() {
            "read" => read::run(&self.cwd, &call.arguments).await,
            "write" => write::run(&self.cwd, &call.arguments).await,
            "edit" => edit::run(&self.cwd, &call.arguments).await,
            "exec" => exec::run(&self.cwd, &call.arguments).await,
            other => Err(anyhow!(
                "unknown tool {other:?}: available tools are read, write, edit, exec"
            )),
        };
        ToolResult {
            id: call.id.clone(),
            name: call.name.clone(),
            output: match outcome {
                Ok(output) => output,
                Err(error) => format!("error: {error:#}"),
            },
        }
    }
}

/// Resolve a tool path against the working directory and normalize `.` / `..`
/// lexically, so tools do not wander off by accident.
fn resolve(cwd: &Path, path: &str) -> Result<PathBuf> {
    if path.trim().is_empty() {
        bail!("`path` must not be empty");
    }
    let path = Path::new(path);
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(path)
    };

    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if !normalized.pop() {
                    normalized.push("..");
                }
            }
            other => normalized.push(other),
        }
    }
    Ok(normalized)
}

/// Read a required string argument.
fn arg_str<'a>(args: &'a Value, key: &str) -> Result<&'a str> {
    match args.get(key).and_then(Value::as_str) {
        Some(value) => Ok(value),
        None => bail!("missing string argument `{key}` in {args}"),
    }
}

/// Cut `text` down to `max` bytes on a char boundary, marking the cut.
pub fn clip(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    let mut end = max;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}\n[output truncated]", &text[..end])
}

/// A fresh scratch directory for the tests of this crate.
#[cfg(test)]
pub fn temp_dir(test: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join("mini-agent-tests")
        .join(format!("{}-{test}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clip_keeps_short_text_and_marks_long_text() {
        assert_eq!(clip("abc", 10), "abc");
        assert_eq!(clip("abcdef", 3), "abc\n[output truncated]");
        // Must not split a multi-byte character.
        assert_eq!(clip("日本語", 4), "日\n[output truncated]");
    }

    #[test]
    fn resolve_normalizes_dots_and_keeps_absolute_paths() {
        let cwd = Path::new("/tmp/work");
        assert_eq!(
            resolve(cwd, "a/b.rs").unwrap(),
            PathBuf::from("/tmp/work/a/b.rs")
        );
        assert_eq!(
            resolve(cwd, "./a/../b.rs").unwrap(),
            PathBuf::from("/tmp/work/b.rs")
        );
        assert_eq!(
            resolve(cwd, "/etc/hosts").unwrap(),
            PathBuf::from("/etc/hosts")
        );
        assert!(resolve(cwd, "  ").is_err());
    }

    #[test]
    fn arg_str_reports_missing_arguments() {
        let args = serde_json::json!({"path": "a.rs"});
        assert_eq!(arg_str(&args, "path").unwrap(), "a.rs");
        let error = arg_str(&args, "content").unwrap_err().to_string();
        assert!(error.contains("content"), "{error}");
    }
}
