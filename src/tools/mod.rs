//! The four built-in tools, and the [`Toolbox`] implementation that dispatches
//! to them.
//!
//! There is no registry and no trait per tool: a tool is a module with a
//! `definition()` and a `run()`, and one `match` arm in [`Toolbox::execute`].
//! Hosts that need different tools implement [`Toolbox`] themselves.

mod edit;
mod exec;
mod read;
mod write;

use std::path::{Component, Path, PathBuf};
use std::time::Duration;

use anyhow::{Result, anyhow, bail};
use serde_json::Value;

use crate::{ToolCall, ToolDefinition, Toolbox, clip};

const DEFAULT_EXEC_TIMEOUT: Duration = Duration::from_secs(120);
const DEFAULT_MAX_OUTPUT: usize = 100 * 1024;
const DEFAULT_READ_LIMIT: usize = 2000;

/// `read`, `write`, `edit` and `exec`, bound to one working directory.
///
/// ```no_run
/// # use agent::CodingTools;
/// # use std::time::Duration;
/// let tools = CodingTools::new(std::env::current_dir().unwrap())
///     .exec_timeout(Duration::from_secs(1800))
///     .max_output(1024 * 1024);
/// ```
pub struct CodingTools {
    /// Every relative path and every command is resolved against this.
    pub cwd: PathBuf,
    pub exec_timeout: Duration,
    pub max_output: usize,
    pub read_limit: usize,
}

impl CodingTools {
    pub fn new(cwd: PathBuf) -> Self {
        Self {
            cwd,
            exec_timeout: DEFAULT_EXEC_TIMEOUT,
            max_output: DEFAULT_MAX_OUTPUT,
            read_limit: DEFAULT_READ_LIMIT,
        }
    }

    /// How long `exec` may run before it is killed.
    pub fn exec_timeout(mut self, timeout: Duration) -> Self {
        self.exec_timeout = timeout;
        self
    }

    /// How much output a tool may return, in bytes. `read` and `exec` clip to
    /// this; `exec` keeps draining a runaway command instead of blocking it.
    pub fn max_output(mut self, bytes: usize) -> Self {
        self.max_output = bytes.max(1);
        self
    }

    /// Default line count for `read` when the model does not pass `limit`.
    pub fn read_limit(mut self, lines: usize) -> Self {
        self.read_limit = lines.max(1);
        self
    }
}

impl Toolbox for CodingTools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![
            read::definition(self.read_limit),
            write::definition(),
            edit::definition(),
            exec::definition(self.exec_timeout),
        ]
    }

    async fn execute(&self, call: &ToolCall) -> Result<String> {
        match call.name.as_str() {
            "read" => read::run(self, &call.arguments).await,
            "write" => write::run(self, &call.arguments).await,
            "edit" => edit::run(self, &call.arguments).await,
            "exec" => exec::run(self, &call.arguments).await,
            other => Err(anyhow!(
                "unknown tool {other:?}: available tools are read, write, edit, exec"
            )),
        }
    }
}

/// Resolve a tool path against the working directory and normalize `.` / `..`
/// lexically. Absolute paths are accepted as they are: `exec` can reach the
/// whole filesystem anyway, so a path sandbox here would be theatre.
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
        None => bail!(
            "missing string argument `{key}` in {}",
            clip(&args.to_string(), 300)
        ),
    }
}

/// The path description every file tool shares.
const PATH_DESCRIPTION: &str = "Path to a UTF-8 text file. Relative paths resolve against the working \
     directory; absolute paths are accepted.";

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

    /// Unparseable arguments reach the tool as a JSON string, so the error says
    /// what the model actually sent.
    #[test]
    fn arg_str_shows_unparseable_arguments() {
        let args = serde_json::Value::String("{\"path\":".to_string());
        let error = arg_str(&args, "path").unwrap_err().to_string();
        assert!(error.contains(r#"{\"path\":"#), "{error}");
    }

    #[tokio::test]
    async fn dispatches_known_tools_and_rejects_unknown_ones() {
        let dir = temp_dir("toolbox-dispatch");
        let tools = CodingTools::new(dir.clone());
        assert_eq!(tools.definitions().len(), 4);

        let write = ToolCall {
            id: "1".into(),
            name: "write".into(),
            arguments: serde_json::json!({"path": "a.txt", "content": "hi"}),
        };
        assert!(
            tools
                .execute(&write)
                .await
                .unwrap()
                .starts_with("wrote 2 bytes")
        );

        let unknown = ToolCall {
            id: "2".into(),
            name: "grep".into(),
            arguments: serde_json::json!({}),
        };
        let error = tools.execute(&unknown).await.unwrap_err().to_string();
        assert!(error.contains("unknown tool"), "{error}");
    }
}
