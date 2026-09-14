//! `write` — create or fully overwrite a file.

use anyhow::{Context, Result};
use serde_json::{Value, json};

use super::{CodingTools, PATH_DESCRIPTION, arg_str, resolve};
use crate::ToolDefinition;

pub fn definition() -> ToolDefinition {
    ToolDefinition {
        name: "write".into(),
        description: "Write a UTF-8 text file, creating it or replacing its whole content. \
                      Missing parent directories are created. Prefer `edit` for small changes."
            .into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": PATH_DESCRIPTION},
                "content": {
                    "type": "string",
                    "description": "Full content of the file."
                }
            },
            "required": ["path", "content"],
            "additionalProperties": false
        }),
    }
}

pub async fn run(tools: &CodingTools, args: &Value) -> Result<String> {
    let path = resolve(&tools.cwd, arg_str(args, "path")?)?;
    let content = arg_str(args, "content")?;

    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    tokio::fs::write(&path, content)
        .await
        .with_context(|| format!("failed to write {}", path.display()))?;

    Ok(format!(
        "wrote {} bytes to {}",
        content.len(),
        path.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::temp_dir;

    #[tokio::test]
    async fn creates_a_file_and_its_parents() {
        let dir = temp_dir("write-create");
        let tools = CodingTools::new(dir.clone());

        let output = run(&tools, &json!({"path": "a/b/c.txt", "content": "hello"}))
            .await
            .unwrap();

        assert!(output.starts_with("wrote 5 bytes"), "{output}");
        let written = tokio::fs::read_to_string(dir.join("a/b/c.txt"))
            .await
            .unwrap();
        assert_eq!(written, "hello");
    }

    #[tokio::test]
    async fn overwrites_an_existing_file() {
        let dir = temp_dir("write-overwrite");
        tokio::fs::write(dir.join("a.txt"), "old and long")
            .await
            .unwrap();
        let tools = CodingTools::new(dir.clone());

        run(&tools, &json!({"path": "a.txt", "content": "new"}))
            .await
            .unwrap();

        let written = tokio::fs::read_to_string(dir.join("a.txt")).await.unwrap();
        assert_eq!(written, "new");
    }

    #[tokio::test]
    async fn reports_missing_content() {
        let dir = temp_dir("write-args");
        let error = run(&CodingTools::new(dir), &json!({"path": "a.txt"}))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("`content`"), "{error}");
    }
}
