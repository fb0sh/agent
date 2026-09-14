//! `edit` — exact string replacement, and nothing more.

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

use super::{CodingTools, PATH_DESCRIPTION, arg_str, resolve};
use crate::ToolDefinition;

pub fn definition() -> ToolDefinition {
    ToolDefinition {
        name: "edit".into(),
        description: "Replace an exact snippet of a file. `old` must appear exactly once, \
                      so include enough surrounding context to make it unique. Read the file first."
            .into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": PATH_DESCRIPTION},
                "old": {
                    "type": "string",
                    "description": "Exact text to replace, copied from the file."
                },
                "new": {
                    "type": "string",
                    "description": "Replacement text. Use an empty string to delete."
                }
            },
            "required": ["path", "old", "new"],
            "additionalProperties": false
        }),
    }
}

pub async fn run(tools: &CodingTools, args: &Value) -> Result<String> {
    let path = resolve(&tools.cwd, arg_str(args, "path")?)?;
    let old = arg_str(args, "old")?;
    let new = arg_str(args, "new")?;

    if old.is_empty() {
        bail!("`old` must not be empty");
    }

    let text = tokio::fs::read_to_string(&path)
        .await
        .with_context(|| format!("failed to read {}", path.display()))?;

    match text.matches(old).count() {
        0 => bail!(
            "`old` does not appear in {}: read the file and copy an exact snippet",
            path.display()
        ),
        1 => {}
        count => bail!(
            "`old` appears {count} times in {}: include more surrounding context to make it unique",
            path.display()
        ),
    }

    tokio::fs::write(&path, text.replacen(old, new, 1))
        .await
        .with_context(|| format!("failed to write {}", path.display()))?;

    Ok(format!("edited {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::temp_dir;

    async fn tools_with(test: &str, content: &str) -> CodingTools {
        let dir = temp_dir(test);
        tokio::fs::write(dir.join("a.txt"), content).await.unwrap();
        CodingTools::new(dir)
    }

    #[tokio::test]
    async fn replaces_a_unique_match() {
        let tools = tools_with("edit-one", "let a = 1;\nlet b = 2;\n").await;

        let output = run(
            &tools,
            &json!({"path": "a.txt", "old": "let b = 2;", "new": "let b = 3;"}),
        )
        .await
        .unwrap();

        assert!(output.starts_with("edited"), "{output}");
        let text = tokio::fs::read_to_string(tools.cwd.join("a.txt"))
            .await
            .unwrap();
        assert_eq!(text, "let a = 1;\nlet b = 3;\n");
    }

    #[tokio::test]
    async fn deletes_with_an_empty_replacement() {
        let tools = tools_with("edit-delete", "keep\ndrop\n").await;

        run(
            &tools,
            &json!({"path": "a.txt", "old": "drop\n", "new": ""}),
        )
        .await
        .unwrap();

        let text = tokio::fs::read_to_string(tools.cwd.join("a.txt"))
            .await
            .unwrap();
        assert_eq!(text, "keep\n");
    }

    #[tokio::test]
    async fn rejects_zero_matches() {
        let tools = tools_with("edit-none", "let a = 1;\n").await;

        let error = run(
            &tools,
            &json!({"path": "a.txt", "old": "let z = 9;", "new": "x"}),
        )
        .await
        .unwrap_err();

        assert!(error.to_string().contains("does not appear"), "{error}");
        let text = tokio::fs::read_to_string(tools.cwd.join("a.txt"))
            .await
            .unwrap();
        assert_eq!(text, "let a = 1;\n");
    }

    #[tokio::test]
    async fn rejects_several_matches() {
        let tools = tools_with("edit-many", "x\nx\nx\n").await;

        let error = run(&tools, &json!({"path": "a.txt", "old": "x", "new": "y"}))
            .await
            .unwrap_err();

        assert!(error.to_string().contains("appears 3 times"), "{error}");
    }

    #[tokio::test]
    async fn rejects_an_empty_old_string() {
        let tools = tools_with("edit-empty", "x\n").await;

        let error = run(&tools, &json!({"path": "a.txt", "old": "", "new": "y"}))
            .await
            .unwrap_err();

        assert!(error.to_string().contains("must not be empty"), "{error}");
    }
}
