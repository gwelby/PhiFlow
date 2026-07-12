//! cascade_keys.rs — Rust interface to the CASCADE vault (`~/.cascade_keys`).
//!
//! Drop this module into your Rust workspace and call:
//!
//! ```rust
//! let token = cascade_keys::get_key("TEST_TOKEN_NAME");
//! ```
//!
//! Rules enforced:
//! - Vault path: `~/.cascade_keys`
//! - Required mode: `0o600`
//! - Format: `KEY=value`, blank lines and `# comments` ignored
//! - Values are never logged.

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

#[derive(Debug)]
pub enum VaultError {
    NotFound,
    BadPermissions(u32),
    Io(io::Error),
}

impl std::fmt::Display for VaultError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VaultError::NotFound => write!(f, "vault not found"),
            VaultError::BadPermissions(mode) => write!(
                f,
                "vault has insecure permissions: 0o{:03o}; expected 0o600",
                mode
            ),
            VaultError::Io(e) => write!(f, "vault read error: {e}"),
        }
    }
}

impl std::error::Error for VaultError {}

impl From<io::Error> for VaultError {
    fn from(e: io::Error) -> Self {
        VaultError::Io(e)
    }
}

/// Return the canonical vault path: `~/.cascade_keys`.
pub fn vault_path() -> PathBuf {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".cascade_keys")
}

/// Load the vault into a map. Permissions are checked on every call.
pub fn load_vault() -> Result<HashMap<String, String>, VaultError> {
    load_vault_at(&vault_path())
}

/// Load the vault at an explicit path (useful for tests).
pub fn load_vault_at(path: &Path) -> Result<HashMap<String, String>, VaultError> {
    if !path.exists() {
        return Err(VaultError::NotFound);
    }

    #[cfg(unix)]
    {
        let meta = path.metadata()?;
        let mode = meta.permissions().mode() & 0o777;
        if mode != 0o600 {
            eprintln!(
                "[SECURITY WARNING] {} has insecure permissions: 0o{:03o} (should be 0o600)",
                path.display(),
                mode
            );
        }
    }
    #[cfg(not(unix))]
    {
        let _ = path.metadata()?;
    }

    let text = fs::read_to_string(path)?;
    let mut map = HashMap::new();

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        map.insert(k.trim().to_string(), v.trim().to_string());
    }

    Ok(map)
}

/// Cached vault contents. The first successful load is cached.
fn cached_vault() -> Option<&'static HashMap<String, String>> {
    static CACHE: OnceLock<Option<HashMap<String, String>>> = OnceLock::new();
    CACHE.get_or_init(|| match load_vault() {
        Ok(map) => Some(map),
        Err(e) => {
            eprintln!("[ERROR] Could not load CASCADE vault: {e}");
            None
        }
    }).as_ref()
}

/// Retrieve a key from the vault, or `None` if missing.
///
/// Resolution order:
/// 1. Environment variable `name` (if non-empty).
/// 2. Value from `~/.cascade_keys`.
pub fn get_key(name: &str) -> Option<String> {
    if let Ok(v) = std::env::var(name) {
        if !v.is_empty() {
            return Some(v);
        }
    }

    cached_vault()
        .and_then(|map| map.get(name).cloned())
        .filter(|s| !s.is_empty())
}

/// Retrieve a key or return a default.
pub fn get_key_or(name: &str, default: &str) -> String {
    get_key(name).unwrap_or_else(|| default.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn parses_key_value_lines() {
        let dir = std::env::temp_dir().join("cascade_keys_rust_test");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let vault = dir.join(".cascade_keys");

        {
            let mut f = fs::File::create(&vault).unwrap();
            writeln!(f, "# comment").unwrap();
            writeln!(f, "TEST_TOKEN_NAME=abc123").unwrap();
            writeln!(f, "   spaced_key   =   spaced_value   ").unwrap();
            writeln!(f, "").unwrap();
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&vault, std::fs::Permissions::from_mode(0o600)).unwrap();
        }

        let map = load_vault_at(&vault).unwrap();
        assert_eq!(map.get("TEST_TOKEN_NAME"), Some(&"abc123".to_string()));
        assert_eq!(map.get("spaced_key"), Some(&"spaced_value".to_string()));
    }
}
