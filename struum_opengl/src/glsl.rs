use std::borrow::Cow;
use struum_types::StruumError;

const GLES_HEADER: &str = "#version 310 es\nprecision highp float;\nprecision highp int;\n";

/// Adapts desktop GLSL (`#version 430`) to whatever the active context needs.
///
/// Kernels are packed on the requester as desktop GLSL, so a GLES worker
/// (Android) rewrites the `#version` directive and adds default precisions.
pub(crate) fn adapt_source(src: &str) -> Result<Cow<'_, str>, StruumError> {
    if crate::gl_context::is_gles() {
        to_gles(src).map(Cow::Owned)
    } else {
        Ok(Cow::Borrowed(src))
    }
}

fn to_gles(src: &str) -> Result<String, StruumError> {
    if contains_word(src, "double") || contains_word(src, "dvec2") || contains_word(src, "dvec3") || contains_word(src, "dvec4") {
        return Err(StruumError::ShaderCreationError(
            "Kernel uses double precision, which OpenGL ES (Android) does not support".to_string(),
        ));
    }

    let mut out = String::with_capacity(src.len() + GLES_HEADER.len());
    let mut replaced = false;
    for line in src.lines() {
        if !replaced && line.trim_start().starts_with("#version") {
            out.push_str(GLES_HEADER);
            replaced = true;
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }

    if !replaced {
        out.insert_str(0, GLES_HEADER);
    }

    Ok(out)
}

fn contains_word(src: &str, word: &str) -> bool {
    src.match_indices(word).any(|(i, _)| {
        let is_ident = |c: char| c.is_ascii_alphanumeric() || c == '_';
        let before = src[..i].chars().next_back().is_none_or(|c| !is_ident(c));
        let after = src[i + word.len()..].chars().next().is_none_or(|c| !is_ident(c));
        before && after
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rewrites_version_and_adds_precision() {
        let src = "\n    #version 430\n\nlayout(local_size_x = 128) in;\nvoid main() {}\n";
        let out = to_gles(src).unwrap();
        assert!(out.starts_with("\n#version 310 es\nprecision highp float;"));
        assert!(!out.contains("430"));
        assert!(out.contains("layout(local_size_x = 128) in;"));
    }

    #[test]
    fn prepends_header_without_version() {
        let out = to_gles("void main() {}").unwrap();
        assert!(out.starts_with(GLES_HEADER));
    }

    #[test]
    fn rejects_double() {
        assert!(to_gles("#version 430\ndouble x[];").is_err());
        assert!(to_gles("#version 430\nfloat doubled;").is_ok());
    }
}
