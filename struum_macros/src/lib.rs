//! Procedural macros for struum.
//!
//! [`macro@gpu_type`] turns a plain Rust struct into a GPU buffer element: `#[repr(C)]`,
//! `Pod`, and a `GPUType` impl that emits the matching GLSL struct.

use proc_macro::TokenStream;
use quote::quote;
use syn::{ItemStruct, parse_macro_input};

/// Makes a struct usable as a GPU buffer element.
///
/// Adds `#[repr(C)]` and derives `Clone`, `Copy`, `bytemuck::Pod` and `bytemuck::Zeroable`,
/// then implements [`GPUType`](../struum_types/gpu_type/trait.GPUType.html) so the struct is
/// declared in the generated GLSL with the same name and fields. Every field must itself
/// be a `GPUType` (e.g. `f32`, `u32`, `i32`), and the crate using it must depend on
/// `bytemuck` and `struum_types`.
///
/// Keep the layout compatible with GLSL `std430`: prefer fields of 4-byte scalars.
///
/// ```ignore
/// use struum_macros::gpu_type;
///
/// #[gpu_type]
/// struct Vec2 {
///     x: f32,
///     y: f32,
/// }
/// // In GLSL: struct Vec2 { float x; float y; };
/// ```
#[proc_macro_attribute]
pub fn gpu_type(_attr: TokenStream, input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as ItemStruct);

    // Get struct name
    let struct_name = &input.ident;

    // Extract named fields
    let fields = match &input.fields {
        syn::Fields::Named(fields) => &fields.named,

        _ => {
            return syn::Error::new_spanned(
                struct_name,
                "GPUType only supports structs with named fields",
            )
            .to_compile_error()
            .into();
        }
    };

    // Generate GLSL field code
    let mut field_code = Vec::new();

    for field in fields {
        // Field name
        let field_name = match &field.ident {
            Some(name) => name,

            None => {
                return syn::Error::new_spanned(field, "GPUType only supports named fields")
                    .to_compile_error()
                    .into();
            }
        };

        // Field type
        let field_type = &field.ty;

        field_code.push(quote! {
            result.push_str("    ");
            result.push_str(
                <#field_type as ::struum_types::GPUType>::glsl_type()
            );
            result.push(' ');
            result.push_str(stringify!(#field_name));
            result.push_str(";\n");
        });
    }

    // Generate everything
    let expanded = quote! {
        #[repr(C)]
        #[derive(
            Clone,
            Copy,
            ::bytemuck::Pod,
            ::bytemuck::Zeroable,
        )]
        #input

        impl ::struum_types::GPUType for #struct_name {
            fn glsl_type() -> &'static str {
                stringify!(#struct_name)
            }

            fn glsl() -> String {
                let mut result = String::new();

                result.push_str("struct ");
                result.push_str(stringify!(#struct_name));
                result.push_str(" {\n");

                #(#field_code)*

                result.push_str("};");

                result
            }
        }
    };

    expanded.into()
}
