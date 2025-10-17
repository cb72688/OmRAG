//! By convention, root.zig is the root source file when making a library.
//! This is the entry point for the Omega-RAG Zig libraries.

const std = @import("std");

/// Library version
pub const version = std.SemanticVersion{
    .major = 0,
    .minor = 1,
    .patch = 0,
};

/// Export all math modules with namespacing to avoid conflicts
pub const math = struct {
    // Import as namespaced modules instead of usingnamespace
    pub const vector_ops = @import("math/vector_ops.zig");
    pub const matrix_ops = @import("math/matrix_ops.zig");
    pub const distance = @import("math/distance.zig");
    pub const similarity = @import("math/similarity.zig");

    // Re-export common types with clear names
    pub const Vector = vector_ops.Vector;
    pub const Matrix = matrix_ops.Matrix;
    
    // Re-export operations with prefixed names to avoid conflicts
    pub const vec_add = vector_ops.add;
    pub const vec_sub = vector_ops.subtract;
    pub const vec_mul = vector_ops.multiply;
    pub const vec_scale = vector_ops.scale;
    pub const vec_dot = vector_ops.dotProduct;
    pub const vec_norm = vector_ops.norm;
    pub const vec_normalize = vector_ops.normalize;
    
    pub const mat_add = matrix_ops.add;
    pub const mat_sub = matrix_ops.subtract;
    pub const mat_mul = matrix_ops.multiply;
    pub const mat_scale = matrix_ops.scale;
    pub const mat_transpose = matrix_ops.transpose;
    pub const mat_determinant = matrix_ops.determinant;
    pub const mat_inverse = matrix_ops.inverse;
    
    // Common utility functions
    pub const vec_init = vector_ops.Vector.init;
    pub const mat_init = matrix_ops.Matrix.init;
};

/// Basic addition function for testing infrastructure
pub fn add(a: i32, b: i32) i32 {
    return a + b;
}

test "basic add functionality" {
    try std.testing.expect(add(3, 7) == 10);
}

test "version format" {
    try std.testing.expect(version.major == 0);
    try std.testing.expect(version.minor == 1);
    try std.testing.expect(version.patch == 0);
}

test "math module structure" {
    // Test that we can access both vector and matrix operations without conflicts
    const vec = math.vector_ops;
    const mat = math.matrix_ops;
    
    // These should compile without name conflicts
    _ = vec;
    _ = mat;
    _ = math.vec_add;
    _ = math.mat_add;
}
