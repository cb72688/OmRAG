//! By convention, root.zig is the root source file when making a library.
//! This is the entry point for the Omega-RAG Zig libraries.

const std = @import("std");

/// Library version
pub const version = std.SemanticVersion{
    .major = 0,
    .minor = 1,
    .patch = 0,
};

/// Export all math modules
pub const math = struct {
    pub const vector_ops = @import("math/vector_ops.zig");
    pub const matrix_ops = @import("math/matrix_ops.zig");
    pub const distance = @import("math/distance.zig");
    pub const similarity = @import("math/similarity.zig");
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
