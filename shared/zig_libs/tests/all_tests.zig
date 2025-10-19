const std = @import("std");

// Import your other test files here
comptime {
    _ = @import("test_vector_ops.zig");
    _ = @import("test_matrix_ops.zig");
    _ = @import("test_distance.zig");
    _ = @import("test_similarity.zig");
}
