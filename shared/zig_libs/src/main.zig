const std = @import("std");
const root = @import("omega_rag_zig");  // Changed from "root"

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const stdout = std.io.getStdOut().writer();
    
    try stdout.print("\n", .{});
    try stdout.print("╔════════════════════════════════════════════════════════════════╗\n", .{});
    try stdout.print("║          Omega-RAG Zig Math Libraries v{}.{}.{}              ║\n", .{root.version.major, root.version.minor, root.version.patch});
    try stdout.print("╚════════════════════════════════════════════════════════════════╝\n", .{});
    try stdout.print("\n", .{});

    try stdout.print("Available Modules:\n", .{});
    try stdout.print("  • math.vector_ops  - Vector operations and computations\n", .{});
    try stdout.print("  • math.matrix_ops  - Matrix operations and linear algebra\n", .{});
    try stdout.print("  • math.distance    - Distance metrics (12+ algorithms)\n", .{});
    try stdout.print("  • math.similarity  - Similarity metrics (10+ algorithms)\n", .{});
    try stdout.print("\n", .{});

    try stdout.print("Commands:\n", .{});
    try stdout.print("  zig build              - Build the library\n", .{});
    try stdout.print("  zig build test         - Run all tests with detailed output\n", .{});
    try stdout.print("  zig build benchmark    - Run performance benchmarks\n", .{});
    try stdout.print("  zig build docs         - Generate documentation\n", .{});
    try stdout.print("\n", .{});

    // Quick demonstration
    try stdout.print("Quick Demo:\n", .{});
    try stdout.print("─────────────────────────────────────────────────────────────\n", .{});
    
    // Check if vector operations module exists and has the expected functions
    if (@hasDecl(root.math, "vec_init")) {
        // Vector operations demo using the prefixed names
        var v1 = try root.math.vec_init(allocator, 3);
        defer v1.deinit();
        try v1.set(0, 1.0);
        try v1.set(1, 2.0);
        try v1.set(2, 3.0);

        var v2 = try root.math.vec_init(allocator, 3);
        defer v2.deinit();
        try v2.set(0, 4.0);
        try v2.set(1, 5.0);
        try v2.set(2, 6.0);

        if (@hasDecl(root.math, "vec_dot")) {
            const dot = try root.math.vec_dot(v1, v2);
            try stdout.print("  Vector dot product: [1,2,3] · [4,5,6] = {d:.2}\n", .{dot});
        }

        // Demonstrate vector addition if available
        if (@hasDecl(root.math, "vec_add")) {
            var result_vec = try root.math.vec_add(v1, v2, allocator);
            defer result_vec.deinit();
            try stdout.print("  Vector addition: [1,2,3] + [4,5,6] = ", .{});
            for (0..result_vec.len()) |i| {
                try stdout.print("{d:.1} ", .{try result_vec.get(i)});
            }
            try stdout.print("\n", .{});
        }

        if (@hasDecl(root.math.distance, "euclidean")) {
            const euclidean_dist = try root.math.distance.euclidean(v1, v2);
            try stdout.print("  Euclidean distance: {d:.4}\n", .{euclidean_dist});
        }

        if (@hasDecl(root.math.similarity, "cosine")) {
            const cosine_sim = try root.math.similarity.cosine(v1, v2);
            try stdout.print("  Cosine similarity: {d:.4}\n", .{cosine_sim});
        }
    } else {
        try stdout.print("  Vector operations module not fully implemented yet.\n", .{});
    }

    // Matrix operations demo if available
    if (@hasDecl(root.math, "mat_init")) {
        try stdout.print("\n", .{});
        try stdout.print("Matrix Operations Demo:\n", .{});
        
        var m1 = try root.math.mat_init(allocator, 2, 2);
        defer m1.deinit();
        try m1.set(0, 0, 1.0);
        try m1.set(0, 1, 2.0);
        try m1.set(1, 0, 3.0);
        try m1.set(1, 1, 4.0);

        var m2 = try root.math.mat_init(allocator, 2, 2);
        defer m2.deinit();
        try m2.set(0, 0, 5.0);
        try m2.set(0, 1, 6.0);
        try m2.set(1, 0, 7.0);
        try m2.set(1, 1, 8.0);

        if (@hasDecl(root.math, "mat_add")) {
            var result_mat = try root.math.mat_add(m1, m2, allocator);
            defer result_mat.deinit();
            try stdout.print("  Matrix addition result:\n", .{});
            for (0..result_mat.rows()) |i| {
                try stdout.print("    [ ", .{});
                for (0..result_mat.cols()) |j| {
                    try stdout.print("{d:.1} ", .{try result_mat.get(i, j)});
                }
                try stdout.print("]\n", .{});
            }
        }
    }

    try stdout.print("\n", .{});
    try stdout.print("For more examples, see the tests/ directory.\n", .{});
    try stdout.print("\n", .{});
}
