const std = @import("std");
const omega_rag = @import("omega_rag");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const stdout_file = std.io.getStdOut();
    const stdout = stdout_file.writer();
    
    try stdout.print("\n", .{});
    try stdout.print("╔════════════════════════════════════════════════════════════════╗\n", .{});
    try stdout.print("║          Omega-RAG Zig Math Libraries v{}.{}.{}              ║\n", .{
        omega_rag.version.major, 
        omega_rag.version.minor, 
        omega_rag.version.patch
    });
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
    try stdout.print("  zig build test-verbose - Run tests with verbose runner\n", .{});
    try stdout.print("  zig build benchmark    - Run performance benchmarks\n", .{});
    try stdout.print("  zig build docs         - Generate documentation\n", .{});
    try stdout.print("\n", .{});

    // Quick demonstration
    try stdout.print("Quick Demo:\n", .{});
    try stdout.print("─────────────────────────────────────────────────────────────────\n", .{});
    
    // Vector operations demo
    var v1 = try omega_rag.math.vector_ops.Vector.init(allocator, 3);
    defer v1.deinit();
    try v1.set(0, 1.0);
    try v1.set(1, 2.0);
    try v1.set(2, 3.0);

    var v2 = try omega_rag.math.vector_ops.Vector.init(allocator, 3);
    defer v2.deinit();
    try v2.set(0, 4.0);
    try v2.set(1, 5.0);
    try v2.set(2, 6.0);

    const dot = try omega_rag.math.vector_ops.dotProduct(v1, v2);
    try stdout.print("  Vector dot product: [1,2,3] · [4,5,6] = {d:.2}\n", .{dot});

    // Vector addition
    var result_vec = try omega_rag.math.vector_ops.add(allocator, v1, v2);
    defer result_vec.deinit();
    try stdout.print("  Vector addition: [1,2,3] + [4,5,6] = [", .{});
    var i: usize = 0;
    while (i < result_vec.dimension()) : (i += 1) {
        if (i > 0) try stdout.print(", ", .{});
        try stdout.print("{d:.1}", .{try result_vec.get(i)});
    }
    try stdout.print("]\n", .{});

    const euclidean_dist = try omega_rag.math.distance.euclidean(v1, v2);
    try stdout.print("  Euclidean distance: {d:.4}\n", .{euclidean_dist});

    const cosine_sim = try omega_rag.math.similarity.cosine(v1, v2);
    try stdout.print("  Cosine similarity: {d:.4}\n", .{cosine_sim});

    try stdout.print("\n", .{});
    try stdout.print("For more examples, see the tests/ directory.\n", .{});
    try stdout.print("\n", .{});
}
