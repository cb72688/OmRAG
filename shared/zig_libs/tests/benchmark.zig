const std = @import("std");
const root = @import("omega_rag");
const stdout_file = std.io.getStdOut();
const stdout = stdout_file.writer();

const BenchmarkResult = struct {
    name: []const u8,
    iterations: usize,
    total_time_ns: i64,
    avg_time_ns: f64,
    ops_per_sec: f64,
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    try stdout.print("\n", .{});
    try stdout.print("╔══════════════════════════════════════════════════════════════════╗\n", .{});
    try stdout.print("║            Omega-RAG Zig Libraries Benchmarks                    ║\n", .{});
    try stdout.print("╚══════════════════════════════════════════════════════════════════╝\n", .{});
    try stdout.print("\n", .{});

    var results = std.ArrayList(BenchmarkResult).init(allocator);
    defer results.deinit();

    // Run benchmarks
    try benchmarkVectorOperations(allocator, &results);
    try benchmarkMatrixOperations(allocator, &results);
    try benchmarkDistanceMetrics(allocator, &results);
    try benchmarkSimilarityMetrics(allocator, &results);

    // Print results
    try printBenchmarkResults(results.items, stdout);
}

fn benchmarkVectorOperations(allocator: std.mem.Allocator, results: *std.ArrayList(BenchmarkResult)) !void {
    const vector_ops = root.math.vector_ops;
    const iterations: usize = 10000;
    const dim: usize = 512;

    var v1 = try vector_ops.Vector.init(allocator, dim);
    defer v1.deinit();
    var v2 = try vector_ops.Vector.init(allocator, dim);
    defer v2.deinit();

    // Fill vectors
    var i: usize = 0;
    while (i < dim) : (i += 1) {
        try v1.set(i, @as(f32, @floatFromInt(i)));
        try v2.set(i, @as(f32, @floatFromInt(i + 1)));
    }

    // Benchmark dot product
    const start = std.time.nanoTimestamp();
    var j: usize = 0;
    while (j < iterations) : (j += 1) {
        _ = try vector_ops.dotProduct(v1, v2);
    }
    const end = std.time.nanoTimestamp();
    const duration = end - start;

    const avg_ns = @as(f64, @floatFromInt(duration)) / @as(f64, @floatFromInt(iterations));
    const ops_per_sec = 1_000_000_000.0 / avg_ns;

    try results.append(.{
        .name = "Vector dot product (512-dim)",
        .iterations = iterations,
        .total_time_ns = duration,
        .avg_time_ns = avg_ns,
        .ops_per_sec = ops_per_sec,
    });
}

fn benchmarkMatrixOperations(allocator: std.mem.Allocator, results: *std.ArrayList(BenchmarkResult)) !void {
    const matrix_ops = root.math.matrix_ops;
    const iterations: usize = 1000;

    var m1 = try matrix_ops.Matrix.init(allocator, 100, 100);
    defer m1.deinit();
    var m2 = try matrix_ops.Matrix.init(allocator, 100, 100);
    defer m2.deinit();

    const start = std.time.nanoTimestamp();
    var i: usize = 0;
    while (i < iterations) : (i += 1) {
        var result = try matrix_ops.multiply(allocator, m1, m2);
        result.deinit();
    }
    const end = std.time.nanoTimestamp();
    const duration = end - start;

    const avg_ns = @as(f64, @floatFromInt(duration)) / @as(f64, @floatFromInt(iterations));
    const ops_per_sec = 1_000_000_000.0 / avg_ns;

    try results.append(.{
        .name = "Matrix multiplication (100x100)",
        .iterations = iterations,
        .total_time_ns = duration,
        .avg_time_ns = avg_ns,
        .ops_per_sec = ops_per_sec,
    });
}

fn benchmarkDistanceMetrics(allocator: std.mem.Allocator, results: *std.ArrayList(BenchmarkResult)) !void {
    const vector_ops = root.math.vector_ops;
    const distance = root.math.distance;
    const iterations: usize = 10000;
    const dim: usize = 384;

    var v1 = try vector_ops.Vector.init(allocator, dim);
    defer v1.deinit();
    var v2 = try vector_ops.Vector.init(allocator, dim);
    defer v2.deinit();

    const start = std.time.nanoTimestamp();
    var i: usize = 0;
    while (i < iterations) : (i += 1) {
        _ = try distance.euclidean(v1, v2);
    }
    const end = std.time.nanoTimestamp();
    const duration = end - start;

    const avg_ns = @as(f64, @floatFromInt(duration)) / @as(f64, @floatFromInt(iterations));
    const ops_per_sec = 1_000_000_000.0 / avg_ns;

    try results.append(.{
        .name = "Euclidean distance (384-dim)",
        .iterations = iterations,
        .total_time_ns = duration,
        .avg_time_ns = avg_ns,
        .ops_per_sec = ops_per_sec,
    });
}

fn benchmarkSimilarityMetrics(allocator: std.mem.Allocator, results: *std.ArrayList(BenchmarkResult)) !void {
    const vector_ops = root.math.vector_ops;
    const similarity = root.math.similarity;
    const iterations: usize = 10000;
    const dim: usize = 768;

    var v1 = try vector_ops.Vector.init(allocator, dim);
    defer v1.deinit();
    var v2 = try vector_ops.Vector.init(allocator, dim);
    defer v2.deinit();

    const start = std.time.nanoTimestamp();
    var i: usize = 0;
    while (i < iterations) : (i += 1) {
        _ = try similarity.cosine(v1, v2);
    }
    const end = std.time.nanoTimestamp();
    const duration = end - start;

    const avg_ns = @as(f64, @floatFromInt(duration)) / @as(f64, @floatFromInt(iterations));
    const ops_per_sec = 1_000_000_000.0 / avg_ns;

    try results.append(.{
        .name = "Cosine similarity (768-dim)",
        .iterations = iterations,
        .total_time_ns = duration,
        .avg_time_ns = avg_ns,
        .ops_per_sec = ops_per_sec,
    });
}

fn printBenchmarkResults(results: []const BenchmarkResult, writer: anytype) !void {
    try writer.print("\n", .{});
    try writer.print("┌──────────────────────────────────────────┬────────────┬─────────────┬─────────────────┐\n", .{});
    try writer.print("│ Benchmark                                │ Iterations │  Avg Time   │   Ops/Second    │\n", .{});
    try writer.print("├──────────────────────────────────────────┼────────────┼─────────────┼─────────────────┤\n", .{});

    for (results) |result| {
        const avg_us = result.avg_time_ns / 1000.0;
        try writer.print("│ {s: <40} │ {d: >10} │ {d: >8.3} μs │ {d: >15.0} │\n", .{
            result.name,
            result.iterations,
            avg_us,
            result.ops_per_sec,
        });
    }

    try writer.print("└──────────────────────────────────────────┴────────────┴─────────────┴─────────────────┘\n", .{});
    try writer.print("\n", .{});
}
