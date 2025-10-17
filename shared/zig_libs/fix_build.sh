#!/bin/bash

echo "Checking Zig version..."
ZIG_VERSION=$(zig version)
echo "Zig version: $ZIG_VERSION"

# Backup current build.zig
cp build.zig build.zig.backup 2>/dev/null || true

# Create build.zig for Zig 0.16-dev
cat > build.zig << 'EOF'
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Create a module for the library
    const omega_rag_module = b.addModule("omega_rag_zig", .{
        .root_source_file = b.path("src/root.zig"),
    });

    // Create the executable
    const exe = b.addExecutable(.{
        .name = "omega_rag_demo",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe.root_module.addImport("root", omega_rag_module);
    b.installArtifact(exe);

    // Run step
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run the demo application");
    run_step.dependOn(&run_cmd.step);

    // Test step
    const test_step = b.step("test", "Run all unit tests");
    
    const test_files = [_][]const u8{
        "tests/test_vector_ops.zig",
        "tests/test_matrix_ops.zig",
        "tests/test_distance.zig",
        "tests/test_similarity.zig",
    };

    for (test_files) |test_file| {
        const unit_tests = b.addTest(.{
            .root_source_file = b.path(test_file),
            .target = target,
            .optimize = optimize,
        });

        const run_tests = b.addRunArtifact(unit_tests);
        test_step.dependOn(&run_tests.step);
    }

    // Test runner
    const test_runner = b.addExecutable(.{
        .name = "test_runner",
        .root_source_file = b.path("tests/test_runner.zig"),
        .target = target,
        .optimize = optimize,
    });
    test_runner.root_module.addImport("root", omega_rag_module);
    b.installArtifact(test_runner);

    const run_test_runner = b.addRunArtifact(test_runner);
    const test_verbose_step = b.step("test-verbose", "Run tests with verbose output");
    test_verbose_step.dependOn(&run_test_runner.step);

    // Benchmark
    const benchmark_exe = b.addExecutable(.{
        .name = "benchmark",
        .root_source_file = b.path("tests/benchmark.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    benchmark_exe.root_module.addImport("root", omega_rag_module);
    b.installArtifact(benchmark_exe);
    
    const run_benchmark = b.addRunArtifact(benchmark_exe);
    const benchmark_step = b.step("benchmark", "Run performance benchmarks");
    benchmark_step.dependOn(&run_benchmark.step);

    // Documentation
    const docs_step = b.step("docs", "Generate documentation");
    const docs_obj = b.addObject(.{
        .name = "omega_rag_zig",
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = .Debug,
    });
    const install_docs = b.addInstallDirectory(.{
        .source_dir = docs_obj.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "docs",
    });
    docs_step.dependOn(&install_docs.step);
}
EOF

echo "✓ Updated build.zig for Zig 0.16-dev"
echo ""
echo "Testing build..."
if zig build; then
    echo ""
    echo "✓ Build successful!"
    echo ""
    echo "Available commands:"
    echo "  zig build           - Build the library"
    echo "  zig build run       - Run the demo"
    echo "  zig build test      - Run tests"
    echo "  zig build benchmark - Run benchmarks"
    echo "  zig build docs      - Generate docs"
else
    echo ""
    echo "✗ Build failed"
    exit 1
fi
