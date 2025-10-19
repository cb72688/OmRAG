const std = @import("std");
const math = std.math;
const Allocator = std.mem.Allocator;
const Vector = @import("vector_ops.zig").Vector;
const VectorError = @import("vector_ops.zig").VectorError;

/// Error types for similarity operations
pub const SimilarityError = error{
    DimensionMismatch,
    InvalidInput,
    DivisionByZero,
};

/// Calculate cosine similarity between two vectors
/// Returns value in range [-1, 1] where 1 means identical direction
pub fn cosine(v1: Vector, v2: Vector) SimilarityError!f32 {
    if (v1.dimension() != v2.dimension()) {
        return SimilarityError.DimensionMismatch;
    }

    var dot_product: f32 = 0.0;
    var norm1: f32 = 0.0;
    var norm2: f32 = 0.0;

    for (v1.data, v2.data) |a, b| {
        dot_product += a * b;
        norm1 += a * a;
        norm2 += b * b;
    }

    norm1 = @sqrt(norm1);
    norm2 = @sqrt(norm2);

    if (norm1 == 0.0 or norm2 == 0.0) {
        return SimilarityError.DivisionByZero;
    }

    return dot_product / (norm1 * norm2);
}

/// Calculate Pearson correlation coefficient
pub fn pearson(v1: Vector, v2: Vector) SimilarityError!f32 {
    if (v1.dimension() != v2.dimension()) {
        return SimilarityError.DimensionMismatch;
    }

    const n: f32 = @floatFromInt(v1.dimension());

    // Calculate means
    var mean1: f32 = 0.0;
    var mean2: f32 = 0.0;
    for (v1.data) |val| {
        mean1 += val;
    }
    for (v2.data) |val| {
        mean2 += val;
    }
    mean1 /= n;
    mean2 /= n;

    // Calculate correlation
    var cov: f32 = 0.0;
    var var1: f32 = 0.0;
    var var2: f32 = 0.0;

    for (v1.data, v2.data) |a, b| {
        const diff1 = a - mean1;
        const diff2 = b - mean2;
        cov += diff1 * diff2;
        var1 += diff1 * diff1;
        var2 += diff2 * diff2;
    }

    if (var1 == 0.0 or var2 == 0.0) {
        return SimilarityError.DivisionByZero;
    }

    return cov / @sqrt(var1 * var2);
}

/// Calculate Jaccard similarity (for continuous vectors, uses min/max interpretation)
pub fn jaccard(v1: Vector, v2: Vector) SimilarityError!f32 {
    if (v1.dimension() != v2.dimension()) {
        return SimilarityError.DimensionMismatch;
    }

    var intersection: f32 = 0.0;
    var union_sum: f32 = 0.0;

    for (v1.data, v2.data) |a, b| {
        intersection += @min(a, b);
        union_sum += @max(a, b);
    }

    if (union_sum == 0.0) {
        return 0.0;
    }

    return intersection / union_sum;
}
