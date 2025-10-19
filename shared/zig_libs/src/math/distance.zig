const std = @import("std");
const math = std.math;
const Allocator = std.mem.Allocator;
const Vector = @import("vector_ops.zig").Vector;
const VectorError = @import("vector_ops.zig").VectorError;

/// Error types for distance operations
pub const DistanceError = error{
    DimensionMismatch,
    InvalidInput,
    DivisionByZero,
};

/// Calculate Euclidean distance (L2 distance) between two vectors
pub fn euclidean(v1: Vector, v2: Vector) DistanceError!f32 {
    if (v1.dimension() != v2.dimension()) {
        return DistanceError.DimensionMismatch;
    }

    var sum: f32 = 0.0;
    for (v1.data, v2.data) |a, b| {
        const diff = a - b;
        sum += diff * diff;
    }
    return @sqrt(sum);
}

/// Calculate squared Euclidean distance (more efficient, avoids sqrt)
pub fn euclideanSquared(v1: Vector, v2: Vector) DistanceError!f32 {
    if (v1.dimension() != v2.dimension()) {
        return DistanceError.DimensionMismatch;
    }

    var sum: f32 = 0.0;
    for (v1.data, v2.data) |a, b| {
        const diff = a - b;
        sum += diff * diff;
    }
    return sum;
}

/// Calculate Manhattan distance (L1 distance)
pub fn manhattan(v1: Vector, v2: Vector) DistanceError!f32 {
    if (v1.dimension() != v2.dimension()) {
        return DistanceError.DimensionMismatch;
    }

    var sum: f32 = 0.0;
    for (v1.data, v2.data) |a, b| {
        sum += @abs(a - b);
    }
    return sum;
}

/// Calculate Chebyshev distance (L-infinity distance)
pub fn chebyshev(v1: Vector, v2: Vector) DistanceError!f32 {
    if (v1.dimension() != v2.dimension()) {
        return DistanceError.DimensionMismatch;
    }

    var max_diff: f32 = 0.0;
    for (v1.data, v2.data) |a, b| {
        const diff = @abs(a - b);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

/// Calculate cosine distance (1 - cosine similarity)
pub fn cosine(v1: Vector, v2: Vector) DistanceError!f32 {
    if (v1.dimension() != v2.dimension()) {
        return DistanceError.DimensionMismatch;
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
        return DistanceError.DivisionByZero;
    }

    const similarity = dot_product / (norm1 * norm2);
    return 1.0 - similarity;
}
