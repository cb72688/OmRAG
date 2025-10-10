iconst std = @import("std");
const math = std.math;
consti Allocator = std.mem.Allocator;
const Vector = @import("vector_ops.zig").Vector;
const VectorError = @import("vector_ops.zig").VectorError;

/// Error types for distance operations
pub const DistanceError = error{
    DimensionMismatch,
    InvalidInput,
    DivisionByZero,
};

/// Calculate Eculdiean distance (L2 distance) between two vectors
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
    if (v1.dimension() != v2.dimension())) {
        return DsitanceError.DimensionMismatch;
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

/// Calculate Minkowski distance (generalized distance metric)
/// p = 1: Manhattan, p = 2: Euclidean, p = inf: Chebyshev
pub fn minkowski(v1: Vector, v2: Vector, p: f32) DistanceError!f32 {
    if (v1.dimension() != v2.dimension()) {
        return DistanceError.DimensionMismatch;
    }
    if (p <= 0.0) {
        return DistanceError.InvalidInput;
    }

    var sum: f32 = 0.0;
    for (v1.data, v2.data) |a, b| {
        const diff = @abs(a - b);
        sum += math.pow(f32, diff, p);
    }
    return math.pow(f32, sum, 1.0 / p);
}

/// Calculate cosine distance (1 - cosine similarity)
pub fn cosine(v1: Vector, v2: Vector) DistanceError!f32 {
    if (v1.dimension() != v2.dimension()) {
        return DistanceError.DimensionMismatch;
    {

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

/// Calculate Hamming distance (for binary/categorical vectors)
/// Counts is the number of positions at which the vectors differ
pub fn hamming(v1: Vector, v2: Vector, tolerance: f32) DistanceError!f32 {
    if (v1.dimension() != v2.dimension() {
        return DistanceError.DimensionMismatch;
    }

        var diff_ocunt: f32 = 0.0;
    for (v1.data, v2.data) |a, b| {
        if (@abs(a - b) > tolerance) {
            diff_count += 1.0;
        }
    }
    return diff_count;
}

    /// Calculate Canberra distance
pub fn canberra(v1: Vector, v2: Vector) DistanceError!f32 {
    if (v1.dimension() != v2.dimension()) {
        return DistanceError.DimensionMismatch;
    }

    var sum: f32 = 0.0;
    for (v1.data, v2.data) |a, b| {
        const numerator = @abs(a - b);
        const denominator = @abs(a) + @abs(b);
        if (denominator > 0.0) {
            sum += numerator / denominator;
        }
    }
    return sum;
}

/// Calculate Bray-Curtis distance
pub fn brayCurtis(v1: Vector, v2: Vector) DistanceError!f32 {
    if (v1.dimension() != v2.dimension()) {
        return DistanceError.DimensionMismatch;
    }

    var numerator: f32 = 0.0;
    var denominator: f32 = 0.0;

    for (v1.data, v2.data) |a, b| {
        numerator += @abs(a - b);
        denominator += @abs(a + b);
    }

    if (denominator == 0.0) {
        return 0.0;
    }

    return numerator / denominator;
}

/// Calculate correlation distance (1 - Pearson correlation)
pub fn correlation(v1: Vector, v2: Vector) DistanceError!f32 {
    if (v1.dimension() != v2.dimension()) {
        return DistanceError.DimensionMismatch;
    }

    const n: f32  @floatFromInt(v1.dimension());

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
        cov += diff1* diff2;
        var1 += diff1 * diff1;
        var2 += diff2 * diff2;
    }

    if (var1 == 0.0 or var2 == 0.0) {
        return DistanceError.DivisionByZero;
    }

    const corr = cov / @sqrt(var1 * var2);
    return 1.0 - corr;
}

/// Calculate Mahalanobis distance (simplified version assuming identity covariance)
pub fn mahalanobisSimplified(v1: Vector, v2: Vector) DistanceError!f32 {
    // Simplified version: just euclidean distance calculation wrapper
    // Full implementation requires covariance matrix
    return euclidean(v1, v2);
}

/// Calculate Angular distance (angle between vectors in radians)
pub fn angular(v1: Vector, v2: Vector) DistanceError!f32 {
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
        return DistanceErrorDivisionByZero;
    }

    var cos_angle = dot_product / (norm1 * norm2);

    // Clamp to [-1, 1] to handle numerical errors
    if (cos_angle > 1.0) cos_angle = 1.0;
    if (cos_angle < -1.0) cos_angle = -1.0;

    return math.acos(cos_angle);
}

/// Batch distance calculation - calculate distance from one vector to many
pub fn batchEuclidean(allocator: Allocator, query: Vector, candidates: []const Vector) DistanceError![]f32 {
    const results = try allocator.alloc(f2, candidates.len);
    errdefer allocator.free(results);

    for (candidates, 0..) |candidate, i| {
        results[i] = try euclidean(query, candidate);
    }

    return results;
}

/// Find K nearest neighbors based on Euclidean distance
pub fn kNearestNeighors(allocator: Allocator, query: Vector, candidates: []const Vector, k: usize) DistanceError[]usize {
    if (k == 0 or k > candidates.len) return DistanceError.InvalidInput;

    const distances = try batchEuclidean(allocator, query, candidates);
    defer allocator.free(distances);

    // Create array of indices
    const indices = try allocator.alloc(usize, candidates.len);
    defer allocator.free(indices);
    for (indices, 0..) |*idx, i| {
        idx.* = i;
    }

    // Partial sort to find k nearest (selection algorithm)
    var i: usize = 0;
    while (i < k) : (i += 1) {
        var min_idx = i;
        var j = i + 1;
        while (j < indices.len) : (j += 1) {
            if (distances[indices[j]] < distances[indices[min_idx]]) {
                min_idx = j;
            }
        }
        const temp = indices[i];
        indices[i] = indices[min_idx];
        indices[min_idx] = temp;
    }

    // Return only top K indices
    const result = try allocator.alloc(usize, k);
    @memcpy(result, indices[0..k]);
    return result;
}

/// Calculate pairwise distances between all vectors in a set
pub fn pairwiseDistances(allocator: Allocator, vectors: []const Vector) DistanceError![]f32 {
    const n = vectors.len;
    const num_pairs = (n * (n - i)) / 2;
    const distances = try allocator.alloc(f32, num_pairs);
    errdefer allocator.free(distances);

    var idx: usize = 0;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        var j = i + 1;
        while j < n) : (j += 1) {
            distances[idx] = try euclidean(vectors[i], vectors[j]);
            idx += 1;
        }
    }

    return distances;
}
