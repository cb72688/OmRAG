/// shared/zig_libs/test/test_similarity.zig
const std = @import("std");
const testing = std.testing;
const similarity = @import("../src/math/similarity.zig");
const vector_ops = @import("../src/math/vector_ops.zig");
const Vector = vector_ops.Vector;

test "Cosine similarity" {
    const allocator = testing.allocator;
    
    // Identical vectors should have similarity of 1.0
    const v1_vals = [_]f32{ 1.0, 2.0, 3.0 };
    var v1 = try Vector.fromSlice(allocator, &v1_vals);
    defer v1.deinit();
    var v2 = try Vector.fromSlice(allocator, &v1_vals);
    defer v2.deinit();
    
    const sim1 = try similarity.cosine(v1, v2);
    try testing.expectApproxEqAbs(@as(f32, 1.0), sim1, 0.0001);
    
    // Orthogonal vectors should have similarity of 0.0
    const v3_vals = [_]f32{ 1.0, 0.0 };
    const v4_vals = [_]f32{ 0.0, 1.0 };
    var v3 = try Vector.fromSlice(allocator, &v3_vals);
    defer v3.deinit();
    var v4 = try Vector.fromSlice(allocator, &v4_vals);
    defer v4.deinit();
    
    const sim2 = try similarity.cosine(v3, v4);
    try testing.expectApproxEqAbs(@as(f32, 0.0), sim2, 0.0001);
}

test "Pearson correlation" {
    const allocator = testing.allocator;
    
    // Perfect positive correlation
    const v1_vals = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const v2_vals = [_]f32{ 2.0, 4.0, 6.0, 8.0, 10.0 };
    
    var v1 = try Vector.fromSlice(allocator, &v1_vals);
    defer v1.deinit();
    var v2 = try Vector.fromSlice(allocator, &v2_vals);
    defer v2.deinit();
    
    const corr = try similarity.pearson(v1, v2);
    try testing.expectApproxEqAbs(@as(f32, 1.0), corr, 0.0001);
}

test "Jaccard similarity" {
    const allocator = testing.allocator;
    
    const v1_vals = [_]f32{ 1.0, 2.0, 3.0 };
    const v2_vals = [_]f32{ 2.0, 3.0, 4.0 };
    
    var v1 = try Vector.fromSlice(allocator, &v1_vals);
    defer v1.deinit();
    var v2 = try Vector.fromSlice(allocator, &v2_vals);
    defer v2.deinit();
    
    const sim = try similarity.jaccard(v1, v2);
    // min: (1,2,3), max: (2,3,4)
    // intersection: 1+2+3=6, union: 2+3+4=9
    // 6/9 = 0.666...
    try testing.expectApproxEqAbs(@as(f32, 0.666), sim, 0.01);
}

test "Dice coefficient" {
    const allocator = testing.allocator;
    
    const v1_vals = [_]f32{ 1.0, 2.0, 3.0 };
    const v2_vals = [_]f32{ 2.0, 3.0, 4.0 };
    
    var v1 = try Vector.fromSlice(allocator, &v1_vals);
    defer v1.deinit();
    var v2 = try Vector.fromSlice(allocator, &v2_vals);
    defer v2.deinit();
    
    const sim = try similarity.dice(v1, v2);
    // intersection: 1+2+3=6, sum1: 6, sum2: 9
    // 2*6/(6+9) = 12/15 = 0.8
    try testing.expectApproxEqAbs(@as(f32, 0.8), sim, 0.01);
}

test "Overlap coefficient" {
    const allocator = testing.allocator;
    
    const v1_vals = [_]f32{ 1.0, 2.0, 3.0 };
    const v2_vals = [_]f32{ 2.0, 3.0, 4.0 };
    
    var v1 = try Vector.fromSlice(allocator, &v1_vals);
    defer v1.deinit();
    var v2 = try Vector.fromSlice(allocator, &v2_vals);
    defer v2.deinit();
    
    const sim = try similarity.overlap(v1, v2);
    // intersection: 6, min(sum1, sum2) = min(6, 9) = 6
    // 6/6 = 1.0
    try testing.expectApproxEqAbs(@as(f32, 1.0), sim, 0.01);
}

test "Tanimoto coefficient" {
    const allocator = testing.allocator;
    
    const v1_vals = [_]f32{ 1.0, 2.0, 3.0 };
    const v2_vals = [_]f32{ 2.0, 3.0, 4.0 };
    
    var v1 = try Vector.fromSlice(allocator, &v1_vals);
    defer v1.deinit();
    var v2 = try Vector.fromSlice(allocator, &v2_vals);
    defer v2.deinit();
    
    const sim = try similarity.tanimoto(v1, v2);
    // dot: 2+6+12=20, sum1²: 1+4+9=14, sum2²: 4+9+16=29
    // 20/(14+29-20) = 20/23 ≈ 0.870
    try testing.expectApproxEqAbs(@as(f32, 0.870), sim, 0.01);
}

test "Exponential similarity (RBF kernel)" {
    const allocator = testing.allocator;
    
    const v1_vals = [_]f32{ 1.0, 2.0 };
    const v2_vals = [_]f32{ 1.0, 2.0 };
    
    var v1 = try Vector.fromSlice(allocator, &v1_vals);
    defer v1.deinit();
    var v2 = try Vector.fromSlice(allocator, &v2_vals);
    defer v2.deinit();
    
    const sim = try similarity.exponential(v1, v2, 1.0);
    // Same vectors: exp(-0) = 1.0
    try testing.expectApproxEqAbs(@as(f32, 1.0), sim, 0.0001);
    
    const v3_vals = [_]f32{ 2.0, 3.0 };
    var v3 = try Vector.fromSlice(allocator, &v3_vals);
    defer v3.deinit();
    
    const sim2 = try similarity.rbf(v1, v3, 0.5);
    // squared distance: (2-1)² + (3-2)² = 2
    // exp(-0.5 * 2) = exp(-1) ≈ 0.368
    try testing.expectApproxEqAbs(@as(f32, 0.368), sim2, 0.01);
}

test "Polynomial kernel" {
    const allocator = testing.allocator;
    
    const v1_vals = [_]f32{ 1.0, 2.0 };
    const v2_vals = [_]f32{ 3.0, 4.0 };
    
    var v1 = try Vector.fromSlice(allocator, &v1_vals);
    defer v1.deinit();
    var v2 = try Vector.fromSlice(allocator, &v2_vals);
    defer v2.deinit();
    
    const sim = try similarity.polynomial(v1, v2, 2.0, 1.0);
    // dot: 1*3 + 2*4 = 11
    // (11 + 1)² = 144
    try testing.expectApproxEqAbs(@as(f32, 144.0), sim, 0.0001);
}

test "Sigmoid kernel" {
    const allocator = testing.allocator;
    
    const v1_vals = [_]f32{ 1.0, 0.0 };
    const v2_vals = [_]f32{ 1.0, 0.0 };
    
    var v1 = try Vector.fromSlice(allocator, &v1_vals);
    defer v1.deinit();
    var v2 = try Vector.fromSlice(allocator, &v2_vals);
    defer v2.deinit();
    
    const sim = try similarity.sigmoid(v1, v2, 1.0, 0.0);
// dot: 1, tanh(1*1 + 0) = tanh(1) ≈ 0.762
    try testing.expectApproxEqAbs(@as(f32, 0.762), sim, 0.01);
}

test "Bhattacharyya coefficient" {
    const allocator = testing.allocator;
    
    const v1_vals = [_]f32{ 0.25, 0.25, 0.25, 0.25 };
    const v2_vals = [_]f32{ 0.25, 0.25, 0.25, 0.25 };
    
    var v1 = try Vector.fromSlice(allocator, &v1_vals);
    defer v1.deinit();
    var v2 = try Vector.fromSlice(allocator, &v2_vals);
    defer v2.deinit();
    
    const sim = try similarity.bhattacharyya(v1, v2);
    // sqrt(0.25*0.25) * 4 = 0.25 * 4 = 1.0
    try testing.expectApproxEqAbs(@as(f32, 1.0), sim, 0.0001);
}

test "Hellinger similarity" {
    const allocator = testing.allocator;
    
    const v1_vals = [_]f32{ 0.25, 0.25, 0.25, 0.25 };
    const v2_vals = [_]f32{ 0.25, 0.25, 0.25, 0.25 };
    
    var v1 = try Vector.fromSlice(allocator, &v1_vals);
    defer v1.deinit();
    var v2 = try Vector.fromSlice(allocator, &v2_vals);
    defer v2.deinit();
    
    const sim = try similarity.hellinger(v1, v2);
    // Identical distributions should have similarity 1.0
    try testing.expectApproxEqAbs(@as(f32, 1.0), sim, 0.0001);
}

test "Batch cosine similarity" {
    const allocator = testing.allocator;
    
    const query_vals = [_]f32{ 1.0, 0.0 };
    var query = try Vector.fromSlice(allocator, &query_vals);
    defer query.deinit();
    
    const c1_vals = [_]f32{ 1.0, 0.0 };  // Same direction
    const c2_vals = [_]f32{ 0.0, 1.0 };  // Orthogonal
    const c3_vals = [_]f32{ 0.707, 0.707 };  // 45 degrees
    
    var c1 = try Vector.fromSlice(allocator, &c1_vals);
    defer c1.deinit();
    var c2 = try Vector.fromSlice(allocator, &c2_vals);
    defer c2.deinit();
    var c3 = try Vector.fromSlice(allocator, &c3_vals);
    defer c3.deinit();
    
    const candidates = [_]Vector{ c1, c2, c3 };
    
    const similarities = try similarity.batchCosine(allocator, query, &candidates);
    defer allocator.free(similarities);
    
    try testing.expectEqual(@as(usize, 3), similarities.len);
    try testing.expectApproxEqAbs(@as(f32, 1.0), similarities[0], 0.0001);
    try testing.expectApproxEqAbs(@as(f32, 0.0), similarities[1], 0.0001);
    try testing.expectApproxEqAbs(@as(f32, 0.707), similarities[2], 0.01);
}

test "Top K similar vectors" {
    const allocator = testing.allocator;
    
    const query_vals = [_]f32{ 1.0, 0.0 };
    var query = try Vector.fromSlice(allocator, &query_vals);
    defer query.deinit();
    
    const c1_vals = [_]f32{ 0.9, 0.1 };   // High similarity
    const c2_vals = [_]f32{ 0.0, 1.0 };   // Low similarity
    const c3_vals = [_]f32{ 0.95, 0.05 }; // Higher similarity
    const c4_vals = [_]f32{ 0.3, 0.7 };   // Medium similarity
    
    var c1 = try Vector.fromSlice(allocator, &c1_vals);
    defer c1.deinit();
    var c2 = try Vector.fromSlice(allocator, &c2_vals);
    defer c2.deinit();
    var c3 = try Vector.fromSlice(allocator, &c3_vals);
    defer c3.deinit();
    var c4 = try Vector.fromSlice(allocator, &c4_vals);
    defer c4.deinit();
    
    const candidates = [_]Vector{ c1, c2, c3, c4 };
    
    const top_k = try similarity.topKSimilar(allocator, query, &candidates, 2);
    defer allocator.free(top_k);
    
    try testing.expectEqual(@as(usize, 2), top_k.len);
    // Should return indices 2 and 0 (highest similarities)
    try testing.expect(top_k[0] == 2 or top_k[0] == 0);
    try testing.expect(top_k[1] == 2 or top_k[1] == 0);
    try testing.expect(top_k[0] != top_k[1]);
}

test "Pairwise similarities" {
    const allocator = testing.allocator;
    
    const v1_vals = [_]f32{ 1.0, 0.0 };
    const v2_vals = [_]f32{ 0.0, 1.0 };
    const v3_vals = [_]f32{ 1.0, 0.0 };
    
    var v1 = try Vector.fromSlice(allocator, &v1_vals);
    defer v1.deinit();
    var v2 = try Vector.fromSlice(allocator, &v2_vals);
    defer v2.deinit();
    var v3 = try Vector.fromSlice(allocator, &v3_vals);
    defer v3.deinit();
    
    const vectors = [_]Vector{ v1, v2, v3 };
    
    const similarities = try similarity.pairwiseSimilarities(allocator, &vectors);
    defer allocator.free(similarities);
    
    // 3 vectors = 3 pairs: (0,1), (0,2), (1,2)
    try testing.expectEqual(@as(usize, 3), similarities.len);
    try testing.expectApproxEqAbs(@as(f32, 0.0), similarities[0], 0.0001); // v1-v2 orthogonal
    try testing.expectApproxEqAbs(@as(f32, 1.0), similarities[1], 0.0001); // v1-v3 identical
    try testing.expectApproxEqAbs(@as(f32, 0.0), similarities[2], 0.0001); // v2-v3 orthogonal
}

test "Similarity to distance conversion" {
    const sim = 0.8;
    const dist = similarity.similarityToDistance(sim);
    try testing.expectApproxEqAbs(@as(f32, 0.2), dist, 0.0001);
    
    const sim2 = 1.0;
    const dist2 = similarity.similarityToDistance(sim2);
    try testing.expectApproxEqAbs(@as(f32, 0.0), dist2, 0.0001);
}

test "Distance to similarity conversion" {
    const dist = 1.0;
    const sim = similarity.distanceToSimilarity(dist);
    try testing.expectApproxEqAbs(@as(f32, 0.5), sim, 0.0001);
    
    const dist2 = 0.0;
    const sim2 = similarity.distanceToSimilarity(dist2);
    try testing.expectApproxEqAbs(@as(f32, 1.0), sim2, 0.0001);
}

test "Normalize similarities" {
    var scores = [_]f32{ 0.2, 0.5, 0.8, 0.3 };
    
    similarity.normalizeSimilarities(&scores);
    
    // Min: 0.2, Max: 0.8, Range: 0.6
    try testing.expectApproxEqAbs(@as(f32, 0.0), scores[0], 0.0001);    // (0.2-0.2)/0.6
    try testing.expectApproxEqAbs(@as(f32, 0.5), scores[1], 0.0001);    // (0.5-0.2)/0.6
    try testing.expectApproxEqAbs(@as(f32, 1.0), scores[2], 0.0001);    // (0.8-0.2)/0.6
    try testing.expectApproxEqAbs(@as(f32, 0.167), scores[3], 0.01);    // (0.3-0.2)/0.6
}

test "Normalize similarities - all same values" {
    var scores = [_]f32{ 0.5, 0.5, 0.5 };
    
    similarity.normalizeSimilarities(&scores);
    
    // All should become 1.0 when range is 0
    try testing.expectApproxEqAbs(@as(f32, 1.0), scores[0], 0.0001);
    try testing.expectApproxEqAbs(@as(f32, 1.0), scores[1], 0.0001);
    try testing.expectApproxEqAbs(@as(f32, 1.0), scores[2], 0.0001);
}

test "Softmax transformation" {
    const allocator = testing.allocator;
    
    const scores = [_]f32{ 1.0, 2.0, 3.0 };
    
    const result = try similarity.softmax(allocator, &scores);
    defer allocator.free(result);
    
    // Check that results sum to 1.0
    var sum: f32 = 0.0;
    for (result) |val| {
        sum += val;
    }
    try testing.expectApproxEqAbs(@as(f32, 1.0), sum, 0.0001);
    
    // Check that higher scores get higher probabilities
    try testing.expect(result[0] < result[1]);
    try testing.expect(result[1] < result[2]);
}

test "Error handling - dimension mismatch" {
    const allocator = testing.allocator;
    
    const v1_vals = [_]f32{ 1.0, 2.0 };
    const v2_vals = [_]f32{ 1.0, 2.0, 3.0 };
    
    var v1 = try Vector.fromSlice(allocator, &v1_vals);
    defer v1.deinit();
    var v2 = try Vector.fromSlice(allocator, &v2_vals);
    defer v2.deinit();
    
    try testing.expectError(similarity.SimilarityError.DimensionMismatch, similarity.cosine(v1, v2));
    try testing.expectError(similarity.SimilarityError.DimensionMismatch, similarity.pearson(v1, v2));
    try testing.expectError(similarity.SimilarityError.DimensionMismatch, similarity.jaccard(v1, v2));
}

test "Error handling - zero vectors" {
    const allocator = testing.allocator;
    
    const v1_vals = [_]f32{ 0.0, 0.0 };
    const v2_vals = [_]f32{ 1.0, 2.0 };
    
    var v1 = try Vector.fromSlice(allocator, &v1_vals);
    defer v1.deinit();
    var v2 = try Vector.fromSlice(allocator, &v2_vals);
    defer v2.deinit();
    
    try testing.expectError(similarity.SimilarityError.DivisionByZero, similarity.cosine(v1, v2));
}

test "Error handling - invalid gamma for exponential" {
    const allocator = testing.allocator;
    
    const v1_vals = [_]f32{ 1.0, 2.0 };
    const v2_vals = [_]f32{ 3.0, 4.0 };
    
    var v1 = try Vector.fromSlice(allocator, &v1_vals);
    defer v1.deinit();
    var v2 = try Vector.fromSlice(allocator, &v2_vals);
    defer v2.deinit();
    
    try testing.expectError(similarity.SimilarityError.InvalidInput, similarity.exponential(v1, v2, 0.0));
    try testing.expectError(similarity.SimilarityError.InvalidInput, similarity.exponential(v1, v2, -1.0));
}

test "Error handling - invalid K for top K" {
    const allocator = testing.allocator;
    
    const query_vals = [_]f32{ 1.0, 0.0 };
    var query = try Vector.fromSlice(allocator, &query_vals);
    defer query.deinit();
    
    const c1_vals = [_]f32{ 0.9, 0.1 };
    var c1 = try Vector.fromSlice(allocator, &c1_vals);
    defer c1.deinit();
    
    const candidates = [_]Vector{c1};
    
    try testing.expectError(similarity.SimilarityError.InvalidInput, similarity.topKSimilar(allocator, query, &candidates, 0));
    try testing.expectError(similarity.SimilarityError.InvalidInput, similarity.topKSimilar(allocator, query, &candidates, 5));
}

// Performance benchmark test
test "Performance - similarity calculations on large vectors" {
    const allocator = testing.allocator;
    
    const dim = 768; // Common embedding dimension
    
    var v1 = try Vector.init(allocator, dim);
    defer v1.deinit();
    var v2 = try Vector.init(allocator, dim);
    defer v2.deinit();
    
    // Fill with sample data
    var i: usize = 0;
    while (i < dim) : (i += 1) {
        try v1.set(i, @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(dim)));
        try v2.set(i, @as(f32, @floatFromInt(i + 1)) / @as(f32, @floatFromInt(dim)));
    }
    
    const start = std.time.nanoTimestamp();
    _ = try similarity.cosine(v1, v2);
    const end = std.time.nanoTimestamp();
    
    const duration_us = @as(f64, @floatFromInt(end - start)) / 1000.0;
    std.debug.print("\nCosine similarity for {d}-dim vectors: {d:.2} μs\n", .{ dim, duration_us });
    
    try testing.expect(duration_us < 1000.0); // Should complete in < 1ms
}

test "Performance - batch similarity calculation" {
    const allocator = testing.allocator;
    
    const dim = 384;
    const num_candidates = 100;
    
    // Create query vector
    var query = try Vector.init(allocator, dim);
    defer query.deinit();
    var i: usize = 0;
    while (i < dim) : (i += 1) {
        try query.set(i, @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(dim)));
    }
    
    // Create candidate vectors
    var candidates = try allocator.alloc(Vector, num_candidates);
    defer {
        for (candidates) |*c| c.deinit();
        allocator.free(candidates);
    }
    
    var j: usize = 0;
    while (j < num_candidates) : (j += 1) {
        candidates[j] = try Vector.init(allocator, dim);
        var k: usize = 0;
        while (k < dim) : (k += 1) {
            const val = @as(f32, @floatFromInt((j + k) % dim)) / @as(f32, @floatFromInt(dim));
            try candidates[j].set(k, val);
        }
    }
    
    const start = std.time.nanoTimestamp();
    const results = try similarity.batchCosine(allocator, query, candidates);
    const end = std.time.nanoTimestamp();
    defer allocator.free(results);
    
    const duration_ms = @as(f64, @floatFromInt(end - start)) / 1_000_000.0;
    std.debug.print("\nBatch cosine similarity for {d} {d}-dim vectors: {d:.2} ms\n", .{ num_candidates, dim, duration_ms });
    
    try testing.expectEqual(num_candidates, results.len);
    try testing.expect(duration_ms < 100.0); // Should complete in < 100ms
}

test "Real-world scenario - document similarity" {
    const allocator = testing.allocator;
    
    // Simulate document embeddings (simplified)
    const doc1_vals = [_]f32{ 0.1, 0.5, 0.3, 0.7, 0.2 };
    const doc2_vals = [_]f32{ 0.15, 0.48, 0.32, 0.69, 0.18 }; // Similar to doc1
    const doc3_vals = [_]f32{ 0.9, 0.1, 0.05, 0.2, 0.8 };      // Different from doc1
    
    var doc1 = try Vector.fromSlice(allocator, &doc1_vals);
    defer doc1.deinit();
    var doc2 = try Vector.fromSlice(allocator, &doc2_vals);
    defer doc2.deinit();
    var doc3 = try Vector.fromSlice(allocator, &doc3_vals);
    defer doc3.deinit();
    
    const sim_1_2 = try similarity.cosine(doc1, doc2);
    const sim_1_3 = try similarity.cosine(doc1, doc3);
    
    // doc1 and doc2 should be more similar than doc1 and doc3
    try testing.expect(sim_1_2 > sim_1_3);
    try testing.expect(sim_1_2 > 0.9); // Should be highly similar
    try testing.expect(sim_1_3 < 0.7); // Should be less similar
}
