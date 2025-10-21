/// shared/zig_libs/src/math/matrix_ops.zig

const std = @import("std");
const math = std.math;
const Allocator = std.mem.Allocator;
const Vector = @import("vector_ops.zig").Vector;
const VectorError = @import("vector_ops.zig").VectorError;

/// Error types for matrix operations
pub const MatrixError = error{
    DimensionMismatch,
    InvalidDimension,
    SingularMatrix,
    AllocationFailed,
    InvalidInput,
    NotSquareMatrix,
};

/// Matrix structure for mathematical operations
pub const Matrix = struct {
    data: []f32,
    rows_count: usize,
    cols_count: usize,
    allocator: Allocator,

    /// Initialize a new matrix with given dimensions
    pub fn init(allocator: Allocator, row_count: usize, col_count: usize) !Matrix {
        if (row_count == 0 or col_count == 0) return MatrixError.InvalidDimension;

        const data = try allocator.alloc(f32, row_count * col_count);
        @memset(data, 0.0);

        return Matrix{
            .data = data,
            .rows_count = row_count,
            .cols_count = col_count,
            .allocator = allocator,
        };
    }

    /// Initialize matrix from 2D slice
    pub fn fromValues(allocator: Allocator, values: []const []const f32) !Matrix {
        if (values.len == 0) return MatrixError.InvalidDimension;
        const nrows = values.len;
        const ncols = values[0].len;
        if (ncols == 0) return MatrixError.InvalidDimension;

        // Verify all rows have same length
        for (values) |nrow| {
            if (nrow.len != ncols) return MatrixError.InvalidDimension;
        }

        var matrix = try Matrix.init(allocator, nrows, ncols);

        for (values, 0..) |nrow, i| {
            for (nrow, 0..) |val, j| {
                try matrix.set(i, j, val);
            }
        }

        return matrix;
    }

    /// Initialize identity matrix
    pub fn identity(allocator: Allocator, sz: usize) !Matrix {
        var matrix = try Matrix.init(allocator, sz, sz);

        var i: usize = 0;
        while (i < sz) : (i += 1) {
            try matrix.set(i, i, 1.0);
        }

        return matrix;
    }

    /// Free matrix memory
    pub fn deinit(self: *Matrix) void {
        self.allocator.free(self.data);
    }

    /// Get number of rows
    pub fn rows(self: Matrix) usize {
        return self.rows_count;
    }

    /// Get number of columns
    pub fn cols(self: Matrix) usize {
        return self.cols_count;
    }

    /// Get value at position (row, col)
    pub fn get(self: Matrix, row_len: usize, col_len: usize) !f32 {
        if (row_len >= self.rows_count or col_len >= self.cols_count) return MatrixError.InvalidInput;
        return self.data[row_len * self.cols_count + col_len];
    }

    /// Set value at position (row, col)
    pub fn set(self: *Matrix, row_pos: usize, col_pos: usize, value_pos: f32) !void {
        if (row_pos >= self.rows_count or col_pos >= self.cols_count) return MatrixError.InvalidInput;
        self.data[row_pos * self.cols_count + col_pos] = value_pos;
    }

    /// Get a row as a vector
    pub fn getRow(self: Matrix, allocator: Allocator, row_vec: usize) !Vector {
        if (row_vec >= self.rows_count) return MatrixError.InvalidInput;

        var vec = try Vector.init(allocator, self.cols_count);
        var col_vec: usize = 0;
        while (col_vec < self.cols_count) : (col_vec += 1) {
            try vec.set(col_vec, try self.get(row_vec, col_vec));
        }
        return vec;
    }

    /// Get a column as a vector
    pub fn getColumn(self: Matrix, allocator: Allocator, colVec: usize) !Vector {
        if (colVec >= self.cols_count) return MatrixError.InvalidInput;

        var vec = try Vector.init(allocator, self.rows_count);
        var rowVec: usize = 0;
        while (rowVec < self.rows_count) : (rowVec += 1) {
            try vec.set(rowVec, try self.get(rowVec, colVec));
        }
        return vec;
    }

    /// Check if matrix is square
    pub fn isSquare(self: Matrix) bool {
        return self.rows_count == self.cols_count;
    }
};

/// Add two matrices
pub fn add(allocator: Allocator, m1: Matrix, m2: Matrix) MatrixError!Matrix {
    if (m1.rows_count != m2.rows_count or m1.cols_count != m2.cols_count) {
        return MatrixError.DimensionMismatch;
    }

    var result = try Matrix.init(allocator, m1.rows_count, m1.cols_count);

    for (m1.data, m2.data, result.data) |a, b, *r| {
        r.* = a + b;
    }

    return result;
}

/// Subtract two matrices (m1 - m2)
pub fn subtract(allocator: Allocator, m1: Matrix, m2: Matrix) MatrixError!Matrix {
    if (m1.rows_count != m2.rows_count or m1.cols_count != m2.cols_count) {
        return MatrixError.DimensionMismatch;
    }

    var result = try Matrix.init(allocator, m1.rows_count, m1.cols_count);

    for (m1.data, m2.data, result.data) |a, b, *r| {
        r.* = a - b;
    }

    return result;
}

/// Multiply matrix by scalar
pub fn scale(allocator: Allocator, m: Matrix, scalar: f32) !Matrix {
    var result = try Matrix.init(allocator, m.rows_count, m.cols_count);

    for (m.data, result.data) |val, *r| {
        r.* = val * scalar;
    }

    return result;
}

/// Matrix multiplication (m1 * m2)
pub fn multiply(allocator: Allocator, m1: Matrix, m2: Matrix) MatrixError!Matrix {
    if (m1.cols_count != m2.rows_count) {
        return MatrixError.DimensionMismatch;
    }

    var result = try Matrix.init(allocator, m1.rows_count, m2.cols_count);

    var i: usize = 0;
    while (i < m1.rows_count) : (i += 1) {
        var j: usize = 0;
        while (j < m2.cols_count) : (j += 1) {
            var sum: f32 = 0.0;
            var k: usize = 0;
            while (k < m1.cols_count) : (k += 1) {
                sum += (try m1.get(i, k)) * (try m2.get(k, j));
            }
            try result.set(i, j, sum);
        }
    }

    return result;
}

/// Transpose a matrix
pub fn transpose(allocator: Allocator, m: Matrix) !Matrix {
    var result = try Matrix.init(allocator, m.cols_count, m.rows_count);

    var i: usize = 0;
    while (i < m.rows_count) : (i += 1) {
        var j: usize = 0;
        while (j < m.cols_count) : (j += 1) {
            try result.set(j, i, try m.get(i, j));
        }
    }

    return result;
}

/// Calculate matrix determinant (2x2 and 3x3 only)
pub fn determinant(m: Matrix) MatrixError!f32 {
    if (!m.isSquare()) return MatrixError.NotSquareMatrix;

    if (m.rows_count == 1) {
        return try m.get(0, 0);
    } else if (m.rows_count == 2) {
        const a = try m.get(0, 0);
        const b = try m.get(0, 1);
        const c = try m.get(1, 0);
        const d = try m.get(1, 1);
        return a * d - b * c;
    } else if (m.rows_count == 3) {
        const a = try m.get(0, 0);
        const b = try m.get(0, 1);
        const c = try m.get(0, 2);
        const d = try m.get(1, 0);
        const e = try m.get(1, 1);
        const f = try m.get(1, 2);
        const g = try m.get(2, 0);
        const h = try m.get(2, 1);
        const i = try m.get(2, 2);

        return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    }

    return MatrixError.InvalidInput;
}

/// Calculate trace (sum of diagonal elements)
pub fn trace(m: Matrix) MatrixError!f32 {
    if (!m.isSquare()) return MatrixError.NotSquareMatrix;

    var sum: f32 = 0.0;
    var i: usize = 0;
    while (i < m.rows_count) : (i += 1) {
        sum += try m.get(i, i);
    }
    return sum;
}

/// Calculate frobenius norm
pub fn frobeniusNorm(m: Matrix) f32 {
    var sum: f32 = 0.0;
    for (m.data) |val| {
        sum += val * val;
    }
    return @sqrt(sum);
}

/// Element-wise (Hadamard) product
pub fn hadamardProduct(allocator: Allocator, m1: Matrix, m2: Matrix) MatrixError!Matrix {
    if (m1.rows_count != m2.rows_count or m1.cols_count != m2.cols_count) {
        return MatrixError.DimensionMismatch;
    }

    var result = try Matrix.init(allocator, m1.rows_count, m1.cols_count);

    for (m1.data, m2.data, result.data) |a, b, *r| {
        r.* = a * b;
    }

    return result;
}

/// Matrix-vector multiplication
pub fn multiplyVector(allocator: Allocator, m: Matrix, v: Vector) !Vector {
    if (m.cols_count != v.dimension()) {
        return MatrixError.DimensionMismatch;
    }

    var result = try Vector.init(allocator, m.rows_count);

    var i: usize = 0;
    while (i < m.rows_count) : (i += 1) {
        var sum: f32 = 0.0;
        var j: usize = 0;
        while (j < m.cols_count) : (j += 1) {
            sum += (try m.get(i, j)) * (try v.get(j));
        }
        try result.set(i, sum);
    }

    return result;
}

/// Outer product of two vectors (creates a matrix)
pub fn outerProduct(allocator: Allocator, v1: Vector, v2: Vector) !Matrix {
    var result = try Matrix.init(allocator, v1.dimension(), v2.dimension());

    var i: usize = 0;
    while (i < v1.dimension()) : (i += 1) {
        var j: usize = 0;
        while (j < v2.dimension()) : (j += 1) {
            try result.set(i, j, (try v1.get(i)) * (try v2.get(j)));
        }
    }

    return result;
}

/// Check if two matrices are approximately equal
pub fn approxEqual(m1: Matrix, m2: Matrix, tolerance: f32) bool {
    if (m1.rows_count != m2.rows_count or m1.cols_count != m2.cols_count) return false;

    for (m1.data, m2.data) |a, b| {
        if (@abs(a - b) > tolerance) return false;
    }

    return true;
}
