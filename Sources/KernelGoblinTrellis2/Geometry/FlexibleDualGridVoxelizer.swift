import Foundation
import simd

// Behavior ported from microsoft/o-voxel 75fbf0183001ed9876c8dbb35de6b68552ee08bd,
// src/convert/flexible_dual_grid.cpp (SHA-256 95ebcdec3818539c52504cd4a89f4092...).

public struct FlexibleDualGridVoxelization: Equatable, Sendable {
    public let coordinates: [SIMD3<Int32>]
    /// Positions relative to `aabbMinimum`, matching O-Voxel's CPU wrapper.
    public let dualVertices: [SIMD3<Float>]
    public let intersections: [SIMD3<UInt8>]
    public let voxelSize: SIMD3<Float>

    public func dualOffsets() -> [SIMD3<Float>] {
        zip(dualVertices, coordinates).map { vertex, coordinate in
            vertex / voxelSize - SIMD3<Float>(
                Float(coordinate.x), Float(coordinate.y), Float(coordinate.z)
            )
        }
    }

    public func shapeEncoderInput(
        spatialShape: SparseSpatialShape
    ) throws -> ShapeEncoderMeshInput {
        guard coordinates.count == dualVertices.count,
              coordinates.count == intersections.count,
              !coordinates.isEmpty else {
            throw NativeRuntimeError.invalidArgument(
                "invalid flexible dual-grid shape encoder handoff"
            )
        }
        let sparseCoordinates = coordinates.map {
            SparseStructureCoordinate(x: $0.x, y: $0.y, z: $0.z)
        }
        _ = try SparseNeighborhood3x3(
            coordinates: sparseCoordinates, spatialShape: spatialShape
        )
        let offsets = dualOffsets()
        var values: [Float] = []
        values.reserveCapacity(coordinates.count * 6)
        for index in coordinates.indices {
            let offset = offsets[index] - SIMD3<Float>(repeating: 0.5)
            let flags = intersections[index]
            values.append(contentsOf: [
                offset.x, offset.y, offset.z,
                Float(flags.x) - 0.5,
                Float(flags.y) - 0.5,
                Float(flags.z) - 0.5,
            ])
        }
        return ShapeEncoderMeshInput(
            values: values, coordinates: sparseCoordinates,
            spatialShape: spatialShape
        )
    }
}

public struct ShapeEncoderMeshInput: Equatable, Sendable {
    public let values: [Float]
    public let coordinates: [SparseStructureCoordinate]
    public let spatialShape: SparseSpatialShape
}

public enum FlexibleDualGridVoxelizer {
    public static func voxelize(
        vertices: [SIMD3<Float>],
        faces: [SIMD3<UInt32>],
        gridSize: SIMD3<Int32>,
        aabbMinimum: SIMD3<Float> = SIMD3(repeating: -0.5),
        aabbMaximum: SIMD3<Float> = SIMD3(repeating: 0.5),
        faceWeight: Float = 1,
        boundaryWeight: Float = 0.2,
        regularizationWeight: Float = 1e-2
    ) throws -> FlexibleDualGridVoxelization {
        guard !vertices.isEmpty, !faces.isEmpty,
              gridSize.x > 0, gridSize.y > 0, gridSize.z > 0,
              allFinite(aabbMinimum), allFinite(aabbMaximum),
              aabbMaximum.x > aabbMinimum.x,
              aabbMaximum.y > aabbMinimum.y,
              aabbMaximum.z > aabbMinimum.z,
              faceWeight.isFinite, faceWeight >= 0,
              boundaryWeight.isFinite, boundaryWeight >= 0,
              regularizationWeight.isFinite, regularizationWeight > 0 else {
            throw NativeRuntimeError.invalidArgument(
                "invalid flexible dual-grid voxelization arguments"
            )
        }
        for vertex in vertices where !allFinite(vertex) {
            throw NativeRuntimeError.invalidArgument("voxelizer vertices must be finite")
        }
        var triangles: [Triangle] = []
        triangles.reserveCapacity(faces.count)
        for face in faces {
            guard face.x < vertices.count, face.y < vertices.count,
                  face.z < vertices.count else {
                throw NativeRuntimeError.invalidArgument("voxelizer face is out of range")
            }
            let triangle = Triangle(
                vertices[Int(face.x)] - aabbMinimum,
                vertices[Int(face.y)] - aabbMinimum,
                vertices[Int(face.z)] - aabbMinimum
            )
            guard triangle.normalLengthSquared > 0,
                  triangle.normalLengthSquared.isFinite,
                  allFinite(triangle.normal) else {
                throw NativeRuntimeError.invalidArgument(
                    "voxelizer faces must be nondegenerate"
                )
            }
            triangles.append(triangle)
        }

        let extent = aabbMaximum - aabbMinimum
        let voxelSize = extent / SIMD3<Float>(
            Float(gridSize.x), Float(gridSize.y), Float(gridSize.z)
        )
        guard voxelSize.x > 0, voxelSize.y > 0, voxelSize.z > 0,
              allFinite(voxelSize) else {
            throw NativeRuntimeError.invalidArgument("voxel size is not representable")
        }
        var state = State()
        intersectQEF(
            voxelSize: voxelSize, gridSize: gridSize,
            triangles: triangles, state: &state
        )
        if faceWeight > 0 {
            faceQEF(
                voxelSize: voxelSize, gridSize: gridSize,
                triangles: triangles, state: &state
            )
        }
        if boundaryWeight > 0 {
            boundaryQEF(
                voxelSize: voxelSize, gridSize: gridSize,
                vertices: vertices.map { $0 - aabbMinimum }, faces: faces,
                weight: boundaryWeight, state: &state
            )
        }

        var dualVertices: [SIMD3<Float>] = []
        dualVertices.reserveCapacity(state.coordinates.count)
        for index in state.coordinates.indices {
            var qef = state.qefs[index]
            let mean = state.means[index] / state.counts[index]
            qef.addRegularization(
                at: mean, weight: regularizationWeight * state.counts[index]
            )
            dualVertices.append(solveConstrained(
                qef, coordinate: state.coordinates[index], voxelSize: voxelSize
            ))
        }
        return FlexibleDualGridVoxelization(
            coordinates: state.coordinates,
            dualVertices: dualVertices,
            intersections: state.intersections,
            voxelSize: voxelSize
        )
    }
}

private struct Triangle {
    let v0: SIMD3<Float>
    let v1: SIMD3<Float>
    let v2: SIMD3<Float>
    let normal: SIMD3<Float>
    let normalLengthSquared: Float

    init(_ v0: SIMD3<Float>, _ v1: SIMD3<Float>, _ v2: SIMD3<Float>) {
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        let cross = simd_cross(v1 - v0, v2 - v1)
        normalLengthSquared = simd_dot(cross, cross)
        normal = cross / sqrt(normalLengthSquared)
    }

    var planeQEF: Matrix4 {
        let plane = SIMD4<Float>(normal, -simd_dot(normal, v0))
        return Matrix4.outer(plane)
    }
}

private struct State {
    var lookup: [SIMD3<Int32>: Int] = [:]
    var coordinates: [SIMD3<Int32>] = []
    var means: [SIMD3<Float>] = []
    var counts: [Float] = []
    var intersections: [SIMD3<UInt8>] = []
    var qefs: [Matrix4] = []

    mutating func accumulate(
        coordinate: SIMD3<Int32>, point: SIMD3<Float>, qef: Matrix4,
        intersectionAxis: Int?
    ) {
        if let index = lookup[coordinate] {
            means[index] += point
            counts[index] += 1
            qefs[index] += qef
            if let axis = intersectionAxis { intersections[index][axis] = 1 }
        } else {
            let index = coordinates.count
            lookup[coordinate] = index
            coordinates.append(coordinate)
            means.append(point)
            counts.append(1)
            var flags = SIMD3<UInt8>(repeating: 0)
            if let axis = intersectionAxis { flags[axis] = 1 }
            intersections.append(flags)
            qefs.append(qef)
        }
    }
}

private struct Matrix4: Equatable {
    var values = [Float](repeating: 0, count: 16)

    subscript(_ row: Int, _ column: Int) -> Float {
        get { values[row * 4 + column] }
        set { values[row * 4 + column] = newValue }
    }

    static func outer(_ vector: SIMD4<Float>) -> Matrix4 {
        var result = Matrix4()
        for row in 0..<4 {
            for column in 0..<4 { result[row, column] = vector[row] * vector[column] }
        }
        return result
    }

    static func += (left: inout Matrix4, right: Matrix4) {
        for index in 0..<16 { left.values[index] += right.values[index] }
    }

    mutating func addScaled(_ other: Matrix4, _ scale: Float) {
        for index in 0..<16 { values[index] += scale * other.values[index] }
    }

    mutating func addRegularization(at point: SIMD3<Float>, weight: Float) {
        var qef = Matrix4()
        qef[0, 0] = 1
        qef[1, 1] = 1
        qef[2, 2] = 1
        qef[0, 3] = -point.x
        qef[1, 3] = -point.y
        qef[2, 3] = -point.z
        qef[3, 0] = -point.x
        qef[3, 1] = -point.y
        qef[3, 2] = -point.z
        qef[3, 3] = simd_dot(point, point)
        addScaled(qef, weight)
    }

    func error(at point: SIMD3<Float>) -> Float {
        let p = SIMD4<Float>(point, 1)
        var result: Float = 0
        for row in 0..<4 {
            var rowValue: Float = 0
            for column in 0..<4 { rowValue += self[row, column] * p[column] }
            result += p[row] * rowValue
        }
        return result
    }
}

private func intersectQEF(
    voxelSize: SIMD3<Float>, gridSize: SIMD3<Int32>,
    triangles: [Triangle], state: inout State
) {
    for triangle in triangles {
        let qef = triangle.planeQEF
        for axis2 in 0..<3 {
            let axis0 = (axis2 + 1) % 3
            let axis1 = (axis2 + 2) % 3
            var projected = [triangle.v0, triangle.v1, triangle.v2].map {
                SIMD3<Double>(Double($0[axis0]), Double($0[axis1]), Double($0[axis2]))
            }
            projected.sort { $0.y < $1.y }
            let start = clampedVoxel(projected[0].y, voxelSize[axis1], gridSize[axis1])
            let middle = clampedVoxel(projected[1].y, voxelSize[axis1], gridSize[axis1])
            let end = clampedVoxel(projected[2].y, voxelSize[axis1], gridSize[axis1])
            scanHalf(
                rowStart: start, rowEnd: middle,
                projected[0], projected[1], projected[2],
                axis0: axis0, axis1: axis1, axis2: axis2,
                voxelSize: voxelSize, gridSize: gridSize, qef: qef, state: &state
            )
            scanHalf(
                rowStart: middle, rowEnd: end,
                projected[2], projected[1], projected[0],
                axis0: axis0, axis1: axis1, axis2: axis2,
                voxelSize: voxelSize, gridSize: gridSize, qef: qef, state: &state
            )
        }
    }
}

private func scanHalf(
    rowStart: Int32, rowEnd: Int32,
    _ t0: SIMD3<Double>, _ t1: SIMD3<Double>, _ t2: SIMD3<Double>,
    axis0: Int, axis1: Int, axis2: Int,
    voxelSize: SIMD3<Float>, gridSize: SIMD3<Int32>, qef: Matrix4,
    state: inout State
) {
    guard rowStart < rowEnd else { return }
    for yIndex in rowStart..<rowEnd {
        let y = Double(yIndex + 1) * Double(voxelSize[axis1])
        var t3 = lerp2(t0.y, t1.y, y, SIMD2(t0.x, t0.z), SIMD2(t1.x, t1.z))
        var t4 = lerp2(t0.y, t2.y, y, SIMD2(t0.x, t0.z), SIMD2(t2.x, t2.z))
        if t3.x > t4.x { swap(&t3, &t4) }
        let lineStart = clampedVoxel(t3.x, voxelSize[axis0], gridSize[axis0])
        let lineEnd = clampedVoxel(t4.x, voxelSize[axis0], gridSize[axis0])
        guard lineStart < lineEnd else { continue }
        for xIndex in lineStart..<lineEnd {
            let x = Double(xIndex + 1) * Double(voxelSize[axis0])
            let z = lerpScalar(t3.x, t4.x, x, t3.y, t4.y)
            let zIndex = Int32(z / Double(voxelSize[axis2]))
            guard zIndex >= 0, zIndex < gridSize[axis2] else { continue }
            var point = SIMD3<Float>(repeating: 0)
            point[axis0] = Float(x)
            point[axis1] = Float(y)
            point[axis2] = Float(z)
            for dx in Int32(0)..<2 {
                for dy in Int32(0)..<2 {
                    var coordinate = SIMD3<Int32>(repeating: 0)
                    coordinate[axis0] = xIndex + dx
                    coordinate[axis1] = yIndex + dy
                    coordinate[axis2] = zIndex
                    state.accumulate(
                        coordinate: coordinate, point: point, qef: qef,
                        intersectionAxis: dx == 0 && dy == 0 ? axis2 : nil
                    )
                }
            }
        }
    }
}

private func faceQEF(
    voxelSize: SIMD3<Float>, gridSize: SIMD3<Int32>,
    triangles: [Triangle], state: inout State
) {
    for triangle in triangles {
        let v0 = triangle.v0, v1 = triangle.v1, v2 = triangle.v2
        let e0 = v1 - v0, e1 = v2 - v1, e2 = v0 - v2
        let n = triangle.normal
        var minimum = SIMD3<Int32>(), maximum = SIMD3<Int32>()
        for axis in 0..<3 {
            let low = min(v0[axis], min(v1[axis], v2[axis])) / voxelSize[axis]
            let high = max(v0[axis], max(v1[axis], v2[axis])) / voxelSize[axis]
            minimum[axis] = max(Int32(low), 0)
            maximum[axis] = min(Int32(high + 1), gridSize[axis])
        }
        let c = SIMD3<Float>(
            n.x > 0 ? voxelSize.x : 0,
            n.y > 0 ? voxelSize.y : 0,
            n.z > 0 ? voxelSize.z : 0
        )
        let d1 = simd_dot(n, c - v0)
        let d2 = simd_dot(n, voxelSize - c - v0)
        let xy = projectionEdges(
            multiplier: n.z < 0 ? -1 : 1,
            edges: [SIMD2(e0.x, e0.y), SIMD2(e1.x, e1.y), SIMD2(e2.x, e2.y)],
            origins: [SIMD2(v0.x, v0.y), SIMD2(v1.x, v1.y), SIMD2(v2.x, v2.y)],
            voxel: SIMD2(voxelSize.x, voxelSize.y)
        )
        let yz = projectionEdges(
            multiplier: n.x < 0 ? -1 : 1,
            edges: [SIMD2(e0.y, e0.z), SIMD2(e1.y, e1.z), SIMD2(e2.y, e2.z)],
            origins: [SIMD2(v0.y, v0.z), SIMD2(v1.y, v1.z), SIMD2(v2.y, v2.z)],
            voxel: SIMD2(voxelSize.y, voxelSize.z)
        )
        let zx = projectionEdges(
            multiplier: n.y < 0 ? -1 : 1,
            edges: [SIMD2(e0.z, e0.x), SIMD2(e1.z, e1.x), SIMD2(e2.z, e2.x)],
            origins: [SIMD2(v0.z, v0.x), SIMD2(v1.z, v1.x), SIMD2(v2.z, v2.x)],
            voxel: SIMD2(voxelSize.z, voxelSize.x)
        )
        guard minimum.x < maximum.x, minimum.y < maximum.y,
              minimum.z < maximum.z else { continue }
        for z in minimum.z..<maximum.z {
            for y in minimum.y..<maximum.y {
                for x in minimum.x..<maximum.x {
                    let p = voxelSize * SIMD3<Float>(Float(x), Float(y), Float(z))
                    let plane = simd_dot(n, p)
                    if (plane + d1) * (plane + d2) > 0 { continue }
                    if !projectionContains(xy, SIMD2(p.x, p.y)) { continue }
                    if !projectionContains(yz, SIMD2(p.y, p.z)) { continue }
                    if !projectionContains(zx, SIMD2(p.z, p.x)) { continue }
                    if let index = state.lookup[SIMD3(x, y, z)] {
                        // Pinned O-Voxel uses faceWeight as an enable flag and adds one Q.
                        state.qefs[index] += triangle.planeQEF
                    }
                }
            }
        }
    }
}

private struct ProjectionEdge {
    let normal: SIMD2<Float>
    let distance: Float
}

private func projectionEdges(
    multiplier: Float, edges: [SIMD2<Float>], origins: [SIMD2<Float>],
    voxel: SIMD2<Float>
) -> [ProjectionEdge] {
    zip(edges, origins).map { edge, origin in
        let normal = SIMD2(-multiplier * edge.y, multiplier * edge.x)
        let positive = SIMD2(max(normal.x, 0), max(normal.y, 0))
        return ProjectionEdge(
            normal: normal,
            distance: -simd_dot(normal, origin) + simd_dot(positive, voxel)
        )
    }
}

private func projectionContains(_ edges: [ProjectionEdge], _ point: SIMD2<Float>) -> Bool {
    edges.allSatisfy { simd_dot($0.normal, point) + $0.distance >= 0 }
}

private struct EdgeKey: Hashable, Comparable {
    let lower: UInt32
    let upper: UInt32
    static func < (left: EdgeKey, right: EdgeKey) -> Bool {
        left.lower == right.lower ? left.upper < right.upper : left.lower < right.lower
    }
}

private func boundaryQEF(
    voxelSize: SIMD3<Float>, gridSize: SIMD3<Int32>, vertices: [SIMD3<Float>],
    faces: [SIMD3<UInt32>], weight: Float, state: inout State
) {
    var counts: [EdgeKey: Int] = [:]
    for face in faces {
        let indices = [face.x, face.y, face.z]
        for index in 0..<3 {
            let a = indices[index], b = indices[(index + 1) % 3]
            let key = EdgeKey(lower: min(a, b), upper: max(a, b))
            counts[key, default: 0] += 1
        }
    }
    for key in counts.keys.filter({ counts[$0] == 1 }).sorted() {
        let v0 = vertices[Int(key.lower)], v1 = vertices[Int(key.upper)]
        let delta = SIMD3<Double>(
            Double(v1.x - v0.x), Double(v1.y - v0.y), Double(v1.z - v0.z)
        )
        let length = sqrt(simd_dot(delta, delta))
        if length < 1e-6 { continue }
        let direction = delta / length
        var a = [Float](repeating: 0, count: 9)
        for row in 0..<3 {
            for column in 0..<3 {
                a[row * 3 + column] = (row == column ? 1 : 0)
                    - Float(direction[row] * direction[column])
            }
        }
        var qef = Matrix4()
        for row in 0..<3 {
            for column in 0..<3 { qef[row, column] = a[row * 3 + column] }
            var b: Float = 0
            for column in 0..<3 { b -= a[row * 3 + column] * v0[column] }
            qef[row, 3] = b
            qef[3, row] = b
        }
        var constant: Float = 0
        for row in 0..<3 {
            for column in 0..<3 { constant += v0[row] * a[row * 3 + column] * v0[column] }
        }
        qef[3, 3] = constant

        var current = SIMD3<Int32>()
        var end = SIMD3<Int32>()
        var step = SIMD3<Int32>()
        var tMax = SIMD3<Double>()
        var tDelta = SIMD3<Double>()
        for axis in 0..<3 {
            current[axis] = Int32(floor(Double(v0[axis] / voxelSize[axis])))
            end[axis] = Int32(floor(Double(v1[axis] / voxelSize[axis])))
            step[axis] = direction[axis] > 0 ? 1 : -1
            if direction[axis] == 0 {
                tMax[axis] = .infinity
                tDelta[axis] = .infinity
            } else {
                let border = voxelSize[axis]
                    * Float(current[axis] + (step[axis] > 0 ? 1 : 0))
                tMax[axis] = Double(border - v0[axis]) / direction[axis]
                tDelta[axis] = Double(voxelSize[axis]) / abs(direction[axis])
            }
        }
        var traversed = [current]
        while current != end {
            let axis: Int
            if tMax.x < tMax.y { axis = tMax.x < tMax.z ? 0 : 2 }
            else { axis = tMax.y < tMax.z ? 1 : 2 }
            if tMax[axis] > length { break }
            current[axis] += step[axis]
            tMax[axis] += tDelta[axis]
            traversed.append(current)
        }
        for coordinate in traversed where inGrid(coordinate, gridSize) {
            if let index = state.lookup[coordinate] {
                state.qefs[index].addScaled(qef, weight)
            }
        }
    }
}

private func solveConstrained(
    _ qef: Matrix4, coordinate: SIMD3<Int32>, voxelSize: SIMD3<Float>
) -> SIMD3<Float> {
    let minimum = voxelSize * SIMD3<Float>(
        Float(coordinate.x), Float(coordinate.y), Float(coordinate.z)
    )
    let maximum = minimum + voxelSize
    var result = solve3(qef)
    if inside(result, minimum, maximum) { return result }
    var best = Float.infinity
    func consider(_ point: SIMD3<Float>) {
        guard allFinite(point) else { return }
        let error = qef.error(at: point)
        if error < best { best = error; result = point }
    }
    for fixedAxis in 0..<3 {
        let axis1 = (fixedAxis + 1) % 3, axis2 = (fixedAxis + 2) % 3
        for fixed in [minimum[fixedAxis], maximum[fixedAxis]] {
            let rhs = SIMD2<Float>(
                -(qef[axis1, fixedAxis] * fixed + qef[axis1, 3]),
                -(qef[axis2, fixedAxis] * fixed + qef[axis2, 3])
            )
            let solution = solve2(
                qef[axis1, axis1], qef[axis1, axis2],
                qef[axis2, axis1], qef[axis2, axis2], rhs
            )
            if solution.x >= minimum[axis1], solution.x <= maximum[axis1],
               solution.y >= minimum[axis2], solution.y <= maximum[axis2] {
                var point = SIMD3<Float>(); point[fixedAxis] = fixed
                point[axis1] = solution.x; point[axis2] = solution.y
                consider(point)
            }
        }
    }
    for freeAxis in 0..<3 {
        let axis1 = (freeAxis + 1) % 3, axis2 = (freeAxis + 2) % 3
        for fixed1 in [minimum[axis1], maximum[axis1]] {
            for fixed2 in [minimum[axis2], maximum[axis2]] {
                let value = -(qef[freeAxis, axis1] * fixed1
                    + qef[freeAxis, axis2] * fixed2 + qef[freeAxis, 3])
                    / qef[freeAxis, freeAxis]
                if value >= minimum[freeAxis], value <= maximum[freeAxis] {
                    var point = SIMD3<Float>(); point[freeAxis] = value
                    point[axis1] = fixed1; point[axis2] = fixed2
                    consider(point)
                }
            }
        }
    }
    for xConstraint in 0..<2 {
        for yConstraint in 0..<2 {
            for zConstraint in 0..<2 {
                consider(SIMD3(
                    xConstraint != 0 ? minimum.x : maximum.x,
                    yConstraint != 0 ? minimum.y : maximum.y,
                    zConstraint != 0 ? minimum.z : maximum.z
                ))
            }
        }
    }
    return result
}

private func solve3(_ qef: Matrix4) -> SIMD3<Float> {
    var rows = [
        [qef[0, 0], qef[0, 1], qef[0, 2], -qef[0, 3]],
        [qef[1, 0], qef[1, 1], qef[1, 2], -qef[1, 3]],
        [qef[2, 0], qef[2, 1], qef[2, 2], -qef[2, 3]],
    ]
    for pivot in 0..<3 {
        var selected = pivot
        for row in (pivot + 1)..<3 where abs(rows[row][pivot]) > abs(rows[selected][pivot]) {
            selected = row
        }
        if selected != pivot { rows.swapAt(selected, pivot) }
        let divisor = rows[pivot][pivot]
        if divisor == 0 { return SIMD3(repeating: .nan) }
        for column in pivot..<4 { rows[pivot][column] /= divisor }
        for row in 0..<3 where row != pivot {
            let factor = rows[row][pivot]
            for column in pivot..<4 { rows[row][column] -= factor * rows[pivot][column] }
        }
    }
    return SIMD3(rows[0][3], rows[1][3], rows[2][3])
}

private func solve2(
    _ a00: Float, _ a01: Float, _ a10: Float, _ a11: Float, _ b: SIMD2<Float>
) -> SIMD2<Float> {
    let determinant = a00 * a11 - a01 * a10
    guard determinant != 0 else { return SIMD2(repeating: .nan) }
    return SIMD2(
        (b.x * a11 - a01 * b.y) / determinant,
        (a00 * b.y - b.x * a10) / determinant
    )
}

private func lerp2(
    _ a: Double, _ b: Double, _ t: Double,
    _ valueA: SIMD2<Double>, _ valueB: SIMD2<Double>
) -> SIMD2<Double> {
    if a == b { return valueA }
    let alpha = (t - a) / (b - a)
    return (1 - alpha) * valueA + alpha * valueB
}

private func lerpScalar(
    _ a: Double, _ b: Double, _ t: Double, _ valueA: Double, _ valueB: Double
) -> Double {
    if a == b { return valueA }
    let alpha = (t - a) / (b - a)
    return (1 - alpha) * valueA + alpha * valueB
}

private func clampedVoxel(_ value: Double, _ size: Float, _ maximum: Int32) -> Int32 {
    min(max(Int32(value / Double(size)), 0), maximum - 1)
}

private func inGrid(_ coordinate: SIMD3<Int32>, _ grid: SIMD3<Int32>) -> Bool {
    coordinate.x >= 0 && coordinate.x < grid.x
        && coordinate.y >= 0 && coordinate.y < grid.y
        && coordinate.z >= 0 && coordinate.z < grid.z
}

private func inside(
    _ point: SIMD3<Float>, _ minimum: SIMD3<Float>, _ maximum: SIMD3<Float>
) -> Bool {
    point.x >= minimum.x && point.x <= maximum.x
        && point.y >= minimum.y && point.y <= maximum.y
        && point.z >= minimum.z && point.z <= maximum.z
}

private func allFinite(_ value: SIMD3<Float>) -> Bool {
    value.x.isFinite && value.y.isFinite && value.z.isFinite
}
