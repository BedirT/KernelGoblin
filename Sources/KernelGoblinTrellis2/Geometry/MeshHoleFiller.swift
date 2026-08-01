import simd

public struct MeshHoleFillResult: Equatable, Sendable {
    public let mesh: FlexibleDualGridMesh
    public let addedFaceCount: Int
}

public enum MeshHoleFiller {
    private struct Edge: Hashable, Comparable {
        let low: UInt32
        let high: UInt32

        init(_ a: UInt32, _ b: UInt32) {
            low = min(a, b)
            high = max(a, b)
        }

        static func < (lhs: Edge, rhs: Edge) -> Bool {
            lhs.low == rhs.low ? lhs.high < rhs.high : lhs.low < rhs.low
        }
    }

    public static func fillTriangleAndQuadHoles(
        _ mesh: FlexibleDualGridMesh
    ) throws -> MeshHoleFillResult {
        guard mesh.faces.count >= 3 else {
            return MeshHoleFillResult(mesh: mesh, addedFaceCount: 0)
        }
        var counts: [Edge: Int] = [:]
        var directed: [Edge: (UInt32, UInt32)] = [:]
        for face in mesh.faces {
            let edges = [(face.x, face.y), (face.y, face.z), (face.z, face.x)]
            for (a, b) in edges {
                guard a != b, Int(a) < mesh.vertices.count, Int(b) < mesh.vertices.count else {
                    throw NativeRuntimeError.invalidArgument("invalid mesh face for hole filling")
                }
                let edge = Edge(a, b)
                counts[edge, default: 0] += 1
                directed[edge] = (a, b)
            }
        }
        let boundary = counts.filter { $0.value == 1 }.map(\.key).sorted()
        guard boundary.count >= 3 else {
            return MeshHoleFillResult(mesh: mesh, addedFaceCount: 0)
        }
        var adjacency: [UInt32: [UInt32]] = [:]
        for edge in boundary {
            adjacency[edge.low, default: []].append(edge.high)
            adjacency[edge.high, default: []].append(edge.low)
        }
        for vertex in adjacency.keys {
            adjacency[vertex]!.sort()
        }
        var unvisited = Set(boundary)
        var added: [SIMD3<UInt32>] = []
        while let firstEdge = unvisited.min() {
            var loop = [firstEdge.low]
            var previous = firstEdge.low
            var current = firstEdge.high
            unvisited.remove(firstEdge)
            var closed = false
            while loop.count <= boundary.count {
                loop.append(current)
                guard let neighbors = adjacency[current], neighbors.count == 2 else { break }
                let next = neighbors[0] == previous ? neighbors[1] : neighbors[0]
                if next == loop[0] {
                    closed = true
                    break
                }
                let nextEdge = Edge(current, next)
                guard unvisited.remove(nextEdge) != nil else { break }
                previous = current
                current = next
            }
            guard closed, loop.count == 3 || loop.count == 4 else { continue }
            var candidates: [SIMD3<UInt32>] = [SIMD3(loop[0], loop[1], loop[2])]
            if loop.count == 4 {
                candidates.append(SIMD3(loop[2], loop[3], loop[0]))
            }
            for candidate in candidates {
                let edge = Edge(candidate.x, candidate.y)
                guard let source = directed[edge] else { continue }
                let oriented = source.0 == candidate.y && source.1 == candidate.x
                    ? candidate : SIMD3(candidate.z, candidate.y, candidate.x)
                let a = mesh.vertices[Int(oriented.x)]
                let b = mesh.vertices[Int(oriented.y)]
                let c = mesh.vertices[Int(oriented.z)]
                guard simd_length_squared(simd_cross(b - a, c - a)) > 0 else { continue }
                added.append(oriented)
            }
        }
        guard !added.isEmpty else {
            return MeshHoleFillResult(mesh: mesh, addedFaceCount: 0)
        }
        return MeshHoleFillResult(
            mesh: FlexibleDualGridMesh(
                vertices: mesh.vertices, faces: mesh.faces + added
            ),
            addedFaceCount: added.count
        )
    }
}
