import Foundation

public enum TeleaInpaint {
    public static let openCVRevision = "fe38fc608f6acb8b68953438a62305d8318f4fcd"
    public static let sourceSHA256 = "15c66e1f742fd8a08590996e80e06895c5355126bd9457ab613be03179019734"

    public static func fill(
        _ source: [UInt8], mask: [UInt8], width: Int, height: Int,
        channels: Int, radius requestedRadius: Int
    ) throws -> [UInt8] {
        let pixels = width.multipliedReportingOverflow(by: height)
        let values = pixels.partialValue.multipliedReportingOverflow(by: channels)
        guard width > 1, height > 1, [1, 3].contains(channels),
              !pixels.overflow, !values.overflow,
              source.count == values.partialValue,
              mask.count == pixels.partialValue else {
            throw NativeRuntimeError.invalidArgument("invalid Telea image shape")
        }
        let radius = min(max(requestedRadius, 1), 100)
        let paddedRows = height.addingReportingOverflow(2)
        let paddedColumns = width.addingReportingOverflow(2)
        guard !paddedRows.overflow, !paddedColumns.overflow else {
            throw NativeRuntimeError.invalidArgument("Telea padded image dimensions overflow Int")
        }
        let rows = paddedRows.partialValue
        let columns = paddedColumns.partialValue
        let paddedCount = rows.multipliedReportingOverflow(by: columns)
        guard !paddedCount.overflow else {
            throw NativeRuntimeError.invalidArgument("Telea padded image size overflows Int")
        }
        var status = [UInt8](repeating: teleaKnown, count: paddedCount.partialValue)
        for row in 0..<height {
            for column in 0..<width where mask[row * width + column] != 0 {
                status[(row + 1) * columns + column + 1] = teleaInside
            }
        }
        var band = [UInt8](repeating: 0, count: paddedCount.partialValue)
        for row in 1...height {
            for column in 1...width where status[row * columns + column] == teleaKnown {
                if status[(row - 1) * columns + column] == teleaInside
                    || status[row * columns + column - 1] == teleaInside
                    || status[(row + 1) * columns + column] == teleaInside
                    || status[row * columns + column + 1] == teleaInside {
                    band[row * columns + column] = 1
                }
            }
        }
        var heap = TeleaHeap()
        for row in 0..<rows {
            for column in 0..<columns where band[row * columns + column] != 0 {
                heap.push(row: row, column: column, distance: 0)
            }
        }
        var distance = [Float](repeating: 1.0e6, count: paddedCount.partialValue)
        for index in band.indices where band[index] != 0 { distance[index] = 0 }

        var signedStatus = [UInt8](repeating: teleaKnown, count: paddedCount.partialValue)
        for row in 1...height {
            for column in 1...width {
                let index = row * columns + column
                guard status[index] != teleaInside, band[index] == 0 else { continue }
                let rowStart = max(1, row - radius)
                let rowEnd = min(height, row + radius)
                let columnStart = max(1, column - radius)
                let columnEnd = min(width, column + radius)
                var nearUnknown = false
                for candidateRow in rowStart...rowEnd {
                    for candidateColumn in columnStart...columnEnd
                    where status[candidateRow * columns + candidateColumn] == teleaInside {
                        nearUnknown = true
                        break
                    }
                    if nearUnknown { break }
                }
                if nearUnknown { signedStatus[index] = teleaInside }
            }
        }
        var outsideHeap = TeleaHeap()
        for row in 0..<rows {
            for column in 0..<columns where band[row * columns + column] != 0 {
                outsideHeap.push(row: row, column: column, distance: 0)
            }
        }
        teleaFastMarch(
            status: &signedStatus, distance: &distance, heap: &outsideHeap,
            rows: rows, columns: columns, negate: true
        )
        var output = source
        teleaFill(
            status: &status, distance: &distance, output: &output,
            width: width, height: height, channels: channels,
            radius: radius, heap: &heap
        )
        return output
    }
}

private let teleaKnown: UInt8 = 0
private let teleaBand: UInt8 = 1
private let teleaInside: UInt8 = 2
private let teleaChange: UInt8 = 3

private struct TeleaNode {
    let distance: Float
    let row: Int
    let column: Int
    let order: Int
}

private struct TeleaHeap {
    private var nodes: [TeleaNode] = []
    private var nextOrder = 0

    mutating func push(row: Int, column: Int, distance: Float) {
        let node = TeleaNode(
            distance: distance, row: row, column: column, order: nextOrder
        )
        nextOrder += 1
        nodes.append(node)
        var index = nodes.count - 1
        while index > 0 {
            let parent = (index - 1) / 2
            guard precedes(nodes[index], nodes[parent]) else { break }
            nodes.swapAt(index, parent)
            index = parent
        }
    }

    mutating func pop() -> TeleaNode? {
        guard !nodes.isEmpty else { return nil }
        if nodes.count == 1 { return nodes.removeLast() }
        let result = nodes[0]
        nodes[0] = nodes.removeLast()
        var index = 0
        while true {
            let left = index * 2 + 1
            guard left < nodes.count else { break }
            let right = left + 1
            var child = left
            if right < nodes.count, precedes(nodes[right], nodes[left]) { child = right }
            guard precedes(nodes[child], nodes[index]) else { break }
            nodes.swapAt(child, index)
            index = child
        }
        return result
    }

    private func precedes(_ lhs: TeleaNode, _ rhs: TeleaNode) -> Bool {
        lhs.distance < rhs.distance
            || (lhs.distance == rhs.distance && lhs.order < rhs.order)
    }
}

private func teleaSolve(
    _ row1: Int, _ column1: Int, _ row2: Int, _ column2: Int,
    status: [UInt8], distance: [Float], columns: Int
) -> Float {
    let firstIndex = row1 * columns + column1
    let secondIndex = row2 * columns + column2
    let first = Double(distance[firstIndex])
    let second = Double(distance[secondIndex])
    let minimum = min(first, second)
    let solution: Double
    if status[firstIndex] != teleaInside {
        if status[secondIndex] != teleaInside {
            if abs(first - second) >= 1 {
                solution = 1 + minimum
            } else {
                let delta = first - second
                solution = (first + second + sqrt(2 - delta * delta)) * 0.5
            }
        } else {
            solution = 1 + first
        }
    } else if status[secondIndex] != teleaInside {
        solution = 1 + second
    } else {
        solution = 1 + minimum
    }
    return Float(solution)
}

private func teleaDistance(
    row: Int, column: Int, status: [UInt8], distance: [Float], columns: Int
) -> Float {
    let first = teleaSolve(
        row - 1, column, row, column - 1,
        status: status, distance: distance, columns: columns
    )
    let second = teleaSolve(
        row + 1, column, row, column - 1,
        status: status, distance: distance, columns: columns
    )
    let third = teleaSolve(
        row - 1, column, row, column + 1,
        status: status, distance: distance, columns: columns
    )
    let fourth = teleaSolve(
        row + 1, column, row, column + 1,
        status: status, distance: distance, columns: columns
    )
    return min(min(first, second), min(third, fourth))
}

private func teleaFastMarch(
    status: inout [UInt8], distance: inout [Float], heap: inout TeleaHeap,
    rows: Int, columns: Int, negate: Bool
) {
    let offsets = [(-1, 0), (0, -1), (1, 0), (0, 1)]
    while let node = heap.pop() {
        status[node.row * columns + node.column] = negate ? teleaChange : teleaKnown
        for offset in offsets {
            let row = node.row + offset.0
            let column = node.column + offset.1
            guard row > 0, column > 0, row < rows - 1, column < columns - 1 else {
                continue
            }
            let index = row * columns + column
            guard status[index] == teleaInside else { continue }
            let value = teleaDistance(
                row: row, column: column, status: status,
                distance: distance, columns: columns
            )
            distance[index] = value
            status[index] = teleaBand
            heap.push(row: row, column: column, distance: value)
        }
    }
    if negate {
        for index in status.indices where status[index] == teleaChange {
            status[index] = teleaKnown
            distance[index] = -distance[index]
        }
    }
}

private func teleaFill(
    status: inout [UInt8], distance: inout [Float], output: inout [UInt8],
    width: Int, height: Int, channels: Int, radius: Int, heap: inout TeleaHeap
) {
    let rows = height + 2
    let columns = width + 2
    let offsets = [(-1, 0), (0, -1), (1, 0), (0, 1)]
    while let node = heap.pop() {
        status[node.row * columns + node.column] = teleaKnown
        for offset in offsets {
            let row = node.row + offset.0
            let column = node.column + offset.1
            guard row > 0, column > 0, row < rows - 1, column < columns - 1 else {
                continue
            }
            let index = row * columns + column
            guard status[index] == teleaInside else { continue }
            let value = teleaDistance(
                row: row, column: column, status: status,
                distance: distance, columns: columns
            )
            distance[index] = value
            var gradientX: Float
            if status[row * columns + column + 1] != teleaInside {
                if status[row * columns + column - 1] != teleaInside {
                    gradientX = (distance[row * columns + column + 1]
                        - distance[row * columns + column - 1]) * 0.5
                } else {
                    gradientX = distance[row * columns + column + 1] - distance[index]
                }
            } else if status[row * columns + column - 1] != teleaInside {
                gradientX = distance[index] - distance[row * columns + column - 1]
            } else {
                gradientX = 0
            }
            var gradientY: Float
            if status[(row + 1) * columns + column] != teleaInside {
                if status[(row - 1) * columns + column] != teleaInside {
                    gradientY = (distance[(row + 1) * columns + column]
                        - distance[(row - 1) * columns + column]) * 0.5
                } else {
                    gradientY = distance[(row + 1) * columns + column] - distance[index]
                }
            } else if status[(row - 1) * columns + column] != teleaInside {
                gradientY = distance[index] - distance[(row - 1) * columns + column]
            } else {
                gradientY = 0
            }
            for channel in 0..<channels {
                var intensity: Float = 0
                var correctionX: Float = 0
                var correctionY: Float = 0
                var weightSum: Float = 1.0e-20
                for sampleRow in (row - radius)...(row + radius) {
                    let rowMinus = sampleRow - 1 + (sampleRow == 1 ? 1 : 0)
                    let rowPlus = sampleRow - 1 - (sampleRow == rows - 2 ? 1 : 0)
                    for sampleColumn in (column - radius)...(column + radius) {
                        let columnMinus = sampleColumn - 1 + (sampleColumn == 1 ? 1 : 0)
                        let columnPlus = sampleColumn - 1
                            - (sampleColumn == columns - 2 ? 1 : 0)
                        guard sampleRow > 0, sampleColumn > 0,
                              sampleRow < rows - 1, sampleColumn < columns - 1,
                              status[sampleRow * columns + sampleColumn] != teleaInside else {
                            continue
                        }
                        let deltaRow = sampleRow - row
                        let deltaColumn = sampleColumn - column
                        guard deltaColumn * deltaColumn + deltaRow * deltaRow
                                <= radius * radius else { continue }
                        let vectorX = Float(column - sampleColumn)
                        let vectorY = Float(row - sampleRow)
                        let lengthSquared = vectorX * vectorX + vectorY * vectorY
                        let destination: Float
                        if channels == 3 {
                            destination = Float(1.0 / (
                                Double(lengthSquared) * sqrt(Double(lengthSquared))
                            ))
                        } else {
                            destination = 1 / (lengthSquared * sqrt(lengthSquared))
                        }
                        let level = 1 / (1 + abs(
                            distance[sampleRow * columns + sampleColumn] - distance[index]
                        ))
                        var direction = vectorX * gradientX + vectorY * gradientY
                        if abs(direction) <= 0.01 { direction = 0.000001 }
                        let weight = abs(destination * level * direction)
                        let sampleIndex = { (r: Int, c: Int) -> Int in
                            (r * width + c) * channels + channel
                        }
                        let gradientImageX: Float
                        if status[sampleRow * columns + sampleColumn + 1] != teleaInside {
                            if status[sampleRow * columns + sampleColumn - 1] != teleaInside {
                                gradientImageX = Float(
                                    Int(output[sampleIndex(rowMinus, columnPlus + 1)])
                                        - Int(output[sampleIndex(rowMinus, columnMinus - 1)])
                                ) * 2
                            } else {
                                gradientImageX = Float(
                                    Int(output[sampleIndex(rowMinus, columnPlus + 1)])
                                        - Int(output[sampleIndex(rowMinus, columnMinus)])
                                )
                            }
                        } else if status[sampleRow * columns + sampleColumn - 1] != teleaInside {
                            gradientImageX = Float(
                                Int(output[sampleIndex(rowMinus, columnPlus)])
                                    - Int(output[sampleIndex(rowMinus, columnMinus - 1)])
                            )
                        } else {
                            gradientImageX = 0
                        }
                        let gradientImageY: Float
                        if status[(sampleRow + 1) * columns + sampleColumn] != teleaInside {
                            if status[(sampleRow - 1) * columns + sampleColumn] != teleaInside {
                                gradientImageY = Float(
                                    Int(output[sampleIndex(rowPlus + 1, columnMinus)])
                                        - Int(output[sampleIndex(rowMinus - 1, columnMinus)])
                                ) * 2
                            } else {
                                gradientImageY = Float(
                                    Int(output[sampleIndex(rowPlus + 1, columnMinus)])
                                        - Int(output[sampleIndex(rowMinus, columnMinus)])
                                )
                            }
                        } else if status[(sampleRow - 1) * columns + sampleColumn] != teleaInside {
                            gradientImageY = Float(
                                Int(output[sampleIndex(rowPlus, columnMinus)])
                                    - Int(output[sampleIndex(rowMinus - 1, columnMinus)])
                            )
                        } else {
                            gradientImageY = 0
                        }
                        intensity += weight * Float(output[
                            ((sampleRow - 1) * width + sampleColumn - 1) * channels + channel
                        ])
                        correctionX -= weight * gradientImageX * vectorX
                        correctionY -= weight * gradientImageY * vectorY
                        weightSum += weight
                    }
                }
                let correction = (correctionX + correctionY)
                    / (sqrt(correctionX * correctionX + correctionY * correctionY) + 1.0e-20)
                let result = intensity / weightSum + correction
                output[((row - 1) * width + column - 1) * channels + channel]
                    = teleaRoundUInt8(result)
            }
            status[index] = teleaBand
            heap.push(row: row, column: column, distance: value)
        }
    }
}

private func teleaRoundUInt8(_ value: Float) -> UInt8 {
    let rounded = (value + 0.5).rounded(.toNearestOrEven)
    guard rounded.isFinite else { return 0 }
    return UInt8(clamping: Int(rounded))
}
