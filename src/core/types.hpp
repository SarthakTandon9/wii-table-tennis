#pragma once

namespace pp {

// Scalar type for the whole simulation. float gives 4 lanes per NEON register
// (double gives 2), which matters for the batch planner in M6. Table-tennis
// scales (metres, tens of m/s, 1 ms steps) sit comfortably in float precision.
// Switch it here if you want to measure the difference.
using real = float;

}  // namespace pp
