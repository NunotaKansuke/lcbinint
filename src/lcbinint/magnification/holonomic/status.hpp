#pragma once

// ATPT holonomic solver (M7) -- status enum (implementation plan sec. 12.2).
// Fail closed: a solve that cannot certify its result returns a non-OK
// status, never a silent approximation.

#include <string>

namespace lcbinint::holonomic {

enum class Status {
    OK,
    OK_VALIDATED,
    OK_ESTIMATED,
    LOCAL_REFERENCE_USED,
    TOPOLOGY_UNCERTAIN,
    CONNECTION_ILL_CONDITIONED,
    TRANSPORT_TOLERANCE_FAILED,
    BASIS_DEGENERATE,
    GRADIENT_UNRELIABLE,
    ARC_TOPOLOGY_INVALID,
    EVENT_UNRESOLVED,
    RESOURCE_LIMIT,
};

inline const char* to_string(Status s) {
    switch (s) {
        case Status::OK: return "OK";
        case Status::OK_VALIDATED: return "OK_VALIDATED";
        case Status::OK_ESTIMATED: return "OK_ESTIMATED";
        case Status::LOCAL_REFERENCE_USED: return "LOCAL_REFERENCE_USED";
        case Status::TOPOLOGY_UNCERTAIN: return "TOPOLOGY_UNCERTAIN";
        case Status::CONNECTION_ILL_CONDITIONED: return "CONNECTION_ILL_CONDITIONED";
        case Status::TRANSPORT_TOLERANCE_FAILED: return "TRANSPORT_TOLERANCE_FAILED";
        case Status::BASIS_DEGENERATE: return "BASIS_DEGENERATE";
        case Status::GRADIENT_UNRELIABLE: return "GRADIENT_UNRELIABLE";
        case Status::ARC_TOPOLOGY_INVALID: return "ARC_TOPOLOGY_INVALID";
        case Status::EVENT_UNRESOLVED: return "EVENT_UNRESOLVED";
        case Status::RESOURCE_LIMIT: return "RESOURCE_LIMIT";
    }
    return "OK";
}

}  // namespace lcbinint::holonomic
