#include "lcbinint/magnification/probe_diagnostics.hpp"

#include <cstdlib>
#include <cstring>

namespace lcbinint::magnification {
namespace {

ProbePolicy parse_policy(const char* spec)
{
    ProbePolicy policy;
    if (spec == nullptr) {
        return policy;
    }
    // Comma-separated key=value, unknown keys ignored so a harness can pass a
    // superset without the library having to know about it.
    std::string text(spec);
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t comma = text.find(',', start);
        const std::string item = text.substr(
            start, comma == std::string::npos ? std::string::npos : comma - start);
        const std::size_t equals = item.find('=');
        if (equals != std::string::npos) {
            const std::string key = item.substr(0, equals);
            const std::string value = item.substr(equals + 1);
            const bool flag = !(value == "0" || value == "false" || value == "off");
            if (key == "tangents") {
                policy.tangents = flag;
            } else if (key == "normals") {
                policy.normals = flag;
            } else if (key == "offsets") {
                policy.offsets = std::atoi(value.c_str());
            }
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return policy;
}

} // namespace

ProbeCounters& probe_counters()
{
    static thread_local ProbeCounters counters;
    return counters;
}

bool probe_stats_enabled()
{
    static const bool enabled = std::getenv("LCBININT_PROBE_STATS") != nullptr;
    return enabled;
}

void reset_probe_counters()
{
    probe_counters() = ProbeCounters {};
}

const ProbePolicy& probe_policy()
{
    static const ProbePolicy policy = parse_policy(std::getenv("LCBININT_PROBE_POLICY"));
    return policy;
}

std::string probe_policy_description()
{
    const auto& policy = probe_policy();
    std::string text;
    text += "normals=";
    text += policy.normals ? "1" : "0";
    text += ",tangents=";
    text += policy.tangents ? "1" : "0";
    text += ",offsets=";
    text += std::to_string(policy.offsets);
    return text;
}

} // namespace lcbinint::magnification
