#pragma once

#include <array>
#include <vector>

namespace lcbinint::obs {

// Sky position of a microlensing event, stored internally in degrees.
// Used for annual parallax (sets lcbi_params::ra, ::dec).
class SkyCoord {
public:
    SkyCoord(double ra_deg, double dec_deg)
        : ra_deg_(ra_deg), dec_deg_(dec_deg) {}

    double ra_deg()  const noexcept { return ra_deg_; }
    double dec_deg() const noexcept { return dec_deg_; }

private:
    double ra_deg_;
    double dec_deg_;
};

enum class SiteKind { ground, space };

// Observatory/telescope descriptor. Ground sites optionally carry geodetic
// coordinates; space sites optionally carry a VBM-style geocentric ephemeris
// table (JD, RA_deg, Dec_deg, distance_AU) converted to ICRF Cartesian AU.
class Site {
public:
    Site() = default;
    Site(double lat_deg, double lon_deg)
        : kind_(SiteKind::ground), has_ground_position_(true),
          lat_deg_(lat_deg), lon_deg_(lon_deg) {}

    explicit Site(std::vector<double> times, std::vector<std::array<double, 3>> positions)
        : kind_(SiteKind::space), times_(std::move(times)), positions_(std::move(positions)) {}

    SiteKind kind() const noexcept { return kind_; }
    bool has_ground_position() const noexcept { return has_ground_position_; }
    bool has_space_ephemeris() const noexcept { return positions_.size() >= 2; }
    double lat_deg() const noexcept { return lat_deg_; }
    double lon_deg() const noexcept { return lon_deg_; }
    const std::vector<double>& times() const noexcept { return times_; }
    const std::vector<std::array<double, 3>>& positions() const noexcept {
        return positions_;
    }
    bool space_position(double time, std::array<double, 3>& position) const noexcept;

private:
    SiteKind kind_ = SiteKind::ground;
    bool has_ground_position_ = false;
    double lat_deg_ = 0.0;
    double lon_deg_ = 0.0;
    std::vector<double> times_;
    std::vector<std::array<double, 3>> positions_;
};

} // namespace lcbinint::obs
