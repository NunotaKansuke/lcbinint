#pragma once

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

// Observatory/telescope position, stored internally in degrees.
// Used for terrestrial parallax (sets lcbi_params::obs_lat, ::obs_lon).
class Site {
public:
    Site(double lat_deg, double lon_deg)
        : lat_deg_(lat_deg), lon_deg_(lon_deg) {}

    double lat_deg() const noexcept { return lat_deg_; }
    double lon_deg() const noexcept { return lon_deg_; }

private:
    double lat_deg_;
    double lon_deg_;
};

} // namespace lcbinint::obs
