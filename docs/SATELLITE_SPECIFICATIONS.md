# DeforestNet - Satellite Sensor & Spectral Band Technical Specifications

> Comprehensive reference document for Sentinel-1 SAR and Sentinel-2 MSI sensor
> specifications, derived vegetation indices, and multi-sensor fusion rationale as
> used in the DeforestNet deforestation detection system.

---

## Table of Contents

1. [Sentinel-2 MSI Optical Bands](#1-sentinel-2-msi-optical-bands)
2. [Sentinel-1 C-SAR Bands](#2-sentinel-1-c-sar-bands)
3. [Derived Spectral and Radar Indices](#3-derived-spectral-and-radar-indices)
4. [Multi-Sensor Fusion Rationale](#4-multi-sensor-fusion-rationale)
5. [Mission-Level Specifications](#5-mission-level-specifications)
6. [DeforestNet Band Stack Architecture](#6-deforestnet-band-stack-architecture)
7. [References](#7-references)

---

## 1. Sentinel-2 MSI Optical Bands

The Multi-Spectral Instrument (MSI) aboard Sentinel-2 captures 13 spectral bands
spanning the visible, near-infrared (NIR), and short-wave infrared (SWIR) regions.
DeforestNet uses four 10-meter resolution bands: B2, B3, B4, and B8.

### 1.1 Instrument Overview

| Parameter                         | Value                                  |
|-----------------------------------|----------------------------------------|
| Instrument                        | Multi-Spectral Instrument (MSI)        |
| Total spectral bands              | 13                                     |
| Spatial resolutions               | 10 m (4 bands), 20 m (6 bands), 60 m (3 bands) |
| Swath width                       | 290 km                                 |
| Radiometric resolution            | 12-bit (0-4095 DN), stored as 16-bit   |
| Absolute radiometric uncertainty  | < 5% (goal 3%)                         |
| Inter-band radiometric uncertainty| 3%                                     |
| Multi-temporal radiometric uncertainty | 1%                                 |
| Linearity knowledge accuracy      | 1%                                     |
| Channel cross-talk                | < 0.5%                                 |
| Wavelength knowledge uncertainty  | < 1 nm                                 |
| Field of view                     | 20.6 degrees                           |
| Detectors per focal plane         | 12 (staggered configuration)           |

### 1.2 Complete Spectral Band Table

| Band  | Name           | Central Wavelength (nm) | Bandwidth (nm) | Spatial Resolution (m) | SNR @ Lref (S2A/S2B) | SNR @ Lref (S2C) |
|-------|----------------|------------------------|-----------------|------------------------|----------------------|-------------------|
| B01   | Aerosol        | 443                    | 20              | 60                     | 129                  | 894               |
| **B02** | **Blue**     | **490**                | **65**          | **10**                 | **154**              | **162**           |
| **B03** | **Green**    | **560**                | **35**          | **10**                 | **168**              | **189**           |
| **B04** | **Red**      | **665**                | **30**          | **10**                 | **142**              | **175**           |
| B05   | Red Edge 1     | 705                    | 15              | 20                     | 117                  | 192               |
| B06   | Red Edge 2     | 740                    | 15              | 20                     | 89                   | 169               |
| B07   | Red Edge 3     | 783                    | 20              | 20                     | 105                  | 161               |
| **B08** | **NIR**      | **842**                | **115**         | **10**                 | **174**              | **174**           |
| B8A   | Narrow NIR     | 865                    | 20              | 20                     | 72                   | 117               |
| B09   | Water Vapour   | 945                    | 20              | 60                     | 114                  | 135               |
| B10   | Cirrus         | 1375                   | 30              | 60                     | 50                   | 301               |
| B11   | SWIR 1         | 1610                   | 90              | 20                     | 100                  | 133               |
| B12   | SWIR 2         | 2190                   | 180             | 20                     | 100                  | 141               |

**Bolded rows** indicate bands used in DeforestNet.

### 1.3 Band-by-Band Detail (DeforestNet Bands)

#### B2 - Blue (490 nm)

| Parameter              | Value                              |
|------------------------|------------------------------------|
| Central wavelength     | 490 nm                             |
| Wavelength range       | 458 - 523 nm (65 nm bandwidth)     |
| Spatial resolution     | 10 m                               |
| SNR at Lref            | 154 (S2A/S2B), 162 (S2C)          |
| Radiometric resolution | 12-bit                             |

**What it detects:** The blue band captures reflected energy in the short-wavelength
visible spectrum. It is sensitive to atmospheric scattering (Rayleigh scattering is
strongest at shorter wavelengths), water body turbidity, and subtle differences in
soil and rock mineralogy.

**Deforestation relevance:**
- Discriminates soil types and exposed earth from vegetation cover
- Essential input to the Enhanced Vegetation Index (EVI), where it corrects for
  atmospheric aerosol influences and soil background reflectance
- Detects sediment-laden water bodies associated with illegal mining operations
- Forest type mapping and identification of man-made clearings
- Bare earth exposed by logging or land clearing shows elevated blue reflectance
  compared to intact canopy

#### B3 - Green (560 nm)

| Parameter              | Value                              |
|------------------------|------------------------------------|
| Central wavelength     | 560 nm                             |
| Wavelength range       | 543 - 578 nm (35 nm bandwidth)     |
| Spatial resolution     | 10 m                               |
| SNR at Lref            | 168 (S2A/S2B), 189 (S2C)          |
| Radiometric resolution | 12-bit                             |

**What it detects:** The green band captures the green reflectance peak of
vegetation. Healthy chlorophyll-containing leaves reflect strongly in this region
while absorbing in the adjacent blue and red bands. The green band also provides
excellent contrast between clear and turbid water.

**Vegetation applications:**
- Indicates vegetation vigor; the "green peak" reflectance increases with leaf
  chlorophyll concentration up to moderate levels
- Useful for distinguishing vegetation types (crops vs. natural forest)
- Highlights actively photosynthesizing canopy versus senescent or stressed
  vegetation
- Contributes to true-color and false-color composites for visual interpretation
- Differentiates agricultural regrowth from intact primary forest based on spectral
  shape

#### B4 - Red (665 nm)

| Parameter              | Value                              |
|------------------------|------------------------------------|
| Central wavelength     | 665 nm                             |
| Wavelength range       | 650 - 680 nm (30 nm bandwidth)     |
| Spatial resolution     | 10 m                               |
| SNR at Lref            | 142 (S2A/S2B), 175 (S2C)          |
| Radiometric resolution | 12-bit                             |

**What it detects:** The red band targets the chlorophyll absorption maximum.
Healthy green vegetation absorbs strongly in the red region for photosynthesis,
resulting in very low reflectance. Dead, stressed, or absent vegetation reflects
significantly more red light.

**Chlorophyll absorption and deforestation relevance:**
- **Chlorophyll absorption:** Chlorophyll-a and chlorophyll-b have a strong
  absorption feature centered near 660-680 nm. Healthy canopy absorbs 80-90% of
  incoming red light. This makes the red band the most sensitive single indicator
  of chlorophyll presence or absence.
- **Deforestation detection:** When forest is cleared, red reflectance increases
  dramatically as exposed soil, dead foliage, or bare earth replaces absorbing
  canopy. This abrupt spectral change is the foundation of NDVI-based change
  detection.
- Dead foliage reflects strongly in the red, making it a key discriminator between
  healthy forest and recently logged or burned areas
- Core input to NDVI, EVI, and SAVI indices
- Distinguishes soil types and urban features from vegetation cover

#### B8 - NIR (842 nm)

| Parameter              | Value                              |
|------------------------|------------------------------------|
| Central wavelength     | 842 nm                             |
| Wavelength range       | 785 - 900 nm (115 nm bandwidth)    |
| Spatial resolution     | 10 m                               |
| SNR at Lref            | 174 (S2A/S2B), 174 (S2C)          |
| Radiometric resolution | 12-bit                             |

**What it detects:** The near-infrared band captures energy reflected by the
internal leaf mesophyll structure. Unlike visible wavelengths that interact with
pigments, NIR photons scatter within the spongy mesophyll cell walls and air-cell
interfaces inside leaves, resulting in very high reflectance (40-60%) from healthy
vegetation.

**Vegetation health and canopy density:**
- **Canopy density:** NIR reflectance is strongly correlated with Leaf Area Index
  (LAI) and canopy density. Multi-layered tropical forest canopies produce the
  highest NIR reflectance due to multiple scattering between leaf layers.
- **Vegetation health:** Water-stressed or nutrient-deficient vegetation shows
  reduced NIR reflectance due to collapsed cell structure.
- Biomass estimation: NIR reflectance scales with above-ground biomass up to
  moderate density levels.
- Core input to all three vegetation indices (NDVI, EVI, SAVI) as the numerator
  component representing vegetation presence.
- The wide bandwidth (115 nm) provides high SNR but lower spectral selectivity
  compared to the narrow NIR band B8A (20 nm bandwidth).
- Mapping shorelines and water-land boundaries, since water absorbs strongly in
  the NIR.

---

## 2. Sentinel-1 C-SAR Bands

Sentinel-1 carries a C-band Synthetic Aperture Radar (C-SAR) instrument that
provides all-weather, day-and-night imaging capability.

### 2.1 Instrument Overview

| Parameter                    | Value                                      |
|------------------------------|--------------------------------------------|
| Instrument                   | C-band Synthetic Aperture Radar (C-SAR)    |
| Center frequency             | 5.405 GHz                                  |
| Wavelength                   | ~5.55 cm                                   |
| Bandwidth                    | 0-100 MHz (programmable)                   |
| Polarization modes           | Dual-pol: VV+VH, HH+HV; Single: VV, HH   |
| Antenna dimensions           | 12.3 m x 0.821 m                           |
| RF peak power                | 4.368 kW (4.075 kW in IW dual-pol)         |
| Pulse width                  | 5-100 microseconds (programmable)          |
| Pulse Repetition Frequency   | 1,000-3,000 Hz (programmable)              |
| Receiver noise figure        | 3.2 dB                                     |
| Maximum NESZ                 | -22 dB (all modes)                         |
| Radiometric accuracy         | 1.0 dB (3-sigma)                           |
| Radiometric stability        | 0.5 dB (3-sigma)                           |
| Ambiguity ratio              | -22 dB or better (distributed targets)     |
| Instrument mass              | 945 kg                                     |
| Data storage (end-of-life)   | 1,410 Gb                                   |

### 2.2 How SAR Works

Synthetic Aperture Radar is an active microwave sensor that transmits its own
electromagnetic pulses and records the backscattered signal. Key principles:

1. **Active illumination:** Unlike optical sensors that rely on sunlight, SAR
   generates its own microwave pulses at C-band (5.405 GHz / 5.55 cm wavelength).
   This enables imaging regardless of solar illumination -- day or night.

2. **Synthetic aperture:** The radar antenna physically measures 12.3 m, but by
   coherently combining returns over the satellite's forward motion, the processor
   synthesizes an effective aperture of several kilometers, achieving fine azimuth
   resolution (5 m in IW mode).

3. **Range resolution:** Determined by the transmitted bandwidth. In IW mode, the
   range resolution is approximately 20 m (single look).

4. **Backscatter coefficient (sigma-nought):** The measured quantity, expressed in
   decibels (dB). It depends on surface roughness, dielectric properties, geometry,
   and the presence of volume scatterers (vegetation canopy).

### 2.3 Why SAR Penetrates Clouds

Microwave radiation at C-band (5.55 cm wavelength) is approximately 100,000 times
longer than visible light wavelengths (~0.5 micrometers). Electromagnetic waves
interact most strongly with objects comparable to or larger than their wavelength.
Cloud droplets (1-100 micrometers) and rain droplets (0.1-5 mm for most tropical
rainfall) are far smaller than the C-band wavelength, so:

- Cloud droplets cause negligible scattering or absorption at 5.55 cm
- Light to moderate rain has minimal effect on C-band signals
- Only extreme precipitation (> 50 mm/hr) causes measurable attenuation
- Smoke and haze from forest fires are also transparent to C-band

This makes SAR indispensable for monitoring tropical forests, where persistent
cloud cover renders optical imagery unusable for 60-90% of the year in regions like
the Amazon basin, Congo basin, and Southeast Asia.

### 2.4 VV Polarization (Co-polarization)

| Parameter                | Value                                    |
|--------------------------|------------------------------------------|
| Polarization             | Vertical transmit, Vertical receive      |
| Configuration in stack   | Channel index 0                          |
| Typical value range (dB) | -25 to 0 dB                             |

**What VV measures:** The radar transmits a vertically polarized wave and receives
the vertically polarized component of the backscatter. This is called
co-polarization because the transmit and receive polarizations are identical.

**Physical interpretation:**
- VV backscatter is dominated by **surface scattering** from the ground surface and
  **double-bounce scattering** between vertical structures and the ground plane
  (e.g., tree trunks and soil)
- Over forest: VV backscatter is moderate to high (-8 to -4 dB) due to both canopy
  volume scattering and trunk-ground double bounce
- Over bare soil: VV is sensitive to surface roughness and soil moisture content;
  smooth dry soil produces low backscatter (-15 to -10 dB)
- Over water: Very low backscatter (< -20 dB) for calm water due to specular
  reflection away from the sensor
- Urban areas: Very high backscatter (> -5 dB) from corner reflectors formed by
  building walls and ground surfaces

### 2.5 VH Polarization (Cross-polarization)

| Parameter                | Value                                    |
|--------------------------|------------------------------------------|
| Polarization             | Vertical transmit, Horizontal receive    |
| Configuration in stack   | Channel index 1                          |
| Typical value range (dB) | -30 to -5 dB                            |

**What VH measures:** The radar transmits a vertically polarized wave but receives
the horizontally polarized component. Cross-polarized returns require a change in
the polarization plane of the scattered wave, which occurs through:

- **Volume scattering:** Multiple interactions within a randomly oriented three-
  dimensional medium (tree branches, leaves, twigs). This is the dominant mechanism
  in vegetated areas and the reason VH is the most important SAR channel for
  vegetation monitoring.
- **Depolarization** by rough surfaces or complex geometries

**Sensitivity to vegetation structure and biomass:**
- VH backscatter is directly proportional to canopy volume, density, and structural
  complexity. Dense multi-layered tropical forests produce the highest VH values.
- VH is the most sensitive SAR polarization to above-ground biomass (AGB) up to
  approximately 100-150 t/ha for C-band.
- Deforestation causes a sharp drop in VH backscatter (typically 3-6 dB decrease)
  because the volume scattering medium (canopy) is removed.
- Forest degradation (selective logging) produces a detectable but smaller VH
  decrease (1-3 dB).
- VH has lower sensitivity to soil moisture than VV, making it a more stable
  indicator of vegetation change.

### 2.6 Acquisition Modes

| Mode                       | Resolution (single look) | Swath Width | Incidence Angle | Polarization Options      |
|----------------------------|--------------------------|-------------|-----------------|---------------------------|
| **Interferometric Wide Swath (IW)** | **5 m x 20 m** | **250 km** | **29.1 - 46.0 deg** | **Dual (VV+VH, HH+HV) or Single** |
| Stripmap (SM)              | 5 m x 5 m               | 80 km       | 18.3 - 46.8 deg | Dual or Single            |
| Extra Wide Swath (EW)      | 20 m x 40 m             | 410 km      | 18.9 - 47.0 deg | Dual or Single            |
| Wave (WV)                  | 5 m x 5 m (vignettes)   | 20 x 20 km  | 21.6 - 38.0 deg | Single                    |

**IW mode (used for DeforestNet):**
- The default and primary acquisition mode over land surfaces globally
- Uses Terrain Observation with Progressive Scans SAR (TOPSAR) technique
- Acquires data in 3 sub-swaths with burst synchronization for interferometry
- Azimuth steering angle: +/- 0.6 degrees
- Provides consistent, wide-area coverage ideal for systematic forest monitoring

### 2.7 Why SAR is Critical for Tropical Forests

Tropical forests present a fundamental challenge for optical monitoring:

1. **Cloud cover:** Tropical regions experience 60-90% cloud cover on average.
   The Amazon basin has fewer than 20 cloud-free days per year in many locations.
   The Congo basin and insular Southeast Asia face similar conditions.

2. **Seasonal variability:** Cloud cover peaks during the wet season, which
   coincides with periods of active deforestation (roads become impassable in the
   heaviest rains, but clearing activity often occurs at the onset and offset of
   wet periods).

3. **Smoke and haze:** Slash-and-burn practices create persistent smoke plumes
   that further obscure optical imagery during peak deforestation periods.

4. **SAR solution:** Sentinel-1 C-SAR provides guaranteed cloud-penetrating,
   smoke-penetrating, day-and-night imaging capability with a 6-day revisit cycle
   (constellation). This ensures continuous monitoring without temporal gaps caused
   by atmospheric interference.

5. **Direct structural sensitivity:** SAR backscatter responds directly to the
   physical structure of the canopy (branch density, canopy height, trunk-ground
   geometry), providing complementary information to the optical spectral
   reflectance from pigments and cell structure.

---

## 3. Derived Spectral and Radar Indices

DeforestNet computes five derived indices from the six raw bands, bringing the
total feature stack to 11 channels. These indices enhance spectral separability
between land cover classes and encode domain-specific knowledge about vegetation
biophysics.

### 3.1 NDVI - Normalized Difference Vegetation Index

**Formula:**

```
NDVI = (NIR - Red) / (NIR + Red)
     = (B8 - B4) / (B8 + B4)
```

| Parameter            | Value                              |
|----------------------|------------------------------------|
| Value range          | -1.0 to +1.0                       |
| Channel index        | 6 (in DeforestNet 11-band stack)   |
| Input bands          | B8 (842 nm), B4 (665 nm)           |

**Physical basis:** NDVI exploits the sharp contrast between strong chlorophyll
absorption in the red band and strong mesophyll scattering in the NIR band. Healthy
vegetation absorbs most red light and reflects most NIR light, producing high NDVI
values. Non-vegetated surfaces reflect similar amounts in both bands, producing
values near zero.

**Value interpretation and thresholds:**

| NDVI Range     | Interpretation                                          |
|----------------|---------------------------------------------------------|
| -1.0 to -0.1  | Water bodies (water absorbs NIR strongly)               |
| -0.1 to +0.1  | Barren surfaces: rock, sand, snow, bare soil            |
| +0.1 to +0.2  | Very sparse vegetation, recently cleared land           |
| +0.2 to +0.4  | Shrubland, grassland, degraded/sparse canopy            |
| +0.4 to +0.6  | Moderate vegetation: crops, open woodland, young regrowth |
| +0.6 to +0.8  | Dense vegetation: healthy forest, mature canopy         |
| +0.8 to +1.0  | Very dense, healthy tropical/temperate rainforest       |

**Deforestation thresholds:**
- Intact dense forest typically: NDVI > 0.6
- Active deforestation (cleared land): NDVI drops to 0.1-0.3
- An abrupt NDVI decrease of > 0.3 between consecutive observations is a strong
  deforestation signal
- Post-fire areas: NDVI typically 0.05-0.15

**Limitations:**
- Saturates over dense canopy (LAI > 3-4), losing sensitivity to biomass
  differences between moderately dense and very dense forest
- Sensitive to atmospheric effects (aerosols, water vapor)
- Sensitive to soil background reflectance in sparse canopy conditions
- Cannot distinguish deforestation causes (fire, logging, mining)

### 3.2 EVI - Enhanced Vegetation Index

**Formula:**

```
EVI = G * (NIR - Red) / (NIR + C1*Red - C2*Blue + L)
    = 2.5 * (B8 - B4) / (B8 + 6*B4 - 7.5*B2 + 1)
```

| Parameter          | Value                               |
|--------------------|-------------------------------------|
| G (gain factor)    | 2.5                                 |
| C1 (red coeff.)    | 6                                   |
| C2 (blue coeff.)   | 7.5                                 |
| L (canopy/soil)    | 1                                   |
| Value range        | -1.0 to +1.0                        |
| Typical vegetation | 0.2 to 0.8                          |
| Channel index      | 7 (in DeforestNet 11-band stack)    |
| Input bands        | B8 (842 nm), B4 (665 nm), B2 (490 nm) |

**Advantages over NDVI:**

1. **Atmospheric correction:** The blue band (B2) is incorporated to correct for
   atmospheric aerosol scattering. Aerosols preferentially scatter blue light, so
   the C2*Blue term compensates for this influence. This is particularly important
   in tropical regions where biomass burning produces persistent haze.

2. **Soil background correction:** The L factor in the denominator reduces
   sensitivity to soil brightness variations, analogous to SAVI but integrated into
   a more comprehensive correction scheme.

3. **Reduced saturation in dense canopy:** EVI maintains sensitivity to vegetation
   differences at high LAI values (LAI > 4) where NDVI saturates. This makes EVI
   superior for monitoring dense tropical forests, which is exactly the DeforestNet
   use case.

4. **Better discrimination of forest density gradients:** In the Amazon basin
   specifically, EVI has been shown to detect seasonal phenological variations in
   dense forest canopy that NDVI cannot resolve.

**Value interpretation:**

| EVI Range      | Interpretation                                          |
|----------------|---------------------------------------------------------|
| < 0            | Water, snow, or cloud shadow                            |
| 0.0 to 0.1    | Bare soil, rock, urban surfaces                         |
| 0.1 to 0.2    | Sparse vegetation, recently cleared                     |
| 0.2 to 0.4    | Grassland, shrubland, open canopy                       |
| 0.4 to 0.6    | Moderate forest, cropland, young regrowth               |
| 0.6 to 0.8    | Dense healthy forest                                    |
| > 0.8          | Very dense tropical rainforest at peak greenness        |

### 3.3 SAVI - Soil Adjusted Vegetation Index

**Formula:**

```
SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)
     = ((B8 - B4) / (B8 + B4 + L)) * (1 + L)
```

where L = 0.5 (standard value).

| Parameter            | Value                              |
|----------------------|------------------------------------|
| L (soil factor)      | 0.5 (standard; range 0 to 1)      |
| Value range          | -1.0 to +1.0                       |
| Channel index        | 8 (in DeforestNet 11-band stack)   |
| Input bands          | B8 (842 nm), B4 (665 nm)           |

**The soil brightness correction factor (L):**
- Developed by Huete (1988) as a transformation to minimize soil brightness
  influences on vegetation indices
- L ranges from 0 to 1:
  - L = 0: Equivalent to NDVI (no soil correction)
  - L = 0.5: Standard value suitable for intermediate vegetation density
  - L = 1.0: Maximum soil correction, suitable for very sparse vegetation
- The L parameter shifts the soil line in NIR-Red space, reducing the differential
  soil brightness effect on the vegetation index

**When SAVI is most useful:**
- Transitional zones between forest and cleared land, where mixed pixels contain
  both vegetation and exposed soil
- Early regrowth monitoring after deforestation, where sparse seedlings grow over
  exposed soil
- Agricultural frontier regions with partial canopy cover
- Areas with high soil color variability (laterite/red soils common in tropical
  deforestation zones)
- SAVI is particularly valuable in the DeforestNet context because deforestation
  boundaries inherently contain mixed vegetation-soil pixels

**Value interpretation:**

| SAVI Range     | Interpretation                                          |
|----------------|---------------------------------------------------------|
| < 0            | Water bodies                                            |
| 0.0 to 0.1    | Bare soil, urban, rock                                  |
| 0.1 to 0.25   | Very sparse vegetation over exposed soil                |
| 0.25 to 0.45  | Moderate vegetation, mixed pixels at forest edges       |
| 0.45 to 0.65  | Dense vegetation, forest canopy with some gaps          |
| > 0.65         | Closed canopy forest                                    |

### 3.4 VV/VH Ratio (SAR Polarization Ratio)

**Formula:**

```
VV/VH Ratio = VV / VH
```

(Computed on linear power scale, not dB. In DeforestNet, the ratio is normalized
by dividing by 20 and clipping to [0, 1].)

| Parameter            | Value                              |
|----------------------|------------------------------------|
| Value range (raw)    | ~1 to 20+ (linear scale)           |
| Value range (norm.)  | 0 to 1 (in DeforestNet stack)      |
| Channel index        | 9 (in DeforestNet 11-band stack)   |
| Input bands          | VV, VH                             |

**What the ratio indicates:**

The VV/VH ratio characterizes the dominant scattering mechanism:

| VV/VH Ratio (linear) | Dominant Mechanism                | Typical Surface           |
|-----------------------|-----------------------------------|---------------------------|
| High (> 8)           | Surface scattering dominates      | Bare soil, smooth surfaces, water |
| Moderate (3-8)       | Mix of surface and volume         | Sparse vegetation, degraded forest, agriculture |
| Low (1-3)            | Volume scattering dominates       | Dense forest canopy       |

**Interpretation for vegetation vs. bare soil:**
- **Dense forest:** VH is relatively high (strong volume scattering), so VV/VH
  is low. The canopy's random branch/leaf structure efficiently cross-polarizes
  the transmitted signal.
- **Bare soil:** VH is very low (minimal depolarization from a smooth surface),
  so VV/VH is high.
- **Deforestation event:** VV/VH increases sharply when canopy is removed, because
  VH drops more dramatically than VV (volume scattering medium is eliminated).
- **Agriculture:** Intermediate values that vary with crop growth stage.

### 3.5 RVI - Radar Vegetation Index

**Formula (dual-pol adaptation for Sentinel-1, RVI4S1):**

```
q   = VH / VV
m   = (1 - q) / (1 + q)
beta = 1 / (1 + q)
RVI = 1 - (m * beta)
    = q * (q + 3) / (q + 1)^2
```

Simplified approximation used in DeforestNet:
```
RVI = 4 * VH / (VV + VH)
```

| Parameter                | Value                              |
|--------------------------|------------------------------------|
| Value range              | 0 to 1                             |
| Channel index            | 10 (in DeforestNet 11-band stack)  |
| Input bands              | VV, VH                             |

**Physical basis:**
- RVI quantifies the degree of depolarization (randomness) in the backscattered
  signal, which is a proxy for vegetation volume scattering
- A pure point target (bare soil, corner reflector) produces fully polarized returns
  where RVI approaches 0
- A fully random scattering medium (dense canopy with randomly oriented branches)
  produces completely depolarized returns where RVI approaches 1

**Vegetation structure sensitivity:**

| RVI Range      | Interpretation                                          |
|----------------|---------------------------------------------------------|
| 0.0 to 0.2    | Bare soil, water, smooth surfaces                       |
| 0.2 to 0.4    | Sparse vegetation, early growth stage, recently cleared |
| 0.4 to 0.6    | Moderate canopy, open woodland, agriculture             |
| 0.6 to 0.8    | Dense vegetation, closed canopy forest                  |
| 0.8 to 1.0    | Very dense, structurally complex forest                 |

**Applications:**
- Temporal monitoring of canopy development and loss
- Separates vegetated terrain from urban areas and bare soil
- Invariant to crop type, applicable globally
- Sensitive to canopy height, branch density, and leaf mass
- Caveat: rough soil (post-tillage) or windy water surfaces can inflate RVI values
  due to surface depolarization effects

---

## 4. Multi-Sensor Fusion Rationale

### 4.1 Why Combining Optical + SAR is Superior

The fusion of Sentinel-2 optical data with Sentinel-1 SAR data provides advantages
that neither sensor can achieve alone:

| Limitation of Single Sensor              | How Fusion Addresses It                                |
|------------------------------------------|--------------------------------------------------------|
| Optical fails under clouds               | SAR provides cloud-penetrating observations            |
| SAR lacks spectral color information     | Optical provides 4+ spectral bands for pigment analysis|
| Optical cannot measure canopy structure  | SAR backscatter encodes branch/trunk geometry          |
| SAR is noisy (speckle)                   | Optical provides clean spatial detail for edges        |
| Optical has daytime-only constraint      | SAR operates day and night                             |
| Optical NDVI saturates in dense canopy   | SAR VH continues to scale with biomass                 |
| Single-sensor temporal gaps              | Combined revisit: observations every 3-6 days          |

### 4.2 Cloud Penetration Advantage

In tropical forest regions, cloud cover statistics demonstrate the necessity of SAR:

| Region                    | Annual Cloud-Free Days | Cloud Cover % |
|---------------------------|------------------------|---------------|
| Central Amazon basin      | < 20                   | > 85%         |
| Congo basin               | 30-50                  | 75-85%        |
| Borneo / Sumatra          | 20-40                  | 80-90%        |
| Central America (wet)     | 40-60                  | 70-80%        |

Without SAR, optical-only monitoring systems have temporal gaps of weeks to months
during peak cloud seasons, during which deforestation can proceed undetected.

### 4.3 Temporal Complementarity

- **Sentinel-2 revisit:** 5 days at equator (2-satellite constellation), but
  effective usable revisit is 15-30+ days in cloudy tropics after filtering
  cloud-contaminated scenes
- **Sentinel-1 revisit:** 6 days (2-satellite constellation), with nearly 100%
  usable observations regardless of weather
- **Combined effective revisit:** A new usable observation every 3-5 days on
  average, with SAR filling all cloud-induced gaps

### 4.4 How the 11-Band Stack Provides Richer Features

Standard imagery approaches and their limitations:

| Approach             | Channels | Limitations                                     |
|----------------------|----------|--------------------------------------------------|
| RGB only             | 3        | No NIR, no vegetation indices, no SAR, cloud-dependent |
| Standard 4-band (RGBN)| 4      | No SAR, no derived indices, cloud-dependent      |
| Sentinel-2 only (4 bands) | 4  | Cloud-dependent, no structural information       |
| Sentinel-1 only (2 bands) | 2  | No spectral information, speckle noise           |
| **DeforestNet 11-band** | **11** | **Full spectral + structural + derived features** |

The DeforestNet 11-band stack provides:

1. **SAR channels (2):** VV and VH provide cloud-independent structural information
   about canopy density, trunk-ground interactions, and surface roughness

2. **Optical channels (4):** B2, B3, B4, B8 provide spectral information about
   chlorophyll content, vegetation vigor, soil exposure, and water presence

3. **Derived indices (5):** NDVI, EVI, SAVI encode vegetation biophysics with
   varying sensitivity ranges and correction factors; VV/VH Ratio and RVI encode
   SAR-derived vegetation structure metrics

This multi-source, multi-index approach gives the U-Net model access to redundant
and complementary features. The model can learn that:
- A simultaneous drop in NDVI + VH indicates clear-cutting with high confidence
- High VV/VH + low NDVI + high B4 indicates bare soil exposure
- Low NDVI + high B2 + specific SAR signature identifies mining with water ponds
- Regular spatial patterns in EVI + specific SAVI values indicate agriculture

### 4.5 Scientific Basis and Key References

The multi-sensor fusion approach for forest monitoring is supported by extensive
peer-reviewed literature:

1. **Reiche et al. (2015)** - "Fusing Landsat and SAR time series to detect
   deforestation in the tropics." International Journal of Applied Earth
   Observation and Geoinformation, 38, 29-38. Demonstrated that SAR+optical fusion
   reduces detection latency by 1-2 months compared to optical-only systems.

2. **Reiche et al. (2018)** - "Improving near-real time deforestation monitoring in
   tropical dry forests by combining dense Sentinel-1 time series with Landsat and
   ALOS-2 PALSAR-2." Remote Sensing of Environment, 204, 147-161.

3. **Shimada et al. (2014)** - "New global forest/non-forest maps from ALOS PALSAR
   data (2007-2010)." Remote Sensing of Environment, 155, 13-31. Established SAR
   forest mapping baselines.

4. **Hansen et al. (2013)** - "High-resolution global maps of 21st-century forest
   cover change." Science, 342(6160), 850-853. Foundational optical forest
   monitoring work.

5. **Mercier et al. (2019)** - "Evaluation of Sentinel-1 & 2 time series for
   predicting wheat and rapeseed phenological stages." ISPRS Journal of
   Photogrammetry and Remote Sensing, 163, 231-256. Demonstrated SAR+optical
   complementarity for vegetation monitoring.

6. **ESA Sentinel-1 Toolbox** documentation and **Copernicus Global Land Service**
   technical documentation provide operational fusion methodologies.

---

## 5. Mission-Level Specifications

### 5.1 Sentinel-2 Mission

| Parameter                         | Value                                      |
|-----------------------------------|--------------------------------------------|
| Programme                         | Copernicus (EU/ESA)                        |
| Current operational satellites    | Sentinel-2B, Sentinel-2C                   |
| Extended campaign                 | Sentinel-2A (since 2025)                   |
| Launch dates                      | S2A: June 2015; S2B: March 2017; S2C: September 2024 |
| Orbit type                        | Sun-synchronous                            |
| Mean altitude                     | 786 km                                     |
| Inclination                       | 98.62 degrees                              |
| Orbital period                    | 100.6 minutes                              |
| Repeat cycle (single satellite)   | 10 days                                    |
| Revisit time (2-satellite)        | **5 days at the equator**                  |
| Local time at descending node     | 10:30 AM                                   |
| Swath width                       | 290 km                                     |
| Ground-track deviation            | +/- 2 km                                   |
| Design lifespan                   | 7.25 years (consumables for 12 years)      |
| Satellite mass                    | ~1,200 kg each                             |

**Geographical coverage:**
- Continental land surfaces between 56 deg S and 82.8 deg N
- Coastal waters up to 20 km from shore
- Islands greater than 100 km squared, plus all EU islands
- Mediterranean Sea and all enclosed seas
- Additional areas per Copernicus service requests (e.g., Antarctica)

### 5.2 Sentinel-1 Mission

| Parameter                         | Value                                      |
|-----------------------------------|--------------------------------------------|
| Programme                         | Copernicus (EU/ESA)                        |
| Current satellites                | Sentinel-1A (decommissioned 2024), S1B (failed 2021), S1C (launched Dec 2024) |
| Orbit type                        | Sun-synchronous, dawn-dusk                 |
| Altitude                          | 693 km                                     |
| Inclination                       | 98.18 degrees                              |
| Orbital period                    | 98.6 minutes                               |
| Repeat cycle (single satellite)   | 12 days (175 orbits)                       |
| Revisit time (2-satellite)        | **6 days** (equator); **< 1 day** at Arctic |
| Revisit over Europe/Canada        | 1-3 days                                   |
| Local time at ascending node      | 18:00                                      |
| Orbital tube                      | +/- 120 m (RMS)                            |
| Spacecraft mass at launch         | ~2,300 kg (including 130 kg fuel)          |
| Downlink rate                     | 2 x 260 Mbps; optical link via EDRS at 520 Mbps |

**Heritage:** ERS-1/2, Envisat ASAR, Radarsat missions.

### 5.3 Copernicus Programme and Data Policy

| Aspect                   | Details                                            |
|--------------------------|----------------------------------------------------|
| Programme name           | Copernicus (formerly GMES)                         |
| Managed by               | European Commission, implemented by ESA            |
| Data policy              | **Free, full, and open access** to all citizens and organizations worldwide |
| Primary data portal      | Copernicus Data Space Ecosystem (dataspace.copernicus.eu) |
| Additional portals       | WEkEO (wekeo.eu), EUMETSAT, NASA Earth Data        |
| Archive                  | Complete archive from each satellite's launch date |
| Processing levels        | Level-0, Level-1 (TOA reflectance/GRD), Level-2 (surface reflectance/ARD) |
| Sentinel-2 Level-2A      | Systematic global production since December 13, 2018 |
| Update frequency         | Near real-time (< 3 hours for NRT, < 24 hours systematic) |
| Access methods           | Web portal, OData API, STAC API, S3 object storage |
| Cloud computing          | Processing available in co-located cloud environments |

---

## 6. DeforestNet Band Stack Architecture

### 6.1 Channel Configuration

The DeforestNet model ingests an 11-channel tensor of shape (11, 256, 256) per
sample, organized as follows:

| Index | Band Name    | Source       | Type              | Value Range (normalized) |
|-------|-------------|--------------|-------------------|--------------------------|
| 0     | VV          | Sentinel-1   | SAR co-pol        | 0.0 - 1.0               |
| 1     | VH          | Sentinel-1   | SAR cross-pol     | 0.0 - 1.0               |
| 2     | B2 (Blue)   | Sentinel-2   | Optical           | 0.0 - 1.0               |
| 3     | B3 (Green)  | Sentinel-2   | Optical           | 0.0 - 1.0               |
| 4     | B4 (Red)    | Sentinel-2   | Optical           | 0.0 - 1.0               |
| 5     | B8 (NIR)    | Sentinel-2   | Optical           | 0.0 - 1.0               |
| 6     | NDVI        | Derived      | Vegetation index  | 0.0 - 1.0 (remapped)    |
| 7     | EVI         | Derived      | Vegetation index  | 0.0 - 1.0 (remapped)    |
| 8     | SAVI        | Derived      | Vegetation index  | 0.0 - 1.0 (remapped)    |
| 9     | VV/VH Ratio | Derived      | SAR ratio         | 0.0 - 1.0 (normalized)  |
| 10    | RVI         | Derived      | Radar veg. index  | 0.0 - 1.0               |

### 6.2 Classification Targets

The model performs 6-class semantic segmentation:

| Class ID | Name           | Description                    | Spectral Signature                |
|----------|----------------|--------------------------------|-----------------------------------|
| 0        | Forest         | Healthy forest cover           | High NIR, high NDVI, moderate VV  |
| 1        | Logging        | Logging activity               | High Red, low NDVI, moderate SAR  |
| 2        | Mining         | Mining operations              | High Blue, low vegetation indices |
| 3        | Agriculture    | Agricultural expansion         | Regular patterns, moderate NDVI   |
| 4        | Fire           | Burnt areas                    | Very low all bands, very low NDVI |
| 5        | Infrastructure | Roads, buildings               | Gray/uniform, high VV (double bounce) |

### 6.3 Resolution and Spatial Context

| Parameter                | Value                                    |
|--------------------------|------------------------------------------|
| Native Sentinel-2 resolution (B2,B3,B4,B8) | 10 m per pixel      |
| Native Sentinel-1 IW resolution | 5 m x 20 m (resampled to 10 m)    |
| Model input size         | 256 x 256 pixels                         |
| Ground coverage per tile | 2.56 km x 2.56 km (at 10 m resolution)  |
| Patch stride             | 128 pixels (50% overlap)                 |

---

## 7. References

1. ESA. (2024). "Sentinel-2 MSI Technical Guide." Copernicus Sentinel Online.
   https://sentiwiki.copernicus.eu/web/s2-mission

2. ESA. (2024). "Sentinel-1 SAR Technical Guide." Copernicus Sentinel Online.
   https://sentiwiki.copernicus.eu/web/s1-mission

3. Sentinel Hub. (2024). "Sentinel-2 Bands." Custom Scripts.
   https://custom-scripts.sentinel-hub.com/sentinel-2/bands/

4. Huete, A. R. (1988). "A soil-adjusted vegetation index (SAVI)." Remote Sensing
   of Environment, 25(3), 295-309.

5. Huete, A. R., et al. (2002). "Overview of the radiometric and biophysical
   performance of the MODIS vegetation indices." Remote Sensing of Environment,
   83(1-2), 195-213. [EVI formulation]

6. Mandal, D., et al. (2020). "Dual polarimetric radar vegetation index for crop
   growth monitoring using Sentinel-1 SAR data." Remote Sensing of Environment,
   247, 111954. [RVI4S1 methodology]

7. Rouse, J. W., et al. (1974). "Monitoring vegetation systems in the Great Plains
   with ERTS." Third Earth Resources Technology Satellite-1 Symposium, NASA SP-351,
   309-317. [Original NDVI formulation]

8. Reiche, J., et al. (2015). "Fusing Landsat and SAR time series to detect
   deforestation in the tropics." International Journal of Applied Earth Observation
   and Geoinformation, 38, 29-38.

9. Reiche, J., et al. (2018). "Improving near-real time deforestation monitoring
   in tropical dry forests." Remote Sensing of Environment, 204, 147-161.

10. Hansen, M. C., et al. (2013). "High-resolution global maps of 21st-century
    forest cover change." Science, 342(6160), 850-853.

11. Copernicus Programme. (2024). "Access to Data." European Commission.
    https://www.copernicus.eu/en/access-data

---

*Document generated for the DeforestNet project. All satellite specifications are
sourced from official ESA/Copernicus documentation and peer-reviewed literature.
Specifications are current as of the S2C era (2024-2025).*
