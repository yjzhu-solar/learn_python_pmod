# IDL → Python (NumPy + SunPy) Cheatsheet

A quick, practical mapping of common IDL/SSW tasks to Python using **NumPy**, **Astropy**, **SunPy**, and a few SciPy helpers. Snippets are drop-in starters.

---

## Setup

```python
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.time import Time
import astropy.units as u

import sunpy.map as smap
from sunpy.coordinates import frames
from sunpy.time import parse_time

# Optional helpers
from scipy import ndimage, signal, interpolate
```

---

## Arrays, creation, shape, dtype

| Task                | IDL                             | Python / NumPy                                               |
| ------------------- | ------------------------------- | ------------------------------------------------------------ |
| 1D/2D integer range | `indgen(n)` / `indgen(n,m)`     | `np.arange(n, dtype=np.int64)` / `np.indices((n,m))[0]`      |
| 1D/2D float range   | `findgen(n)`                    | `np.arange(n, dtype=float)`                                  |
| Zeros/Ones          | `fltarr(n,m)` / `intarr(n)`     | `np.zeros((n,m), float)` / `np.ones(n, int)`                 |
| Random uniform      | `randomu(seed, n)`              | `np.random.default_rng().random(n)`                          |
| Size / shape        | `n_elements(a)`, `size(a,/dim)` | `a.size`, `a.shape`, `a.ndim`                                |
| Type                | `size(a,/type)`                 | `a.dtype`                                                    |
| Reshape             | `reform(a, nx, ny)`             | `a.reshape(ny, nx)`                                          |
| Transpose           | `transpose(a)`                  | `a.T` or `np.transpose(a, axes)`                             |
| Reverse             | `reverse(a, dim)`               | `np.flip(a, axis=dim)`                                       |
| Concatenate         | `[a,b]`, `[[a],[b]]`            | `np.concatenate([a,b], axis=...)` / `np.vstack`, `np.hstack` |

---

## Indexing, slicing, logicals

| Task                | IDL                              | Python / NumPy                           |
| ------------------- | -------------------------------- | ---------------------------------------- |
| Slice               | `a[0:10:*]`                      | `a[0:10:step]`                           |
| Last element        | `a[-1]`                          | `a[-1]`                                  |
| Conditional indices | `where(a gt 0, n)`               | `idx = np.where(a > 0); n = idx[0].size` |
| Masking             | `a[where(mask)] = !values.f_nan` | `a[mask] = np.nan`                       |
| Any/All             | `(total(a ne 0) gt 0)`           | `np.any(a != 0)`, `np.all(a == 0)`       |
| Unique              | `uniq(a)`                        | `np.unique(a)`                           |

---

## Statistics & reductions

| Task         | IDL                                        | Python / NumPy                                    |
| ------------ | ------------------------------------------ | ------------------------------------------------- |
| Sum/Mean/Std | `total(a, /nan)` / `mean(a)` / `stddev(a)` | `np.nansum(a)` / `np.nanmean(a)` / `np.nanstd(a)` |
| Min/Max      | `min(a, max=maxv)`                         | `np.nanmin(a)`, `np.nanmax(a)`                    |
| Histogram    | `histogram(a, bins=nbin)`                  | `np.histogram(a, bins=nbin)`                      |
| Percentile   | `percentile(a, p)`                         | `np.nanpercentile(a, p)`                          |

---

## Scaling, interpolation, rebinning

| Task                  | IDL                              | Python                                                                 |
| --------------------- | -------------------------------- | ---------------------------------------------------------------------- |
| Scale to byte         | `bytscl(a, min=, max=, top=255)` | `(a - vmin)/(vmax-vmin); img = np.clip(255*_,0,255).astype(np.uint8)`  |
| Rebin (block average) | `rebin(a, nx, ny)`               | `a.reshape(ny, f, nx, f).mean((1,3))` *(see snippet)*                  |
| Interpolate at points | `interpolate(a, x, y)`           | `ndimage.map_coordinates(a, [y, x], order=1)`                          |
| Resample to new shape | `congrid(a, newdims)`            | `ndimage.zoom(a, zoom_factors, order=1)` or `skimage.transform.resize` |

**Block-average rebin example** (scale by integer factor `f`):

```python
def block_reduce_mean(a, f):
    ny, nx = a.shape
    a = a[:ny - ny % f, :nx - nx % f]  # crop to multiple
    return a.reshape(a.shape[0]//f, f, a.shape[1]//f, f).mean(axis=(1,3))
```

---

## Convolution, filtering, morphology

| Task              | IDL                              | Python                                                       |
| ----------------- | -------------------------------- | ------------------------------------------------------------ |
| 1D/2D convolution | `convol(a, kernel)`              | `signal.convolve2d(a, kernel, mode='same', boundary='symm')` |
| Gaussian smooth   | `smooth(a, n)` or `gauss_smooth` | `ndimage.gaussian_filter(a, sigma)`                          |
| Median filter     | `median(a, n)`                   | `ndimage.median_filter(a, size)`                             |
| Sobel/Canny edges | SSW/`sobel()`                    | `ndimage.sobel(a)` / `skimage.feature.canny(a)`              |
| Label regions     | `label_region`                   | `ndimage.label(a > thr)`                                     |

---

## FFTs

| Task                   | IDL                  | Python / NumPy                                  |
| ---------------------- | -------------------- | ----------------------------------------------- |
| 1D/2D FFT              | `fft(a)`             | `np.fft.fft(a)`, `np.fft.fft2(a)`               |
| Shift zero-freq center | `shift(fft(a), ...)` | `np.fft.fftshift(...)`, `np.fft.ifftshift(...)` |
| Power spectrum         | `abs(fft(a))^2`      | `np.abs(np.fft.fft2(a))**2`                     |

---

## Logic for NaNs & masks

```python
mask = ~np.isfinite(a)           # NaN/Inf mask
a_filled = np.where(np.isfinite(a), a, 0.0)
good = np.isfinite(a) & (a > 0)
```

---

## Plotting (quick matches)

| IDL                      | Python (Matplotlib)                                    |
| ------------------------ | ------------------------------------------------------ |
| `plot,x,y`               | `plt.plot(x,y); plt.show()`                            |
| `plot,x,y, psym=1`       | `plt.scatter(x,y)`                                     |
| `contour, a, levels=...` | `plt.contour(a, levels=...)`                           |
| `tv, a` / `tvscl,a`      | `plt.imshow(a, origin='lower', cmap='gray')`           |
| `oplot,x,y2`             | `plt.plot(x,y2)`                                       |
| `colorbar`               | `plt.colorbar()`                                       |
| `save`                   | `plt.savefig("fig.png", dpi=300, bbox_inches='tight')` |

---

## FITS I/O

| Task       | IDL                                     | Python                                                  |
| ---------- | --------------------------------------- | ------------------------------------------------------- |
| Read FITS  | `readfits, file, data, hdr` / `mrdfits` | `data = fits.getdata(file); hdr = fits.getheader(file)` |
| Write FITS | `writefits, file, data, hdr`            | `fits.writeto(file, data, hdr, overwrite=True)`         |

---

## Time handling

| Task          | IDL                             | Python                                                               |
| ------------- | ------------------------------- | -------------------------------------------------------------------- |
| Now           | `systime(/utc)`                 | `Time.now()`                                                         |
| Parse string  | `anytim('2022-10-24T12:00:00')` | `parse_time('2022-10-24T12:00:00')` or `Time('2022-10-24T12:00:00')` |
| Convert scale | `utc→tai`                       | `Time(..., scale='utc').tai`                                         |
| Format        | `time_string(t,/ecs)`           | `t.isot`, `t.to_value('isot')`                                       |

---

## SunPy essentials

### Load a solar map (e.g., AIA FITS)

```python
m = smap.Map("aia_lev1.fits")
m.peek()  # quicklook
```

### Plot with overlays

```python
fig = plt.figure()
ax = plt.subplot(projection=m)
m.plot(axes=ax)
m.draw_limb()
m.draw_grid(grid_spacing=10*u.deg)
plt.colorbar()
```

### Submap (cropping by world coords or pixels)

```python
# by world coords (HPC)
from astropy.coordinates import SkyCoord
bl = SkyCoord(-200*u.arcsec, -200*u.arcsec, frame=m.coordinate_frame)
tr = SkyCoord( 200*u.arcsec,  200*u.arcsec, frame=m.coordinate_frame)
m_sub = m.submap(bl, top_right=tr)

# by pixels
m_sub2 = m.submap((x0, x1)*u.pix, (y0, y1)*u.pix)
```

### Resample / Reproject

```python
# Resample in pixels
m2 = m.resample((512, 512)*u.pix)

# Reproject to another map’s WCS (needs reproject package)
# pip install reproject
from reproject import reproject_interp
m_target = m2  # e.g., target header/WCS
out, _ = reproject_interp(m, m_target.wcs, m_target.data.shape)
m_reproj = smap.Map(out, m_target.meta)
```

### Cutouts around a coordinate

```python
c = SkyCoord(100*u.arcsec, -200*u.arcsec, frame=m.coordinate_frame)
width = height = 300*u.arcsec
m_cut = m.submap(c, width=width, height=height)
```

### Map meta → NumPy array

```python
data = m.data        # NumPy array
hdr  = m.meta        # dict-like header
```

---

## Coordinates (Sun-centered)

| Task                   | IDL/SSW idea      | Python (Astropy/SunPy)                                                                     |
| ---------------------- | ----------------- | ------------------------------------------------------------------------------------------ |
| Create HPC coordinate  | `xy2lonlat` (SSW) | `SkyCoord(x, y, frame=frames.Helioprojective, obstime=t, observer='earth', unit=u.arcsec)` |
| Transform frames       | `hpc→hgs/hgc/hci` | `c.transform_to(frames.HeliographicStonyhurst)` / `.HeliocentricInertial`                  |
| Observer at spacecraft | `get_stony`       | `SkyCoord(..., observer=observer_coord)` or `observer='SOHO'` (with `obstime`)             |
| Angle/Separation       | `angsep`          | `c1.separation(c2)`                                                                        |

**Example: HPC → HGS**

```python
t = parse_time("2022-10-24T12:00:00")
c_hpc = SkyCoord(250*u.arcsec, -100*u.arcsec,
                 frame=frames.Helioprojective, obstime=t, observer='earth')
c_hgs = c_hpc.transform_to(frames.HeliographicStonyhurst(obstime=t))
lon, lat = c_hgs.lon, c_hgs.lat
```

---

## Common SSW/IDL image ops → Python

| IDL / SSW                           | Python                                                 |
| ----------------------------------- | ------------------------------------------------------ |
| `rot` / `rot_image` (rotate by deg) | `ndimage.rotate(a, angle_deg, reshape=False, order=1)` |
| `shift(a, dx, dy)`                  | `ndimage.shift(a, (dy, dx), order=1)`                  |
| `deriv`                             | `np.gradient(a)`                                       |
| `boxcar_smooth`                     | `ndimage.uniform_filter(a, size)`                      |
| `median(a, s)`                      | `ndimage.median_filter(a, size=s)`                     |
| `sobel(a)`                          | `ndimage.sobel(a)`                                     |

---

## Reading SDO/AIA, GOES, etc. (SunPy Fido)

```python
from sunpy.net import Fido, attrs as a

# AIA 193 for a time range
tr = a.Time("2022-10-24 12:00", "2022-10-24 12:05")
res = Fido.search(tr, a.Instrument.aia, a.Wavelength(193*u.angstrom))
files = Fido.fetch(res)
maps = [smap.Map(f) for f in files]
```

```python
# GOES XRS flux
res = Fido.search(tr, a.Instrument.goes, a.goes.SatelliteNumber(16))
files = Fido.fetch(res)
```

---

## Masks, NaNs, and SunPy maps

```python
m_masked = smap.Map(np.where(condition, m.data, np.nan), m.meta)
m_masked.plot(); plt.show()
```

---

## Practical snippets you’ll use a lot

### Percent clip + byte scale (IDL `bytscl`-like)

```python
def scale_to_byte(a, pmin=1, pmax=99):
    vmin, vmax = np.nanpercentile(a, [pmin, pmax])
    s = (a - vmin) / (vmax - vmin)
    return np.clip(255 * s, 0, 255).astype(np.uint8)
```

### 2-D power spectrum (IDL feel)

```python
def power_spectrum_2d(img):
    f = np.fft.fft2(img)
    f = np.fft.fftshift(f)
    return (np.abs(f)**2)
```

### Fast “where & count”

```python
idx = np.nonzero(a > thr)     # same as np.where
count = idx[0].size
```

### Safe division (avoid IDL’s divide-by-zero)

```python
def safediv(a, b):
    out = np.full_like(a, np.nan, dtype=float)
    m = b != 0
    out[m] = a[m] / b[m]
    return out
```

---

## Performance notes (IDL mindset → Python)

* Prefer **vectorization** over Python loops (as in IDL).
* Use `numexpr` or `np.einsum` for complex expressions on large arrays.
* `ndimage` & `signal` are compiled and fast for filtering/convolution.
* For very large FITS, memory-map: `fits.open(file, memmap=True)`.

---

## Quick “gotchas” when switching

* Index order: NumPy uses `(row, col)` = `(y, x)` like IDL, but be mindful when mixing with WCS (`x` first in headers).
* Integer division: Python `//` is integer floor; use `/` for float.
* Broadcasting is powerful—embrace it instead of `rebin`, `reform` hacks.
* NaN-aware reductions: use `np.nan*` variants.

---

## Minimal end-to-end example

```python
# Load AIA map, make a limb+grid plot, make a cutout, rebin, and save
m = smap.Map("aia_lev1.fits")
fig = plt.figure()
ax = plt.subplot(projection=m)
m.plot(axes=ax); m.draw_limb(); m.draw_grid(grid_spacing=10*u.deg)
plt.savefig("aia_overview.png", dpi=200, bbox_inches="tight")

# Cutout 400x400 arcsec around disk center
c = SkyCoord(0*u.arcsec, 0*u.arcsec, frame=m.coordinate_frame)
m_cut = m.submap(c, width=400*u.arcsec, height=400*u.arcsec)

# Rebin by 2 using block mean
rebinned = block_reduce_mean(m_cut.data, 2)
m_reb = smap.Map(rebinned, m_cut.meta)   # meta still approximates OK
m_reb.save("aia_cut_rebinned.fits", overwrite=True)
```

---

## Handy package equivalents

| IDL/SSW                     | Python package                  |
| --------------------------- | ------------------------------- |
| `readfits/mrdfits`, `sxpar` | `astropy.io.fits`               |
| SSW time utils              | `astropy.time`, `sunpy.time`    |
| Mapping/projections         | `sunpy.map`, `reproject`        |
| Filters/conv                | `scipy.ndimage`, `scipy.signal` |
| Morphology                  | `scipy.ndimage`, `scikit-image` |
| Optimization                | `scipy.optimize`                |

---

If you want, I can tailor this to your most used IDL routines (e.g., `rebin`, `congrid`, SSW coordinate transforms, EIS/SPICE helpers) and add one-liners you can paste into notebooks.
