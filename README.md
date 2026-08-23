# AFM Pore Size

Measure complete circular pores in rendered AFM images and export ImageJ-style
tables, annotated images, and masks.

The tool does not hard-code a scale-bar value. It detects the scale-bar line in
the image and combines that pixel length with the value you provide, such as
`--scale-bar-value 4 --scale-bar-unit um`.

## Requirements

Python 3.10+ with:

```powershell
pip install -r requirements.txt
```

## Usage

Analyze a directory of AFM TIFF files and use a prompted scale-bar value:

```powershell
python scripts/analyze_afm_pores.py "C:\path\to\afm_images" --output afm_results --scale-bar-value 4 --scale-bar-unit um --method blob
```

If the automatic scale-bar detector chooses the wrong line, restrict it to a
region:

```powershell
python scripts/analyze_afm_pores.py "C:\path\to\image.spm.tif" --scale-bar-value 500 --scale-bar-unit nm --scale-bar-roi 400,520,130,40
```

If you already know the calibration:

```powershell
python scripts/analyze_afm_pores.py "C:\path\to\afm_images" --unit-per-px 0.0377 --unit um
```

## Outputs

The output directory contains:

- `pores_detail.csv`: one row per complete circular pore.
- `pores_summary.csv`: per-image counts and pore diameter summary.
- `pores_results.xlsx`: Excel workbook with `Summary` and `Detail` sheets.
- `annotated/*_pores.png`: detected pores and the scale bar overlay.
- `masks/*_mask.png`: binary candidate mask used for measurement.

The main CSV fields follow ImageJ/Fiji names where practical: `Area`,
`Perim.`, `Circ.`, `Feret`, `FeretX`, `FeretY`, `FeretAngle`, `MinFeret`,
`AR`, `Round`, and `Solidity`. Pixel fields such as `Feret_px` and
`CenterX_px` are always included.

## Complete-Circle Criteria

A pore is counted only when it is a closed, round region that does not touch the
effective image boundary. The defaults reject truncated edge pores, elongated
ellipses, low-circularity regions, and fragmented or merged objects.

Useful tuning flags:

```powershell
--min-diameter-px 20
--max-diameter-px 150
--edge-margin-px 18
--min-circularity 0.75
--min-round 0.80
--max-ar 1.25
--min-solidity 0.90
--polarity dark
```

## Self-Test

```powershell
python scripts/analyze_afm_pores.py --self-test --output test_results
python -m unittest discover -s tests
```
