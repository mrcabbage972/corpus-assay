# Brand assets

| File | Use |
| --- | --- |
| `logo.svg` | Horizontal logo for light backgrounds (README default) |
| `logo-dark.svg` | Horizontal logo for dark backgrounds |
| `icon.svg` | Standalone icon |
| `icon-mono.svg` | Single-color icon |
| `icon-dark.svg` | Icon on a dark tile |
| `favicon.svg` | Simplified icon for small sizes (favicons, ≤ 32 px) |
| `social-preview.png` | 1280×640 GitHub social preview |

The icon shows a benchmark item (light blue) and a corpus document (blue) whose
overlap (amber) is the contamination being measured, framed by brackets with a
ruler.

## Palette

| Role | Hex |
| --- | --- |
| Primary blue (corpus token) | `#3B82F6` |
| Light blue (benchmark token) | `#93C5FD` |
| Overlap accent | `#F59E0B` |
| Navy (text, brackets) | `#0F172A` |
| Tagline (light backgrounds) | `#64748B` |

## Typography

The wordmark is set in [Poppins](https://github.com/itfoundry/Poppins), Bold for
`corpus-assay` and Medium for the tagline. Poppins is licensed under the SIL Open
Font License 1.1. The text in the SVGs is converted to outlines, so no font is needed
to display them.

`make_logos.py` regenerates every SVG. See its docstring for the font files it needs.
